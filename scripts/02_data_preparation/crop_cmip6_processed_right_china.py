# --- centralized path bootstrap (see config/project_paths.json) ---
import sys as _sys
from pathlib import Path as _Path
_SCRIPTS_DIR = _Path(__file__).resolve().parents[1]
_OWN_DIR = _Path(__file__).resolve().parent
for _dir in (_OWN_DIR, _SCRIPTS_DIR / "common", _SCRIPTS_DIR / "03_bias_correction", _SCRIPTS_DIR / "04_energy_metrics"):
    if _dir.is_dir() and str(_dir) not in _sys.path:
        _sys.path.insert(0, str(_dir))
from project_paths import PROJECT_ROOT, get_path  # noqa: E402
from area_weights import china_intersection_weights  # noqa: E402
# -------------------------------------------------------------------
import argparse
from pathlib import Path

import numpy as np
import xarray as xr


BASE_DIR = get_path("data_raw_cmip6_downloaded")

INPUT_ROOT = (
    get_path("data_interim_cmip6_processed_16models")
)

OUTPUT_ROOT = (
    get_path("data_interim_cmip6_china_clipped")
)

MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "AWI-CM-1-1-MR",
    "BCC-CSM2-MR",
    "CanESM5",
    "CESM2-WACCM",
    "CMCC-CM2-SR5",
    "CMCC-ESM2",
    "FGOALS-f3-L",
    "FIO-ESM-2-0",
    "IPSL-CM6A-LR",
    "KACE-1-0-G",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "TaiESM1",
]

EXPERIMENTS = [
    "historical",
    "ssp126",
    "ssp245",
    "ssp585",
]

VARIABLES = [
    "tas",
    "sfcWind",
    "rsds",
]

LAT_MIN = 15.0
LAT_MAX = 54.0

LON_MIN = 70.0
LON_MAX = 140.0

SHAPEFILE = get_path("china_shapefile")

EXPECTED_MONTHS = {
    "historical": 672,
    "ssp126": 1032,
    "ssp245": 1032,
    "ssp585": 1032,
}

EXPECTED_PERIODS = {
    "historical": (
        1959,
        1,
        2014,
        12,
    ),
    "ssp126": (
        2015,
        1,
        2100,
        12,
    ),
    "ssp245": (
        2015,
        1,
        2100,
        12,
    ),
    "ssp585": (
        2015,
        1,
        2100,
        12,
    ),
}


def get_time_coder():

    return xr.coders.CFDatetimeCoder(
        use_cftime=True
    )


def find_input_file(
    model,
    experiment,
    variable,
):

    directory = (
        INPUT_ROOT
        / model
        / experiment
        / variable
    )

    files = sorted(
        directory.glob("*.nc")
    )

    if len(files) != 1:

        raise ValueError(
            f"{model} | {experiment} | {variable}: "
            f"expected 1 input file, found {len(files)}"
        )

    return files[0]


def make_output_file(
    model,
    experiment,
    variable,
    input_file,
):

    output_directory = (
        OUTPUT_ROOT
        / model
        / experiment
        / variable
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        output_directory
        / input_file.name
    )


def get_year_month_ids(
    ds,
):

    years = (
        ds[
            "time"
        ]
        .dt.year
        .values
        .astype(
            np.int64
        )
    )

    months = (
        ds[
            "time"
        ]
        .dt.month
        .values
        .astype(
            np.int64
        )
    )

    return (
        years * 12
        + months
        - 1
    )


def check_time(
    ds,
    experiment,
):

    expected_count = EXPECTED_MONTHS[
        experiment
    ]

    actual_count = ds.sizes[
        "time"
    ]

    if actual_count != expected_count:

        raise ValueError(
            f"Expected {expected_count} months, "
            f"found {actual_count}"
        )

    (
        start_year,
        start_month,
        end_year,
        end_month,
    ) = EXPECTED_PERIODS[
        experiment
    ]

    actual_start_year = int(
        ds[
            "time"
        ].dt.year.values[
            0
        ]
    )

    actual_start_month = int(
        ds[
            "time"
        ].dt.month.values[
            0
        ]
    )

    actual_end_year = int(
        ds[
            "time"
        ].dt.year.values[
            -1
        ]
    )

    actual_end_month = int(
        ds[
            "time"
        ].dt.month.values[
            -1
        ]
    )

    if (
        actual_start_year,
        actual_start_month,
    ) != (
        start_year,
        start_month,
    ):

        raise ValueError(
            f"Unexpected start month: "
            f"{actual_start_year:04d}-"
            f"{actual_start_month:02d}"
        )

    if (
        actual_end_year,
        actual_end_month,
    ) != (
        end_year,
        end_month,
    ):

        raise ValueError(
            f"Unexpected end month: "
            f"{actual_end_year:04d}-"
            f"{actual_end_month:02d}"
        )

    month_ids = get_year_month_ids(
        ds
    )

    if len(
        np.unique(
            month_ids
        )
    ) != expected_count:

        raise ValueError(
            "Duplicate year-month entries detected"
        )

    expected_ids = np.arange(
        month_ids[
            0
        ],
        month_ids[
            0
        ]
        + expected_count,
    )

    if not np.array_equal(
        month_ids,
        expected_ids,
    ):

        raise ValueError(
            "Time sequence is not continuous"
        )


def check_coordinates(
    ds,
):

    if "lat" not in ds.coords:

        raise ValueError(
            "Latitude coordinate is missing"
        )

    if "lon" not in ds.coords:

        raise ValueError(
            "Longitude coordinate is missing"
        )

    if ds[
        "lat"
    ].ndim != 1:

        raise ValueError(
            "Latitude is not one-dimensional"
        )

    if ds[
        "lon"
    ].ndim != 1:

        raise ValueError(
            "Longitude is not one-dimensional"
        )


def crop_spatial_domain(
    ds,
):

    # Keep the minimal contiguous lat/lon index range that fully encloses every
    # cell whose box intersects the China boundary polygon.  This retains edge
    # cells that straddle the boundary (e.g. CanESM5's cell centred at
    # 54.4162N, whose southern half overlaps China's northernmost tip) without
    # any fillna, neighbour-copy, or extra interpolation, while dropping
    # pure-ocean halo cells that carry zero intersection area.
    ds = ds.sortby(
        "lat"
    ).sortby(
        "lon"
    )

    weights = china_intersection_weights(
        ds["lat"].values,
        ds["lon"].values,
        SHAPEFILE,
    )

    intersects = weights[
        "china_intersects"
    ]

    if not intersects.any():

        raise ValueError(
            "No cell intersects the China boundary"
        )

    lat_indices, lon_indices = np.where(
        intersects
    )

    lat_slice = slice(
        int(lat_indices.min()),
        int(lat_indices.max()) + 1,
    )

    lon_slice = slice(
        int(lon_indices.min()),
        int(lon_indices.max()) + 1,
    )

    return ds.isel(
        lat=lat_slice,
        lon=lon_slice,
    )


def check_spatial_domain(
    ds,
):

    if ds.sizes[
        "lat"
    ] == 0:

        raise ValueError(
            "No latitude points remain after cropping"
        )

    if ds.sizes[
        "lon"
    ] == 0:

        raise ValueError(
            "No longitude points remain after cropping"
        )

    lat_values = ds[
        "lat"
    ].values

    lon_values = ds[
        "lon"
    ].values

    if not np.all(
        np.diff(
            lat_values
        ) > 0
    ):

        raise ValueError(
            "Latitude is not strictly ascending"
        )

    if not np.all(
        np.diff(
            lon_values
        ) > 0
    ):

        raise ValueError(
            "Longitude is not strictly ascending"
        )

    # Completeness: the cropped grid must still cover the full China boundary
    # (intersection area / boundary area ~= 1).  A ratio below this threshold
    # means an intersecting cell was dropped by the crop.
    weights = china_intersection_weights(
        lat_values,
        lon_values,
        SHAPEFILE,
    )

    boundary_area = float(
        weights["china_boundary_area_km2"]
    )

    intersection_area = float(
        weights["china_intersection_area_km2"].sum()
    )

    coverage = (
        intersection_area / boundary_area
        if boundary_area > 0.0
        else np.nan
    )

    if coverage < 0.9999:

        raise ValueError(
            "Cropped grid does not fully cover the China boundary "
            f"(coverage ratio {coverage:.6f})"
        )


def check_variable(
    ds,
    variable,
):

    if variable not in ds.data_vars:

        raise ValueError(
            f"{variable} is missing"
        )

    da = ds[
        variable
    ]

    required_dims = {
        "time",
        "lat",
        "lon",
    }

    if not required_dims.issubset(
        set(
            da.dims
        )
    ):

        raise ValueError(
            f"Unexpected dimensions for {variable}: "
            f"{da.dims}"
        )


def write_output(
    ds,
    variable,
    output_file,
):

    encoding = {
        variable: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
    }

    if output_file.exists():

        output_file.unlink()

    ds.to_netcdf(
        output_file,
        engine="netcdf4",
        encoding=encoding,
    )


def verify_saved_file(
    output_file,
    experiment,
    variable,
):

    time_coder = get_time_coder()

    ds = xr.open_dataset(
        output_file,
        decode_times=time_coder,
    )

    try:

        check_coordinates(
            ds
        )

        check_time(
            ds,
            experiment,
        )

        check_variable(
            ds,
            variable,
        )

        check_spatial_domain(
            ds
        )

    finally:

        ds.close()


def process_group(
    model,
    experiment,
    variable,
):

    input_file = find_input_file(
        model,
        experiment,
        variable,
    )

    output_file = make_output_file(
        model,
        experiment,
        variable,
        input_file,
    )

    time_coder = get_time_coder()

    ds = xr.open_dataset(
        input_file,
        decode_times=time_coder,
    )

    cropped = None

    try:

        check_coordinates(
            ds
        )

        check_time(
            ds,
            experiment,
        )

        check_variable(
            ds,
            variable,
        )

        original_lat_count = ds.sizes[
            "lat"
        ]

        original_lon_count = ds.sizes[
            "lon"
        ]

        cropped = crop_spatial_domain(
            ds[
                [
                    variable
                ]
            ]
        )

        check_time(
            cropped,
            experiment,
        )

        check_variable(
            cropped,
            variable,
        )

        check_spatial_domain(
            cropped
        )

        cropped.load()

        write_output(
            cropped,
            variable,
            output_file,
        )

        verify_saved_file(
            output_file,
            experiment,
            variable,
        )

        result = {
            "input_lat": original_lat_count,
            "input_lon": original_lon_count,
            "output_lat": cropped.sizes[
                "lat"
            ],
            "output_lon": cropped.sizes[
                "lon"
            ],
            "lat_min": float(
                cropped[
                    "lat"
                ].min().values
            ),
            "lat_max": float(
                cropped[
                    "lat"
                ].max().values
            ),
            "lon_min": float(
                cropped[
                    "lon"
                ].min().values
            ),
            "lon_max": float(
                cropped[
                    "lon"
                ].max().values
            ),
            "output_file": output_file,
        }

        return result

    finally:

        if cropped is not None:

            cropped.close()

        ds.close()


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Crop processed CMIP6 files to the minimal contiguous lat/lon "
            "index range enclosing every China-intersecting cell."
        )
    )

    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Optional subset of model names to process (default: all models "
            "in the script's MODELS list)."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    models = (
        args.models
        if args.models
        else MODELS
    )

    print()
    print(
        "=" * 120
    )

    print(
        "CMIP6 CHINA DOMAIN CROPPING"
    )

    print(
        "=" * 120
    )

    print(
        f"Input root: {INPUT_ROOT}"
    )

    print(
        f"Output root: {OUTPUT_ROOT}"
    )

    print(
        f"Reference domain (informational): "
        f"lat {LAT_MIN} to {LAT_MAX}, "
        f"lon {LON_MIN} to {LON_MAX}"
    )

    print(
        f"Shapefile: {SHAPEFILE}"
    )

    print(
        "Crop rule: minimal contiguous range enclosing all "
        "China-intersecting cells"
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_groups = (
        len(
            models
        )
        *
        len(
            EXPERIMENTS
        )
        *
        len(
            VARIABLES
        )
    )

    success_count = 0
    failed_count = 0

    failed_groups = []

    model_grid_info = {}

    group_number = 0

    for model in models:

        for experiment in EXPERIMENTS:

            for variable in VARIABLES:

                group_number += 1

                print()
                print(
                    "-" * 120
                )

                print(
                    f"GROUP "
                    f"{group_number}/"
                    f"{total_groups}"
                )

                print(
                    f"{model} | "
                    f"{experiment} | "
                    f"{variable}"
                )

                print(
                    "-" * 120
                )

                try:

                    result = process_group(
                        model,
                        experiment,
                        variable,
                    )

                    success_count += 1

                    grid_key = model

                    current_grid = (
                        result[
                            "output_lat"
                        ],
                        result[
                            "output_lon"
                        ],
                        result[
                            "lat_min"
                        ],
                        result[
                            "lat_max"
                        ],
                        result[
                            "lon_min"
                        ],
                        result[
                            "lon_max"
                        ],
                    )

                    if grid_key not in model_grid_info:

                        model_grid_info[
                            grid_key
                        ] = current_grid

                    else:

                        if (
                            model_grid_info[
                                grid_key
                            ]
                            != current_grid
                        ):

                            raise ValueError(
                                f"Inconsistent cropped grid "
                                f"within model {model}"
                            )

                    print(
                        f"Input grid: "
                        f"{result['input_lat']} x "
                        f"{result['input_lon']}"
                    )

                    print(
                        f"Output grid: "
                        f"{result['output_lat']} x "
                        f"{result['output_lon']}"
                    )

                    print(
                        f"Latitude: "
                        f"{result['lat_min']:.6f} "
                        f"to "
                        f"{result['lat_max']:.6f}"
                    )

                    print(
                        f"Longitude: "
                        f"{result['lon_min']:.6f} "
                        f"to "
                        f"{result['lon_max']:.6f}"
                    )

                    print(
                        "Status: OK"
                    )

                except Exception as exc:

                    failed_count += 1

                    failed_groups.append(
                        (
                            model,
                            experiment,
                            variable,
                            f"{type(exc).__name__}: {exc}",
                        )
                    )

                    print(
                        "Status: FAILED"
                    )

                    print(
                        f"Error: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

    print()
    print(
        "=" * 120
    )

    print(
        "MODEL GRID SUMMARY"
    )

    print(
        "=" * 120
    )

    for model in models:

        if model not in model_grid_info:

            continue

        (
            lat_count,
            lon_count,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
        ) = model_grid_info[
            model
        ]

        print()
        print(
            model
        )

        print(
            f"  grid: "
            f"{lat_count} x {lon_count}"
        )

        print(
            f"  latitude: "
            f"{lat_min:.6f} "
            f"to "
            f"{lat_max:.6f}"
        )

        print(
            f"  longitude: "
            f"{lon_min:.6f} "
            f"to "
            f"{lon_max:.6f}"
        )

    print()
    print(
        "=" * 120
    )

    print(
        "CROPPING SUMMARY"
    )

    print(
        "=" * 120
    )

    print(
        f"Groups: {total_groups}"
    )

    print(
        f"Success: {success_count}"
    )

    print(
        f"Failed: {failed_count}"
    )

    if failed_groups:

        print()
        print(
            "FAILED GROUPS"
        )

        for (
            model,
            experiment,
            variable,
            error,
        ) in failed_groups:

            print(
                f"{model} | "
                f"{experiment} | "
                f"{variable} | "
                f"{error}"
            )

    else:

        print()
        print(
            "ALL CMIP6 CHINA DOMAIN FILES PASSED"
        )


if __name__ == "__main__":
    main()