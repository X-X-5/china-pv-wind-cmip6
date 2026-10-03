# --- centralized path bootstrap (see config/project_paths.json) ---
import sys as _sys
from pathlib import Path as _Path
_SCRIPTS_DIR = _Path(__file__).resolve().parents[1]
_OWN_DIR = _Path(__file__).resolve().parent
for _dir in (_OWN_DIR, _SCRIPTS_DIR / "common", _SCRIPTS_DIR / "03_bias_correction", _SCRIPTS_DIR / "04_energy_metrics"):
    if _dir.is_dir() and str(_dir) not in _sys.path:
        _sys.path.insert(0, str(_dir))
from project_paths import PROJECT_ROOT, get_path  # noqa: E402
# -------------------------------------------------------------------
from pathlib import Path

import numpy as np
import xarray as xr


BASE_DIR = PROJECT_ROOT

MODEL = "ACCESS-CM2"

ERA5_FILE = (
    PROJECT_ROOT / "archive" / "needs_review" / "era5_old_processed"
    / MODEL
    / f"ERA5_{MODEL}_199401-201412.nc"
)

CMIP6_ROOT = (
    get_path("data_interim_cmip6_processed_16models")
    / MODEL
    / "historical"
)

VARIABLES = [
    "tas",
    "sfcWind",
    "rsds",
]


def find_cmip6_file(variable):

    variable_dir = (
        CMIP6_ROOT
        / variable
    )

    files = sorted(
        variable_dir.glob(
            "*.nc"
        )
    )

    if len(files) != 1:

        raise ValueError(
            f"Expected exactly one file for "
            f"{variable}, found {len(files)}"
        )

    return files[0]


def normalize_longitude(ds):

    lon = ds[
        "lon"
    ]

    if float(
        lon.max()
    ) > 180.0:

        new_lon = (
            (lon + 180.0) % 360.0
            - 180.0
        )

        ds = ds.assign_coords(
            lon=new_lon
        )

        ds = ds.sortby(
            "lon"
        )

    return ds


def crop_to_era5_grid(
    ds,
    era5,
):

    lat_min = float(
        era5[
            "lat"
        ].min()
    )

    lat_max = float(
        era5[
            "lat"
        ].max()
    )

    lon_min = float(
        era5[
            "lon"
        ].min()
    )

    lon_max = float(
        era5[
            "lon"
        ].max()
    )

    lat_mask = (
        (ds["lat"] >= lat_min)
        &
        (ds["lat"] <= lat_max)
    )

    lon_mask = (
        (ds["lon"] >= lon_min)
        &
        (ds["lon"] <= lon_max)
    )

    ds = ds.sel(
        lat=ds[
            "lat"
        ][
            lat_mask
        ],
        lon=ds[
            "lon"
        ][
            lon_mask
        ],
    )

    return ds


def get_month_ids(ds):

    years = ds[
        "time"
    ].dt.year.values

    months = ds[
        "time"
    ].dt.month.values

    return (
        years * 12
        + months
        - 1
    )


def print_stats(
    label,
    da,
):

    finite = np.isfinite(
        da.values
    )

    total_count = (
        da.size
    )

    finite_count = int(
        finite.sum()
    )

    nan_count = (
        total_count
        - finite_count
    )

    nan_fraction = (
        nan_count
        / total_count
    )

    print(
        f"  {label}"
    )

    print(
        f"    shape: "
        f"{da.shape}"
    )

    print(
        f"    units: "
        f"{da.attrs.get('units', '')}"
    )

    print(
        f"    mean: "
        f"{float(da.mean(skipna=True).values):.6f}"
    )

    print(
        f"    min: "
        f"{float(da.min(skipna=True).values):.6f}"
    )

    print(
        f"    max: "
        f"{float(da.max(skipna=True).values):.6f}"
    )

    print(
        f"    finite: "
        f"{finite_count}"
    )

    print(
        f"    nan: "
        f"{nan_count}"
    )

    print(
        f"    nan fraction: "
        f"{nan_fraction:.8f}"
    )


def main():

    print(
        "=" * 100
    )

    print(
        f"QM PAIR CHECK: {MODEL}"
    )

    print(
        "=" * 100
    )

    era5 = xr.open_dataset(
        ERA5_FILE,
        engine="netcdf4",
        decode_times=True,
    )

    try:

        print()
        print(
            "ERA5-on-GCM grid:"
        )

        print(
            f"  time: "
            f"{era5.sizes['time']}"
        )

        print(
            f"  lat: "
            f"{era5.sizes['lat']}"
        )

        print(
            f"  lon: "
            f"{era5.sizes['lon']}"
        )

        print(
            f"  latitude: "
            f"{float(era5.lat.min())} "
            f"to "
            f"{float(era5.lat.max())}"
        )

        print(
            f"  longitude: "
            f"{float(era5.lon.min())} "
            f"to "
            f"{float(era5.lon.max())}"
        )

        for variable in VARIABLES:

            print()
            print(
                "=" * 100
            )

            print(
                variable
            )

            print(
                "=" * 100
            )

            cmip6_file = find_cmip6_file(
                variable
            )

            cmip6 = xr.open_dataset(
                cmip6_file,
                engine="netcdf4",
                decode_times=True,
            )

            try:

                cmip6 = normalize_longitude(
                    cmip6
                )

                cmip6 = crop_to_era5_grid(
                    cmip6,
                    era5,
                )

                if cmip6.sizes[
                    "time"
                ] != 252:

                    raise ValueError(
                        f"{variable}: "
                        f"unexpected CMIP6 time count"
                    )

                if era5.sizes[
                    "time"
                ] != 252:

                    raise ValueError(
                        f"{variable}: "
                        f"unexpected ERA5 time count"
                    )

                if not np.array_equal(
                    era5[
                        "lat"
                    ].values,
                    cmip6[
                        "lat"
                    ].values,
                ):

                    raise ValueError(
                        f"{variable}: "
                        f"latitude coordinates do not match"
                    )

                if not np.array_equal(
                    era5[
                        "lon"
                    ].values,
                    cmip6[
                        "lon"
                    ].values,
                ):

                    raise ValueError(
                        f"{variable}: "
                        f"longitude coordinates do not match"
                    )

                era5_months = get_month_ids(
                    era5
                )

                cmip6_months = get_month_ids(
                    cmip6
                )

                if not np.array_equal(
                    era5_months,
                    cmip6_months,
                ):

                    raise ValueError(
                        f"{variable}: "
                        f"year-month coordinates do not match"
                    )

                print(
                    "Coordinate alignment: OK"
                )

                print(
                    f"CMIP6 file: "
                    f"{cmip6_file.name}"
                )

                print()

                print_stats(
                    "ERA5",
                    era5[
                        variable
                    ],
                )

                print()

                print_stats(
                    "CMIP6",
                    cmip6[
                        variable
                    ],
                )

                era5_units = (
                    era5[
                        variable
                    ].attrs.get(
                        "units",
                        "",
                    )
                )

                cmip6_units = (
                    cmip6[
                        variable
                    ].attrs.get(
                        "units",
                        "",
                    )
                )

                print()

                print(
                    f"  ERA5 units: "
                    f"{era5_units}"
                )

                print(
                    f"  CMIP6 units: "
                    f"{cmip6_units}"
                )

                if (
                    era5[
                        variable
                    ].shape
                    !=
                    cmip6[
                        variable
                    ].shape
                ):

                    raise ValueError(
                        f"{variable}: "
                        f"data shapes do not match"
                    )

                print(
                    "Shape alignment: OK"
                )

            finally:

                cmip6.close()

        print()
        print(
            "=" * 100
        )

        print(
            "QM PAIR CHECK PASSED"
        )

        print(
            "=" * 100
        )

    finally:

        era5.close()


if __name__ == "__main__":
    main()