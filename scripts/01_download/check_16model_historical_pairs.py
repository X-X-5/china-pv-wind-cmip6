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


ROOT = PROJECT_ROOT

CMIP6_ROOT = (
    get_path("data_interim_cmip6_china_clipped")
)

ERA5_ROOT = (
    get_path("data_interim_era5_on_gcm_grid")
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

VARIABLES = [
    "tas",
    "sfcWind",
    "rsds",
]

EXPECTED_MONTHS = 672

EXPECTED_UNITS = {
    "tas": [
        "K",
        "kelvin",
    ],
    "sfcWind": [
        "m s-1",
        "m s**-1",
        "m/s",
        "m s^-1",
    ],
    "rsds": [
        "W m-2",
        "W m**-2",
        "W/m2",
        "W m^-2",
    ],
}


def get_time_coder():

    return xr.coders.CFDatetimeCoder(
        use_cftime=True
    )


def normalize_unit(unit):

    if unit is None:

        return ""

    return (
        str(unit)
        .strip()
        .lower()
        .replace(" ", "")
    )


def check_unit(
    variable,
    unit,
):

    normalized = normalize_unit(
        unit
    )

    allowed = [
        normalize_unit(x)
        for x in EXPECTED_UNITS[
            variable
        ]
    ]

    if normalized not in allowed:

        raise ValueError(
            f"Unexpected unit for {variable}: {unit}"
        )


def find_cmip_file(
    model,
    variable,
):

    directory = (
        CMIP6_ROOT
        / model
        / "historical"
        / variable
    )

    files = sorted(
        directory.glob("*.nc")
    )

    if len(files) != 1:

        raise ValueError(
            f"{model} | {variable}: "
            f"expected 1 CMIP6 file, found {len(files)}"
        )

    return files[0]


def find_era5_file(
    model,
):

    file_path = (
        ERA5_ROOT
        / model
        / f"ERA5_{model}_195901-201412.nc"
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"ERA5 file not found: {file_path}"
        )

    return file_path


def get_month_ids(ds):

    years = (
        ds["time"]
        .dt.year
        .values
        .astype(np.int64)
    )

    months = (
        ds["time"]
        .dt.month
        .values
        .astype(np.int64)
    )

    return (
        years * 12
        + months
        - 1
    )


def check_time(
    ds,
    label,
):

    if ds.sizes[
        "time"
    ] != EXPECTED_MONTHS:

        raise ValueError(
            f"{label}: expected {EXPECTED_MONTHS} months, "
            f"found {ds.sizes['time']}"
        )

    start = (
        int(
            ds.time.dt.year.values[
                0
            ]
        ),
        int(
            ds.time.dt.month.values[
                0
            ]
        ),
    )

    end = (
        int(
            ds.time.dt.year.values[
                -1
            ]
        ),
        int(
            ds.time.dt.month.values[
                -1
            ]
        ),
    )

    if start != (
        1959,
        1,
    ):

        raise ValueError(
            f"{label}: unexpected start "
            f"{start[0]:04d}-{start[1]:02d}"
        )

    if end != (
        2014,
        12,
    ):

        raise ValueError(
            f"{label}: unexpected end "
            f"{end[0]:04d}-{end[1]:02d}"
        )

    month_ids = get_month_ids(
        ds
    )

    if len(
        np.unique(
            month_ids
        )
    ) != EXPECTED_MONTHS:

        raise ValueError(
            f"{label}: duplicate year-month entries"
        )

    expected_ids = np.arange(
        month_ids[
            0
        ],
        month_ids[
            0
        ]
        + EXPECTED_MONTHS,
    )

    if not np.array_equal(
        month_ids,
        expected_ids,
    ):

        raise ValueError(
            f"{label}: non-continuous time sequence"
        )


def check_grid(
    era5,
    cmip,
    model,
    variable,
):

    if era5.sizes[
        "lat"
    ] != cmip.sizes[
        "lat"
    ]:

        raise ValueError(
            f"{model} | {variable}: latitude count mismatch"
        )

    if era5.sizes[
        "lon"
    ] != cmip.sizes[
        "lon"
    ]:

        raise ValueError(
            f"{model} | {variable}: longitude count mismatch"
        )

    if not np.allclose(
        era5[
            "lat"
        ].values,
        cmip[
            "lat"
        ].values,
        rtol=0.0,
        atol=1e-10,
    ):

        raise ValueError(
            f"{model} | {variable}: latitude values mismatch"
        )

    if not np.allclose(
        era5[
            "lon"
        ].values,
        cmip[
            "lon"
        ].values,
        rtol=0.0,
        atol=1e-10,
    ):

        raise ValueError(
            f"{model} | {variable}: longitude values mismatch"
        )


def check_nan_and_finite(
    da,
    label,
):

    values = da.values

    nan_count = int(
        np.isnan(
            values
        ).sum()
    )

    if nan_count != 0:

        raise ValueError(
            f"{label}: {nan_count} NaN values"
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):

        raise ValueError(
            f"{label}: non-finite values detected"
        )


def get_stats(
    da,
):

    return {
        "mean": float(
            da.mean().values
        ),
        "min": float(
            da.min().values
        ),
        "max": float(
            da.max().values
        ),
        "std": float(
            da.std().values
        ),
    }


def process_model(
    model,
):

    time_coder = get_time_coder()

    era5_file = find_era5_file(
        model
    )

    era5 = xr.open_dataset(
        era5_file,
        decode_times=time_coder,
    )

    cmip_datasets = {}

    try:

        check_time(
            era5,
            f"{model} ERA5",
        )

        for variable in VARIABLES:

            if variable not in era5.data_vars:

                raise ValueError(
                    f"{model}: ERA5 missing {variable}"
                )

            era5_unit = era5[
                variable
            ].attrs.get(
                "units"
            )

            check_unit(
                variable,
                era5_unit,
            )

            check_nan_and_finite(
                era5[
                    variable
                ],
                f"{model} ERA5 {variable}",
            )

            cmip_file = find_cmip_file(
                model,
                variable,
            )

            cmip = xr.open_dataset(
                cmip_file,
                decode_times=time_coder,
            )

            cmip_datasets[
                variable
            ] = cmip

            check_time(
                cmip,
                f"{model} CMIP6 {variable}",
            )

            if variable not in cmip.data_vars:

                raise ValueError(
                    f"{model}: CMIP6 missing {variable}"
                )

            cmip_unit = cmip[
                variable
            ].attrs.get(
                "units"
            )

            check_unit(
                variable,
                cmip_unit,
            )

            check_grid(
                era5,
                cmip,
                model,
                variable,
            )

            check_nan_and_finite(
                cmip[
                    variable
                ],
                f"{model} CMIP6 {variable}",
            )

        print(
            f"  Grid: "
            f"{era5.sizes['lat']} x "
            f"{era5.sizes['lon']}"
        )

        for variable in VARIABLES:

            era5_stats = get_stats(
                era5[
                    variable
                ]
            )

            cmip_stats = get_stats(
                cmip_datasets[
                    variable
                ][
                    variable
                ]
            )

            era5_unit = era5[
                variable
            ].attrs.get(
                "units",
                "",
            )

            cmip_unit = (
                cmip_datasets[
                    variable
                ][
                    variable
                ]
                .attrs
                .get(
                    "units",
                    "",
                )
            )

            print()
            print(
                f"  {variable}"
            )

            print(
                f"    ERA5 unit: {era5_unit}"
            )

            print(
                f"    CMIP6 unit: {cmip_unit}"
            )

            print(
                f"    ERA5 mean: "
                f"{era5_stats['mean']:.6f}"
            )

            print(
                f"    CMIP6 mean: "
                f"{cmip_stats['mean']:.6f}"
            )

            print(
                f"    ERA5 min/max: "
                f"{era5_stats['min']:.6f} / "
                f"{era5_stats['max']:.6f}"
            )

            print(
                f"    CMIP6 min/max: "
                f"{cmip_stats['min']:.6f} / "
                f"{cmip_stats['max']:.6f}"
            )

            print(
                f"    ERA5 std: "
                f"{era5_stats['std']:.6f}"
            )

            print(
                f"    CMIP6 std: "
                f"{cmip_stats['std']:.6f}"
            )

        return True

    finally:

        era5.close()

        for ds in cmip_datasets.values():

            ds.close()


def main():

    print()
    print("=" * 120)
    print("CMIP6 HISTORICAL PAIR QC")
    print("=" * 120)

    success_count = 0
    failed_count = 0

    failures = []

    for index, model in enumerate(
        MODELS,
        start=1,
    ):

        print()
        print("-" * 120)

        print(
            f"MODEL {index}/{len(MODELS)}: "
            f"{model}"
        )

        print("-" * 120)

        try:

            process_model(
                model
            )

            success_count += 1

            print()
            print(
                "Status: OK"
            )

        except Exception as exc:

            failed_count += 1

            failures.append(
                (
                    model,
                    f"{type(exc).__name__}: {exc}",
                )
            )

            print()
            print(
                "Status: FAILED"
            )

            print(
                f"Error: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    print()
    print("=" * 120)
    print("PAIR QC SUMMARY")
    print("=" * 120)

    print(
        f"Models: {len(MODELS)}"
    )

    print(
        f"Success: {success_count}"
    )

    print(
        f"Failed: {failed_count}"
    )

    if failures:

        print()
        print(
            "FAILED MODELS"
        )

        for (
            model,
            error,
        ) in failures:

            print(
                f"{model} | {error}"
            )

    else:

        print()
        print(
            "ALL HISTORICAL PAIRS PASSED"
        )


if __name__ == "__main__":
    main()