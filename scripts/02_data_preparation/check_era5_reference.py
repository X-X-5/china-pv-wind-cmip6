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


ERA5_FILE = get_path("data_raw_era5_raw") / "ERA5_reference_199401-201412.nc"


EXPECTED_VARIABLES = [
    "tas",
    "sfcWind",
    "rsds",
]


def check_monthly_sequence(ds):

    years = ds["time"].dt.year.values
    months = ds["time"].dt.month.values

    month_ids = (
        years * 12
        + months
        - 1
    )

    if len(month_ids) != 252:
        raise ValueError(
            f"Unexpected time count: "
            f"{len(month_ids)} != 252"
        )

    if len(np.unique(month_ids)) != 252:
        raise ValueError(
            "Duplicate months were found."
        )

    differences = np.diff(
        month_ids
    )

    if not np.all(
        differences == 1
    ):
        raise ValueError(
            "Monthly sequence is not continuous."
        )

    if (
        int(years[0]) != 1994
        or int(months[0]) != 1
    ):
        raise ValueError(
            "Unexpected first month."
        )

    if (
        int(years[-1]) != 2014
        or int(months[-1]) != 12
    ):
        raise ValueError(
            "Unexpected last month."
        )


def check_coordinates(ds):

    if "lat" not in ds.coords:
        raise ValueError(
            "Latitude coordinate is missing."
        )

    if "lon" not in ds.coords:
        raise ValueError(
            "Longitude coordinate is missing."
        )

    lat = ds["lat"].values
    lon = ds["lon"].values

    if len(lat) != 157:
        raise ValueError(
            f"Unexpected latitude size: "
            f"{len(lat)} != 157"
        )

    if len(lon) != 281:
        raise ValueError(
            f"Unexpected longitude size: "
            f"{len(lon)} != 281"
        )

    if not np.all(
        np.diff(lat) > 0
    ):
        raise ValueError(
            "Latitude is not strictly ascending."
        )

    if not np.all(
        np.diff(lon) > 0
    ):
        raise ValueError(
            "Longitude is not strictly ascending."
        )

    lat_resolution = float(
        np.median(
            np.diff(lat)
        )
    )

    lon_resolution = float(
        np.median(
            np.diff(lon)
        )
    )

    if not np.isclose(
        lat_resolution,
        0.25,
    ):
        raise ValueError(
            f"Unexpected latitude resolution: "
            f"{lat_resolution}"
        )

    if not np.isclose(
        lon_resolution,
        0.25,
    ):
        raise ValueError(
            f"Unexpected longitude resolution: "
            f"{lon_resolution}"
        )


def check_variables(ds):

    for variable in EXPECTED_VARIABLES:

        if variable not in ds.data_vars:
            raise ValueError(
                f"Missing variable: "
                f"{variable}"
            )

        da = ds[
            variable
        ]

        expected_shape = (
            252,
            157,
            281,
        )

        if da.shape != expected_shape:
            raise ValueError(
                f"Unexpected shape for "
                f"{variable}: "
                f"{da.shape}"
            )

        if not np.isfinite(
            da.values
        ).any():
            raise ValueError(
                f"No finite values found "
                f"for {variable}."
            )


def print_statistics(
    ds,
    variable,
):

    da = ds[
        variable
    ]

    print(
        f"{variable}"
    )

    print(
        f"  units: "
        f"{da.attrs.get('units', '')}"
    )

    print(
        f"  mean: "
        f"{float(da.mean(skipna=True).values):.6f}"
    )

    print(
        f"  min: "
        f"{float(da.min(skipna=True).values):.6f}"
    )

    print(
        f"  max: "
        f"{float(da.max(skipna=True).values):.6f}"
    )


def main():

    print(
        "=" * 80
    )

    print(
        "ERA5 REFERENCE QC"
    )

    print(
        "=" * 80
    )

    if not ERA5_FILE.exists():
        raise FileNotFoundError(
            f"File not found: "
            f"{ERA5_FILE}"
        )

    ds = xr.open_dataset(
        ERA5_FILE,
        engine="netcdf4",
        decode_times=True,
    )

    try:

        check_monthly_sequence(
            ds
        )

        check_coordinates(
            ds
        )

        check_variables(
            ds
        )

        print()
        print(
            "Time check: OK"
        )

        print(
            "Coordinate check: OK"
        )

        print(
            "Variable check: OK"
        )

        print()

        for variable in EXPECTED_VARIABLES:

            print_statistics(
                ds,
                variable,
            )

            print()

        print(
            "Coordinate summary:"
        )

        print(
            f"  Time: "
            f"{ds.time.values[0]} "
            f"to "
            f"{ds.time.values[-1]}"
        )

        print(
            f"  Grid: "
            f"{ds.sizes['lat']} x "
            f"{ds.sizes['lon']}"
        )

        print(
            f"  Latitude: "
            f"{float(ds.lat.min())} "
            f"to "
            f"{float(ds.lat.max())}"
        )

        print(
            f"  Longitude: "
            f"{float(ds.lon.min())} "
            f"to "
            f"{float(ds.lon.max())}"
        )

        print()

        print(
            "=" * 80
        )

        print(
            "ERA5 REFERENCE QC PASSED"
        )

        print(
            "=" * 80
        )

    finally:

        ds.close()


if __name__ == "__main__":
    main()