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


ERA5_FILE = get_path("data_raw_era5_raw") / "116b0ea5b6757707487efaa87f9aaae.grib"


def estimate_resolution(values):

    values = np.asarray(
        values
    )

    if values.ndim != 1:
        return None

    if values.size < 2:
        return None

    diffs = np.diff(
        values.astype(float)
    )

    finite = diffs[
        np.isfinite(
            diffs
        )
    ]

    if finite.size == 0:
        return None

    return float(
        np.median(
            np.abs(
                finite
            )
        )
    )


def direction(values):

    values = np.asarray(
        values
    )

    diffs = np.diff(
        values.astype(float)
    )

    if np.all(
        diffs > 0
    ):
        return "ascending"

    if np.all(
        diffs < 0
    ):
        return "descending"

    return "non_monotonic"


def inspect_dataset(
    name,
    ds,
):

    print()
    print(
        "=" * 80
    )

    print(
        name
    )

    print(
        "=" * 80
    )

    print(
        "Dimensions:"
    )

    for key, value in ds.sizes.items():

        print(
            f"  {key}: {value}"
        )

    print()

    print(
        "Coordinates:"
    )

    for coord in ds.coords:

        print(
            f"  {coord}"
        )

    print()

    if "latitude" in ds.coords:

        lat_name = "latitude"

    elif "lat" in ds.coords:

        lat_name = "lat"

    else:

        lat_name = None

    if "longitude" in ds.coords:

        lon_name = "longitude"

    elif "lon" in ds.coords:

        lon_name = "lon"

    else:

        lon_name = None

    if lat_name is not None:

        lat = ds[
            lat_name
        ].values

        print(
            f"Latitude name: "
            f"{lat_name}"
        )

        print(
            f"Latitude size: "
            f"{lat.size}"
        )

        print(
            f"Latitude min: "
            f"{float(np.nanmin(lat))}"
        )

        print(
            f"Latitude max: "
            f"{float(np.nanmax(lat))}"
        )

        print(
            f"Latitude resolution: "
            f"{estimate_resolution(lat)}"
        )

        print(
            f"Latitude direction: "
            f"{direction(lat)}"
        )

    print()

    if lon_name is not None:

        lon = ds[
            lon_name
        ].values

        print(
            f"Longitude name: "
            f"{lon_name}"
        )

        print(
            f"Longitude size: "
            f"{lon.size}"
        )

        print(
            f"Longitude min: "
            f"{float(np.nanmin(lon))}"
        )

        print(
            f"Longitude max: "
            f"{float(np.nanmax(lon))}"
        )

        print(
            f"Longitude resolution: "
            f"{estimate_resolution(lon)}"
        )

        print(
            f"Longitude direction: "
            f"{direction(lon)}"
        )

    print()

    print(
        "Variables:"
    )

    for variable in ds.data_vars:

        print(
            f"  {variable}: "
            f"{ds[variable].shape}"
        )


def main():

    print(
        "=" * 80
    )

    print(
        "ERA5 GRID SCAN"
    )

    print(
        "=" * 80
    )

    ds_main = xr.open_dataset(
        ERA5_FILE,
        engine="cfgrib",
    )

    inspect_dataset(
        "ERA5 MAIN",
        ds_main,
    )

    ds_ssrd = xr.open_dataset(
        ERA5_FILE,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "shortName": "ssrd"
            }
        },
    )

    inspect_dataset(
        "ERA5 SSRD",
        ds_ssrd,
    )

    ds_main.close()
    ds_ssrd.close()


if __name__ == "__main__":

    main()