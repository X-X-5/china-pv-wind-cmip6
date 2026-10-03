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
import sys

import numpy as np
import xarray as xr


CMIP6_ROOT = get_path("data_raw_cmip6_downloaded")

MODEL = "CAS-ESM2-0"
VARIANT = "r1i1p1f1"
TABLE_ID = "Amon"
GRID_LABEL = "gn"

EXPERIMENTS = {
    "historical": {
        "start_year": 1850,
        "end_year": 2014,
        "time_size": 1980,
    },
    "ssp126": {
        "start_year": 2015,
        "end_year": 2100,
        "time_size": 1032,
    },
    "ssp245": {
        "start_year": 2015,
        "end_year": 2100,
        "time_size": 1032,
    },
    "ssp585": {
        "start_year": 2015,
        "end_year": 2100,
        "time_size": 1032,
    },
}

VARIABLES = {
    "tas": {
        "units": {"K", "kelvin"},
        "minimum": 150.0,
        "maximum": 350.0,
    },
    "rsds": {
        "units": {"W m-2", "W m**-2", "W/m2"},
        "minimum": -5.0,
        "maximum": 1500.0,
    },
    "sfcWind": {
        "units": {"m s-1", "m s**-1", "m/s"},
        "minimum": 0.0,
        "maximum": 150.0,
    },
}


def get_year(value):
    if hasattr(value, "year"):
        return int(value.year)

    return int(str(value)[:4])


def get_coordinate_name(dataset, candidates):
    for name in candidates:
        if name in dataset.coords:
            return name

    raise KeyError(f"No coordinate found among {candidates}")


def find_file(experiment, variable):
    directory = (
        CMIP6_ROOT
        / MODEL
        / experiment
        / VARIANT
        / TABLE_ID
        / variable
        / GRID_LABEL
    )

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    files = sorted(directory.glob("*.nc"))

    if len(files) != 1:
        raise RuntimeError(
            f"Expected one NetCDF file, found {len(files)} in {directory}"
        )

    return files[0]


def main():
    print("=" * 100)
    print("CAS-ESM2-0 RAW CMIP6 QUALITY CONTROL")
    print("=" * 100)
    print(f"CMIP6 root: {CMIP6_ROOT}")

    passed = 0
    failed = 0
    reference_latitude = None
    reference_longitude = None

    total = len(EXPERIMENTS) * len(VARIABLES)
    current = 0

    for experiment, expected in EXPERIMENTS.items():
        for variable, limits in VARIABLES.items():
            current += 1
            errors = []

            print()
            print("-" * 100)
            print(f"[{current}/{total}] {MODEL} {experiment} {variable}")
            print("-" * 100)

            try:
                path = find_file(experiment, variable)

                with xr.open_dataset(path, decode_times=True) as dataset:
                    if variable not in dataset.data_vars:
                        raise KeyError(
                            f"Variable {variable!r} is missing."
                        )

                    data = dataset[variable]

                    latitude_name = get_coordinate_name(
                        dataset,
                        ("lat", "latitude"),
                    )
                    longitude_name = get_coordinate_name(
                        dataset,
                        ("lon", "longitude"),
                    )

                    if "time" not in data.dims:
                        errors.append("Time dimension is missing.")

                    if latitude_name not in data.dims:
                        errors.append("Latitude dimension is missing.")

                    if longitude_name not in data.dims:
                        errors.append("Longitude dimension is missing.")

                    time = dataset["time"]
                    latitude = np.asarray(dataset[latitude_name].values)
                    longitude = np.asarray(dataset[longitude_name].values)

                    time_size = dataset.sizes["time"]
                    first_time = time.values[0]
                    last_time = time.values[-1]

                    if time_size != expected["time_size"]:
                        errors.append(
                            f"Unexpected time size: {time_size}; "
                            f"expected {expected['time_size']}."
                        )

                    if get_year(first_time) != expected["start_year"]:
                        errors.append(
                            f"Unexpected first year: {first_time}."
                        )

                    if get_year(last_time) != expected["end_year"]:
                        errors.append(
                            f"Unexpected last year: {last_time}."
                        )

                    if not dataset.indexes["time"].is_unique:
                        errors.append("Duplicate time values were found.")

                    if not dataset.indexes["time"].is_monotonic_increasing:
                        errors.append(
                            "Time is not monotonically increasing."
                        )

                    if latitude.ndim != 1 or longitude.ndim != 1:
                        errors.append(
                            "Latitude or longitude is not one-dimensional."
                        )

                    latitude_monotonic = (
                        np.all(np.diff(latitude) > 0)
                        or np.all(np.diff(latitude) < 0)
                    )
                    longitude_monotonic = (
                        np.all(np.diff(longitude) > 0)
                        or np.all(np.diff(longitude) < 0)
                    )

                    if not latitude_monotonic:
                        errors.append("Latitude is not strictly monotonic.")

                    if not longitude_monotonic:
                        errors.append("Longitude is not strictly monotonic.")

                    units = str(data.attrs.get("units", ""))

                    if units not in limits["units"]:
                        errors.append(f"Unexpected units: {units!r}.")

                    if reference_latitude is None:
                        reference_latitude = latitude.copy()
                        reference_longitude = longitude.copy()
                    else:
                        same_grid = (
                            latitude.shape == reference_latitude.shape
                            and longitude.shape == reference_longitude.shape
                            and np.allclose(
                                latitude,
                                reference_latitude,
                                equal_nan=True,
                            )
                            and np.allclose(
                                longitude,
                                reference_longitude,
                                equal_nan=True,
                            )
                        )

                        if not same_grid:
                            errors.append(
                                "Grid differs from the first file."
                            )

                    sample_indices = sorted(
                        {
                            0,
                            time_size // 2,
                            time_size - 1,
                        }
                    )

                    sample = data.isel(
                        time=sample_indices
                    ).load().values

                    finite = np.isfinite(sample)
                    finite_percent = float(finite.mean() * 100.0)

                    if not finite.any():
                        errors.append(
                            "All sampled values are non-finite."
                        )
                        sample_minimum = np.nan
                        sample_maximum = np.nan
                    else:
                        sample_minimum = float(np.nanmin(sample))
                        sample_maximum = float(np.nanmax(sample))

                        if sample_minimum < limits["minimum"]:
                            errors.append(
                                f"Suspicious sample minimum: "
                                f"{sample_minimum}."
                            )

                        if sample_maximum > limits["maximum"]:
                            errors.append(
                                f"Suspicious sample maximum: "
                                f"{sample_maximum}."
                            )

                    print(f"  File       : {path.name}")
                    print(f"  Size       : {path.stat().st_size} bytes")
                    print(
                        f"  Time       : {first_time} to {last_time} "
                        f"({time_size} values)"
                    )
                    print(
                        f"  Grid       : "
                        f"{latitude.size} x {longitude.size}"
                    )
                    print(
                        f"  Latitude   : "
                        f"{latitude.min()} to {latitude.max()}"
                    )
                    print(
                        f"  Longitude  : "
                        f"{longitude.min()} to {longitude.max()}"
                    )
                    print(f"  Units      : {units}")
                    print(
                        f"  Sample     : min={sample_minimum:.6g}, "
                        f"max={sample_maximum:.6g}, "
                        f"finite={finite_percent:.3f}%"
                    )

                if errors:
                    failed += 1
                    print("  Status     : FAILED")

                    for error in errors:
                        print(f"  Error      : {error}")
                else:
                    passed += 1
                    print("  Status     : PASSED")

            except Exception as error:
                failed += 1
                print("  Status     : FAILED")
                print(
                    f"  Error      : "
                    f"{type(error).__name__}: {error}"
                )

    print()
    print("=" * 100)
    print("QC SUMMARY")
    print("=" * 100)
    print(f"Expected files: {total}")
    print(f"Passed files  : {passed}")
    print(f"Failed files  : {failed}")

    if failed == 0:
        print()
        print("ALL 12 CAS-ESM2-0 RAW FILES PASSED")
        return 0

    print()
    print("CAS-ESM2-0 RAW FILE QC FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())