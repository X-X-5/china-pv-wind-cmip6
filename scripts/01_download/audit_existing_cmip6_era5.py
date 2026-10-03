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
from collections import Counter
from pathlib import Path
import csv
import sys

import numpy as np
import xarray as xr


PROJECT_ROOT = PROJECT_ROOT

CMIP6_ROOT = (
    get_path("data_interim_cmip6_china_clipped")
)

ERA5_REFERENCE_FILE = (
    get_path("data_interim_era5_reference") / "ERA5_reference_195901_201412.nc"
)

ERA5_GRID_ROOT = (
    get_path("data_interim_era5_on_gcm_grid")
)

REPORT_CSV = (
    get_path("results_audits") / "existing_cmip6_era5_audit.csv"
)

EXCLUDED_MODELS = {
    "CAS-ESM2-0",
}

EXPECTED_EXISTING_MODELS = 16

EXPERIMENTS = (
    "historical",
    "ssp126",
    "ssp245",
    "ssp585",
)

VARIABLES = (
    "tas",
    "rsds",
    "sfcWind",
)

EXPECTED_UNITS = {
    "tas": {
        "K",
        "kelvin",
    },
    "rsds": {
        "W m-2",
        "W m**-2",
        "W/m2",
    },
    "sfcWind": {
        "m s-1",
        "m s**-1",
        "m/s",
    },
}

SAMPLE_LIMITS = {
    "tas": (150.0, 350.0),
    "rsds": (-5.0, 1500.0),
    "sfcWind": (0.0, 150.0),
}


def year_month_code(year, month):
    return int(year) * 12 + int(month) - 1


def code_to_text(code):
    year = code // 12
    month = code % 12 + 1
    return f"{year:04d}-{month:02d}"


def find_coordinate(dataset, candidates):
    for name in candidates:
        if name in dataset.coords:
            return name

    raise KeyError(
        f"No coordinate found among {candidates}"
    )


def grids_equal(first, second):
    if first is None or second is None:
        return False

    return (
        first["lat"].shape == second["lat"].shape
        and first["lon"].shape == second["lon"].shape
        and np.allclose(
            first["lat"],
            second["lat"],
            equal_nan=True,
        )
        and np.allclose(
            first["lon"],
            second["lon"],
            equal_nan=True,
        )
    )


def discover_models():
    if not CMIP6_ROOT.exists():
        raise FileNotFoundError(
            f"CMIP6 root does not exist: {CMIP6_ROOT}"
        )

    models = sorted(
        path.name
        for path in CMIP6_ROOT.iterdir()
        if path.is_dir()
        and path.name not in EXCLUDED_MODELS
        and (path / "historical").exists()
    )

    return models


def get_netcdf_files(directory):
    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".nc", ".nc4"}
        and not path.name.endswith(".tmp.nc")
    )


def inspect_cmip6_group(
    model,
    experiment,
    variable,
):
    directory = (
        CMIP6_ROOT
        / model
        / experiment
        / variable
    )

    files = get_netcdf_files(directory)
    errors = []
    warnings = []

    result = {
        "section": "CMIP6",
        "model": model,
        "experiment": experiment,
        "variable": variable,
        "file_count": len(files),
        "first_time": "",
        "last_time": "",
        "time_count": 0,
        "lat_count": 0,
        "lon_count": 0,
        "lat_min": np.nan,
        "lat_max": np.nan,
        "lon_min": np.nan,
        "lon_max": np.nan,
        "units": "",
        "dtype": "",
        "sample_min": np.nan,
        "sample_max": np.nan,
        "finite_percent": np.nan,
        "status": "FAILED",
        "message": "",
        "_grid": None,
        "_codes": [],
    }

    if not files:
        result["message"] = (
            f"No NetCDF files found in {directory}"
        )
        return result

    reference_grid = None
    all_codes = []
    units = set()
    dtypes = set()
    sample_minimums = []
    sample_maximums = []
    finite_counts = 0
    total_counts = 0

    for path in files:
        try:
            with xr.open_dataset(
                path,
                decode_times=True,
            ) as dataset:
                if variable not in dataset.data_vars:
                    errors.append(
                        f"{path.name}: variable "
                        f"{variable!r} is missing"
                    )
                    continue

                latitude_name = find_coordinate(
                    dataset,
                    ("lat", "latitude"),
                )
                longitude_name = find_coordinate(
                    dataset,
                    ("lon", "longitude"),
                )

                data = dataset[variable]

                required_dimensions = {
                    "time",
                    latitude_name,
                    longitude_name,
                }

                if not required_dimensions.issubset(
                    set(data.dims)
                ):
                    errors.append(
                        f"{path.name}: unexpected "
                        f"dimensions {data.dims}"
                    )
                    continue

                latitude = np.asarray(
                    dataset[latitude_name].values
                )
                longitude = np.asarray(
                    dataset[longitude_name].values
                )

                current_grid = {
                    "lat": latitude,
                    "lon": longitude,
                }

                if reference_grid is None:
                    reference_grid = current_grid
                elif not grids_equal(
                    reference_grid,
                    current_grid,
                ):
                    errors.append(
                        f"{path.name}: grid differs "
                        f"within the group"
                    )

                years = np.asarray(
                    dataset["time"].dt.year.values
                )
                months = np.asarray(
                    dataset["time"].dt.month.values
                )

                codes = [
                    year_month_code(year, month)
                    for year, month in zip(
                        years,
                        months,
                    )
                ]

                all_codes.extend(codes)

                units.add(
                    str(data.attrs.get("units", ""))
                )
                dtypes.add(str(data.dtype))

                time_size = data.sizes["time"]
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

                finite_counts += int(finite.sum())
                total_counts += int(finite.size)

                if finite.any():
                    sample_minimums.append(
                        float(np.nanmin(sample))
                    )
                    sample_maximums.append(
                        float(np.nanmax(sample))
                    )

        except Exception as error:
            errors.append(
                f"{path.name}: "
                f"{type(error).__name__}: {error}"
            )

    if reference_grid is not None:
        latitude = reference_grid["lat"]
        longitude = reference_grid["lon"]

        result["lat_count"] = latitude.size
        result["lon_count"] = longitude.size
        result["lat_min"] = float(
            np.nanmin(latitude)
        )
        result["lat_max"] = float(
            np.nanmax(latitude)
        )
        result["lon_min"] = float(
            np.nanmin(longitude)
        )
        result["lon_max"] = float(
            np.nanmax(longitude)
        )
        result["_grid"] = reference_grid

        latitude_monotonic = (
            np.all(np.diff(latitude) > 0)
            or np.all(np.diff(latitude) < 0)
        )
        longitude_monotonic = (
            np.all(np.diff(longitude) > 0)
            or np.all(np.diff(longitude) < 0)
        )

        if not latitude_monotonic:
            errors.append(
                "Latitude is not strictly monotonic"
            )

        if not longitude_monotonic:
            errors.append(
                "Longitude is not strictly monotonic"
            )

    sorted_codes = sorted(all_codes)
    unique_codes = sorted(set(all_codes))

    result["time_count"] = len(all_codes)
    result["_codes"] = unique_codes

    if unique_codes:
        result["first_time"] = code_to_text(
            unique_codes[0]
        )
        result["last_time"] = code_to_text(
            unique_codes[-1]
        )

    if len(unique_codes) != len(all_codes):
        errors.append(
            "Duplicate year-month values were found"
        )

    if len(unique_codes) > 1:
        differences = np.diff(unique_codes)

        if not np.all(differences == 1):
            missing_count = int(
                np.sum(differences - 1)
            )
            errors.append(
                f"Monthly time series contains "
                f"{missing_count} missing months"
            )

    if all_codes != sorted_codes:
        warnings.append(
            "Files are not ordered chronologically"
        )

    result["units"] = "; ".join(sorted(units))
    result["dtype"] = "; ".join(sorted(dtypes))

    if len(units) != 1:
        errors.append(
            f"Inconsistent units: {sorted(units)}"
        )
    elif next(iter(units)) not in EXPECTED_UNITS[
        variable
    ]:
        errors.append(
            f"Unexpected units: {next(iter(units))!r}"
        )

    if sample_minimums:
        result["sample_min"] = min(
            sample_minimums
        )
        result["sample_max"] = max(
            sample_maximums
        )

        lower, upper = SAMPLE_LIMITS[variable]

        if result["sample_min"] < lower:
            errors.append(
                f"Suspicious sample minimum: "
                f"{result['sample_min']}"
            )

        if result["sample_max"] > upper:
            errors.append(
                f"Suspicious sample maximum: "
                f"{result['sample_max']}"
            )
    else:
        errors.append(
            "All sampled values are non-finite"
        )

    if total_counts:
        result["finite_percent"] = (
            finite_counts / total_counts * 100.0
        )

        if result["finite_percent"] < 99.0:
            warnings.append(
                f"Sample finite percentage is "
                f"{result['finite_percent']:.3f}%"
            )

    if errors:
        result["status"] = "FAILED"
    elif warnings:
        result["status"] = "WARNING"
    else:
        result["status"] = "PASSED"

    result["message"] = " | ".join(
        errors + warnings
    )

    return result


def inspect_era5_file(
    path,
    model_name,
    section,
):
    results = []

    if not path.exists():
        for variable in VARIABLES:
            results.append(
                {
                    "section": section,
                    "model": model_name,
                    "experiment": "historical",
                    "variable": variable,
                    "file_count": 0,
                    "first_time": "",
                    "last_time": "",
                    "time_count": 0,
                    "lat_count": 0,
                    "lon_count": 0,
                    "lat_min": np.nan,
                    "lat_max": np.nan,
                    "lon_min": np.nan,
                    "lon_max": np.nan,
                    "units": "",
                    "dtype": "",
                    "sample_min": np.nan,
                    "sample_max": np.nan,
                    "finite_percent": np.nan,
                    "status": "FAILED",
                    "message": (
                        f"ERA5 file does not exist: "
                        f"{path}"
                    ),
                    "_grid": None,
                    "_codes": [],
                }
            )

        return results

    try:
        with xr.open_dataset(
            path,
            decode_times=True,
        ) as dataset:
            latitude_name = find_coordinate(
                dataset,
                ("lat", "latitude"),
            )
            longitude_name = find_coordinate(
                dataset,
                ("lon", "longitude"),
            )

            latitude = np.asarray(
                dataset[latitude_name].values
            )
            longitude = np.asarray(
                dataset[longitude_name].values
            )

            grid = {
                "lat": latitude,
                "lon": longitude,
            }

            years = np.asarray(
                dataset["time"].dt.year.values
            )
            months = np.asarray(
                dataset["time"].dt.month.values
            )

            codes = [
                year_month_code(year, month)
                for year, month in zip(
                    years,
                    months,
                )
            ]

            unique_codes = sorted(set(codes))

            time_errors = []

            if len(unique_codes) != len(codes):
                time_errors.append(
                    "Duplicate year-month values "
                    "were found"
                )

            if len(unique_codes) > 1:
                if not np.all(
                    np.diff(unique_codes) == 1
                ):
                    time_errors.append(
                        "Monthly time series has gaps"
                    )

            for variable in VARIABLES:
                errors = list(time_errors)
                warnings = []

                result = {
                    "section": section,
                    "model": model_name,
                    "experiment": "historical",
                    "variable": variable,
                    "file_count": 1,
                    "first_time": (
                        code_to_text(unique_codes[0])
                        if unique_codes else ""
                    ),
                    "last_time": (
                        code_to_text(unique_codes[-1])
                        if unique_codes else ""
                    ),
                    "time_count": len(codes),
                    "lat_count": latitude.size,
                    "lon_count": longitude.size,
                    "lat_min": float(
                        np.nanmin(latitude)
                    ),
                    "lat_max": float(
                        np.nanmax(latitude)
                    ),
                    "lon_min": float(
                        np.nanmin(longitude)
                    ),
                    "lon_max": float(
                        np.nanmax(longitude)
                    ),
                    "units": "",
                    "dtype": "",
                    "sample_min": np.nan,
                    "sample_max": np.nan,
                    "finite_percent": np.nan,
                    "status": "FAILED",
                    "message": "",
                    "_grid": grid,
                    "_codes": unique_codes,
                }

                if variable not in dataset.data_vars:
                    errors.append(
                        f"Variable {variable!r} "
                        f"is missing"
                    )
                else:
                    data = dataset[variable]
                    units = str(
                        data.attrs.get("units", "")
                    )

                    result["units"] = units
                    result["dtype"] = str(data.dtype)

                    if units not in EXPECTED_UNITS[
                        variable
                    ]:
                        errors.append(
                            f"Unexpected units: "
                            f"{units!r}"
                        )

                    sample_indices = sorted(
                        {
                            0,
                            data.sizes["time"] // 2,
                            data.sizes["time"] - 1,
                        }
                    )

                    sample = data.isel(
                        time=sample_indices
                    ).load().values

                    finite = np.isfinite(sample)

                    result["finite_percent"] = (
                        float(
                            finite.mean() * 100.0
                        )
                    )

                    if finite.any():
                        result["sample_min"] = float(
                            np.nanmin(sample)
                        )
                        result["sample_max"] = float(
                            np.nanmax(sample)
                        )

                        lower, upper = SAMPLE_LIMITS[
                            variable
                        ]

                        if (
                            result["sample_min"]
                            < lower
                        ):
                            errors.append(
                                "Suspicious sample "
                                f"minimum: "
                                f"{result['sample_min']}"
                            )

                        if (
                            result["sample_max"]
                            > upper
                        ):
                            errors.append(
                                "Suspicious sample "
                                f"maximum: "
                                f"{result['sample_max']}"
                            )
                    else:
                        errors.append(
                            "All sampled values are "
                            "non-finite"
                        )

                    if (
                        result["finite_percent"]
                        < 99.0
                    ):
                        warnings.append(
                            f"Sample finite percentage is "
                            f"{result['finite_percent']:.3f}%"
                        )

                if errors:
                    result["status"] = "FAILED"
                elif warnings:
                    result["status"] = "WARNING"
                else:
                    result["status"] = "PASSED"

                result["message"] = " | ".join(
                    errors + warnings
                )

                results.append(result)

    except Exception as error:
        message = (
            f"{type(error).__name__}: {error}"
        )

        for variable in VARIABLES:
            results.append(
                {
                    "section": section,
                    "model": model_name,
                    "experiment": "historical",
                    "variable": variable,
                    "file_count": 1,
                    "first_time": "",
                    "last_time": "",
                    "time_count": 0,
                    "lat_count": 0,
                    "lon_count": 0,
                    "lat_min": np.nan,
                    "lat_max": np.nan,
                    "lon_min": np.nan,
                    "lon_max": np.nan,
                    "units": "",
                    "dtype": "",
                    "sample_min": np.nan,
                    "sample_max": np.nan,
                    "finite_percent": np.nan,
                    "status": "FAILED",
                    "message": message,
                    "_grid": None,
                    "_codes": [],
                }
            )

    return results


def find_model_era5_file(model):
    directory = ERA5_GRID_ROOT / model

    files = get_netcdf_files(directory)

    if len(files) != 1:
        return None, (
            f"Expected one ERA5 file, found "
            f"{len(files)} in {directory}"
        )

    return files[0], ""


def build_pair_result(
    model,
    variable,
    cmip_result,
    era_result,
):
    errors = []

    if cmip_result["status"] == "FAILED":
        errors.append(
            "CMIP6 historical data failed"
        )

    if era_result["status"] == "FAILED":
        errors.append(
            "ERA5-on-grid data failed"
        )

    if not grids_equal(
        cmip_result["_grid"],
        era_result["_grid"],
    ):
        errors.append(
            "CMIP6 and ERA5 grids differ"
        )

    cmip_codes = set(
        cmip_result["_codes"]
    )
    era_codes = set(
        era_result["_codes"]
    )

    if not era_codes:
        errors.append(
            "ERA5 time axis is empty"
        )
    elif not era_codes.issubset(cmip_codes):
        missing = len(
            era_codes - cmip_codes
        )
        errors.append(
            f"CMIP6 historical data does not "
            f"cover {missing} ERA5 months"
        )

    if (
        cmip_result["units"]
        != era_result["units"]
    ):
        errors.append(
            f"Unit mismatch: "
            f"CMIP6={cmip_result['units']!r}, "
            f"ERA5={era_result['units']!r}"
        )

    return {
        "section": "PAIR",
        "model": model,
        "experiment": "historical",
        "variable": variable,
        "file_count": 2,
        "first_time": era_result["first_time"],
        "last_time": era_result["last_time"],
        "time_count": era_result["time_count"],
        "lat_count": era_result["lat_count"],
        "lon_count": era_result["lon_count"],
        "lat_min": era_result["lat_min"],
        "lat_max": era_result["lat_max"],
        "lon_min": era_result["lon_min"],
        "lon_max": era_result["lon_max"],
        "units": era_result["units"],
        "dtype": "",
        "sample_min": np.nan,
        "sample_max": np.nan,
        "finite_percent": np.nan,
        "status": (
            "FAILED"
            if errors
            else "PASSED"
        ),
        "message": " | ".join(errors),
        "_grid": None,
        "_codes": [],
    }


def public_row(result):
    return {
        key: value
        for key, value in result.items()
        if not key.startswith("_")
    }


def write_report(results):
    rows = [
        public_row(result)
        for result in results
    ]

    fieldnames = list(rows[0])

    with REPORT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file_object:
        writer = csv.DictWriter(
            file_object,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def print_group_result(result):
    print(
        f"  {result['experiment']:10s} "
        f"{result['variable']:8s} "
        f"{result['first_time']:7s} to "
        f"{result['last_time']:7s}  "
        f"time={result['time_count']:4d}  "
        f"grid={result['lat_count']:3d}x"
        f"{result['lon_count']:3d}  "
        f"{result['status']}"
    )

    if result["message"]:
        print(
            f"    {result['message']}"
        )


def most_common_periods(
    results,
    experiment,
):
    periods = [
        (
            result["first_time"],
            result["last_time"],
            result["time_count"],
        )
        for result in results
        if result["section"] == "CMIP6"
        and result["experiment"] == experiment
        and result["variable"] == "tas"
        and result["status"] != "FAILED"
    ]

    return Counter(periods)


def main():
    print("=" * 110)
    print("EXISTING CMIP6 AND ERA5 FULL DATA AUDIT")
    print("=" * 110)
    print(f"CMIP6 root        : {CMIP6_ROOT}")
    print(f"ERA5 reference    : {ERA5_REFERENCE_FILE}")
    print(f"ERA5 on-grid root : {ERA5_GRID_ROOT}")
    print(f"CAS excluded      : {'CAS-ESM2-0'}")

    models = discover_models()

    print()
    print("DISCOVERED MODELS")
    print("-" * 110)

    for model in models:
        print(f"  {model}")

    print()
    print(
        f"Model count: {len(models)} "
        f"(expected {EXPECTED_EXISTING_MODELS})"
    )

    results = []

    print()
    print("=" * 110)
    print("ERA5 REFERENCE")
    print("=" * 110)

    reference_results = inspect_era5_file(
        ERA5_REFERENCE_FILE,
        "ERA5_REFERENCE",
        "ERA5_REFERENCE",
    )

    results.extend(reference_results)

    for result in reference_results:
        print_group_result(result)

    cmip_lookup = {}
    era_lookup = {}

    print()
    print("=" * 110)
    print("CMIP6 AND ERA5-ON-GCM-GRID")
    print("=" * 110)

    for model in models:
        print()
        print("-" * 110)
        print(model)
        print("-" * 110)

        for experiment in EXPERIMENTS:
            for variable in VARIABLES:
                result = inspect_cmip6_group(
                    model,
                    experiment,
                    variable,
                )

                results.append(result)
                cmip_lookup[
                    (model, experiment, variable)
                ] = result

                print_group_result(result)

        era5_path, era5_error = (
            find_model_era5_file(model)
        )

        if era5_path is None:
            era_results = inspect_era5_file(
                Path("__missing__"),
                model,
                "ERA5_ON_GRID",
            )

            for result in era_results:
                result["message"] = era5_error
        else:
            era_results = inspect_era5_file(
                era5_path,
                model,
                "ERA5_ON_GRID",
            )

        results.extend(era_results)

        print("  ERA5 ON MODEL GRID")

        for result in era_results:
            era_lookup[
                (model, result["variable"])
            ] = result

            print_group_result(result)

    print()
    print("=" * 110)
    print("HISTORICAL CMIP6-ERA5 PAIRS")
    print("=" * 110)

    pair_results = []

    for model in models:
        print()
        print(model)

        for variable in VARIABLES:
            pair_result = build_pair_result(
                model,
                variable,
                cmip_lookup[
                    (model, "historical", variable)
                ],
                era_lookup[(model, variable)],
            )

            pair_results.append(pair_result)
            results.append(pair_result)

            print_group_result(pair_result)

    write_report(results)

    failed = sum(
        result["status"] == "FAILED"
        for result in results
    )
    warnings = sum(
        result["status"] == "WARNING"
        for result in results
    )
    passed = sum(
        result["status"] == "PASSED"
        for result in results
    )

    model_count_ok = (
        len(models)
        == EXPECTED_EXISTING_MODELS
    )

    pair_failed = sum(
        result["status"] == "FAILED"
        for result in pair_results
    )

    print()
    print("=" * 110)
    print("DETECTED PROCESSING CONVENTIONS")
    print("=" * 110)

    for experiment in EXPERIMENTS:
        periods = most_common_periods(
            results,
            experiment,
        )

        print(f"{experiment}:")

        for period, count in periods.most_common():
            first, last, time_count = period
            print(
                f"  {first} to {last}, "
                f"{time_count} months: "
                f"{count} models"
            )

    print()
    print("=" * 110)
    print("AUDIT SUMMARY")
    print("=" * 110)
    print(f"Existing models : {len(models)}")
    print(f"Expected models : {EXPECTED_EXISTING_MODELS}")
    print(f"Passed checks   : {passed}")
    print(f"Warning checks  : {warnings}")
    print(f"Failed checks   : {failed}")
    print(f"Failed pairs    : {pair_failed}")
    print(f"CSV report      : {REPORT_CSV}")

    if (
        not model_count_ok
        or failed > 0
        or pair_failed > 0
    ):
        print()
        print(
            "EXISTING DATA AUDIT DID NOT "
            "FULLY PASS"
        )
        return 1

    print()
    print(
        "ALL EXISTING CMIP6 AND ERA5 "
        "DATA PASSED"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())