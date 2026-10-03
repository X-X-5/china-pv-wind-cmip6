from __future__ import annotations
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

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr


MODEL = "CAS-ESM2-0"
VARIANT = "r1i1p1f1"
TABLE = "Amon"
GRID = "gn"

DEFAULT_RAW_ROOT = get_path("data_raw_cmip6_downloaded") / "CAS-ESM2-0"

DEFAULT_OUTPUT_ROOT = get_path("data_interim_cmip6_china_clipped") / "CAS-ESM2-0"

LON_MIN = 70.0
LON_MAX = 140.0
LAT_MIN = 15.0
LAT_MAX = 54.0

VARIABLES = ("tas", "rsds", "sfcWind")

EXPERIMENTS = {
    "historical": {
        "start": (1959, 1),
        "end": (2014, 12),
        "expected_months": 672,
        "output_period": "195901-201412",
    },
    "ssp126": {
        "start": (2015, 1),
        "end": (2100, 12),
        "expected_months": 1032,
        "output_period": "201501-210012",
    },
    "ssp245": {
        "start": (2015, 1),
        "end": (2100, 12),
        "expected_months": 1032,
        "output_period": "201501-210012",
    },
    "ssp585": {
        "start": (2015, 1),
        "end": (2100, 12),
        "expected_months": 1032,
        "output_period": "201501-210012",
    },
}

EXPECTED_UNITS = {
    "tas": {"k", "kelvin"},
    "rsds": {"wm-2", "wm**-2", "wm^-2"},
    "sfcWind": {"ms-1", "ms**-1", "ms^-1"},
}

PHYSICAL_LIMITS = {
    "tas": (100.0, 400.0),
    "rsds": (-1.0, 1500.0),
    "sfcWind": (-0.1, 200.0),
}


def print_separator(character: str = "=", width: int = 100) -> None:
    print(character * width)


def normalize_unit(unit: str | None) -> str:
    if unit is None:
        return ""

    value = str(unit).strip().lower()
    value = value.replace("watts", "w")
    value = value.replace("watt", "w")
    value = value.replace("metres", "m")
    value = value.replace("metre", "m")
    value = value.replace("meters", "m")
    value = value.replace("meter", "m")
    value = value.replace("seconds", "s")
    value = value.replace("second", "s")
    value = value.replace("degrees_kelvin", "kelvin")
    value = value.replace("degree_kelvin", "kelvin")
    value = re.sub(r"\s+", "", value)
    return value


def validate_unit(variable: str, unit: str | None) -> None:
    normalized = normalize_unit(unit)

    if normalized not in EXPECTED_UNITS[variable]:
        accepted = ", ".join(sorted(EXPECTED_UNITS[variable]))
        raise ValueError(
            f"Unexpected units for {variable}: {unit!r}. "
            f"Accepted normalized units: {accepted}"
        )


def find_coordinate_name(
    dataset: xr.Dataset,
    candidates: tuple[str, ...],
    coordinate_type: str,
) -> str:
    for name in candidates:
        if name in dataset.coords or name in dataset.dims:
            return name

    for name in dataset.coords:
        standard_name = str(
            dataset[name].attrs.get("standard_name", "")
        ).lower()

        axis = str(dataset[name].attrs.get("axis", "")).upper()

        if coordinate_type == "longitude":
            if standard_name == "longitude" or axis == "X":
                return name

        if coordinate_type == "latitude":
            if standard_name == "latitude" or axis == "Y":
                return name

        if coordinate_type == "time":
            if standard_name == "time" or axis == "T":
                return name

    raise ValueError(f"Could not identify the {coordinate_type} coordinate")


def build_expected_month_keys(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[int]:
    start_key = start[0] * 12 + start[1] - 1
    end_key = end[0] * 12 + end[1] - 1
    return list(range(start_key, end_key + 1))


def get_month_keys(time_coordinate: xr.DataArray) -> list[int]:
    years = np.asarray(time_coordinate.dt.year.values, dtype=np.int64)
    months = np.asarray(time_coordinate.dt.month.values, dtype=np.int64)

    return (
        years * 12 + months - 1
    ).astype(np.int64).tolist()


def month_key_to_text(month_key: int) -> str:
    year = month_key // 12
    month = month_key % 12 + 1
    return f"{year:04d}-{month:02d}"


def find_source_file(
    raw_root: Path,
    experiment: str,
    variable: str,
) -> Path:
    pattern = (
        f"{variable}_{TABLE}_{MODEL}_{experiment}_"
        f"{VARIANT}_{GRID}_*.nc"
    )

    experiment_root = raw_root / experiment

    if not experiment_root.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist: {experiment_root}"
        )

    matches = sorted(
        path
        for path in experiment_root.rglob(pattern)
        if path.is_file()
    )

    if not matches:
        raise FileNotFoundError(
            f"No source file matched {pattern!r} under {experiment_root}"
        )

    if len(matches) > 1:
        formatted = "\n".join(f"    {path}" for path in matches)
        raise RuntimeError(
            f"Multiple source files matched {experiment}/{variable}:\n"
            f"{formatted}"
        )

    return matches[0]


def make_output_path(
    output_root: Path,
    experiment: str,
    variable: str,
    output_period: str,
) -> Path:
    filename = (
        f"{variable}_{TABLE}_{MODEL}_{experiment}_"
        f"{VARIANT}_{GRID}_{output_period}.nc"
    )

    return output_root / experiment / variable / filename


def standardize_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    longitude_name = find_coordinate_name(
        dataset,
        ("lon", "longitude", "x"),
        "longitude",
    )

    latitude_name = find_coordinate_name(
        dataset,
        ("lat", "latitude", "y"),
        "latitude",
    )

    time_name = find_coordinate_name(
        dataset,
        ("time",),
        "time",
    )

    rename_map = {}

    if longitude_name != "lon":
        rename_map[longitude_name] = "lon"

    if latitude_name != "lat":
        rename_map[latitude_name] = "lat"

    if time_name != "time":
        rename_map[time_name] = "time"

    if rename_map:
        dataset = dataset.rename(rename_map)

    if dataset["lon"].ndim != 1:
        raise ValueError("Only one-dimensional longitude coordinates are supported")

    if dataset["lat"].ndim != 1:
        raise ValueError("Only one-dimensional latitude coordinates are supported")

    if dataset["time"].ndim != 1:
        raise ValueError("Only one-dimensional time coordinates are supported")

    return dataset


def normalize_longitude(dataset: xr.Dataset) -> xr.Dataset:
    longitude_values = np.asarray(dataset["lon"].values, dtype=np.float64)

    if np.nanmax(longitude_values) > 180.0:
        longitude_attributes = dict(dataset["lon"].attrs)

        normalized_values = (
            (longitude_values + 180.0) % 360.0
        ) - 180.0

        dataset = dataset.assign_coords(
            lon=("lon", normalized_values)
        )

        dataset["lon"].attrs.update(longitude_attributes)

    return dataset.sortby("lon")


def select_time_period(
    dataset: xr.Dataset,
    start: tuple[int, int],
    end: tuple[int, int],
    expected_months: int,
) -> xr.Dataset:
    month_keys = np.asarray(
        get_month_keys(dataset["time"]),
        dtype=np.int64,
    )

    start_key = start[0] * 12 + start[1] - 1
    end_key = end[0] * 12 + end[1] - 1

    indices = np.flatnonzero(
        (month_keys >= start_key) & (month_keys <= end_key)
    )

    if indices.size == 0:
        raise ValueError(
            f"No time steps were found between "
            f"{start[0]:04d}-{start[1]:02d} and "
            f"{end[0]:04d}-{end[1]:02d}"
        )

    selected = dataset.isel(time=indices)

    actual_keys = get_month_keys(selected["time"])
    expected_keys = build_expected_month_keys(start, end)

    if actual_keys != expected_keys:
        actual_start = (
            month_key_to_text(actual_keys[0])
            if actual_keys else "none"
        )

        actual_end = (
            month_key_to_text(actual_keys[-1])
            if actual_keys else "none"
        )

        raise ValueError(
            "Time coordinate is incomplete, duplicated, or out of order. "
            f"Expected {expected_months} months from "
            f"{start[0]:04d}-{start[1]:02d} to "
            f"{end[0]:04d}-{end[1]:02d}; found "
            f"{len(actual_keys)} months from {actual_start} to {actual_end}"
        )

    if selected.sizes["time"] != expected_months:
        raise ValueError(
            f"Expected {expected_months} time steps, "
            f"but found {selected.sizes['time']}"
        )

    return selected


def select_spatial_domain(dataset: xr.Dataset) -> xr.Dataset:
    dataset = normalize_longitude(dataset)
    dataset = dataset.sortby("lat")

    longitude_values = np.asarray(
        dataset["lon"].values,
        dtype=np.float64,
    )

    latitude_values = np.asarray(
        dataset["lat"].values,
        dtype=np.float64,
    )

    longitude_indices = np.flatnonzero(
        (longitude_values >= LON_MIN)
        & (longitude_values <= LON_MAX)
    )

    latitude_indices = np.flatnonzero(
        (latitude_values >= LAT_MIN)
        & (latitude_values <= LAT_MAX)
    )

    if longitude_indices.size == 0:
        raise ValueError(
            f"No longitude cells were found inside "
            f"{LON_MIN}-{LON_MAX} degrees east"
        )

    if latitude_indices.size == 0:
        raise ValueError(
            f"No latitude cells were found inside "
            f"{LAT_MIN}-{LAT_MAX} degrees north"
        )

    selected = dataset.isel(
        lon=longitude_indices,
        lat=latitude_indices,
    )

    return selected


def remove_singleton_extra_dimensions(
    dataset: xr.Dataset,
    variable: str,
) -> xr.Dataset:
    required_dimensions = {"time", "lat", "lon"}

    extra_dimensions = [
        dimension
        for dimension in dataset[variable].dims
        if dimension not in required_dimensions
    ]

    for dimension in extra_dimensions:
        if dataset.sizes[dimension] != 1:
            raise ValueError(
                f"Variable {variable} has unsupported dimension "
                f"{dimension!r} with size {dataset.sizes[dimension]}"
            )

    if extra_dimensions:
        dataset = dataset.squeeze(
            dim=extra_dimensions,
            drop=True,
        )

    actual_dimensions = set(dataset[variable].dims)

    if actual_dimensions != required_dimensions:
        raise ValueError(
            f"Variable {variable} dimensions are "
            f"{dataset[variable].dims}; expected time, lat, and lon"
        )

    return dataset.transpose("time", "lat", "lon")


def validate_data_values(
    dataset: xr.Dataset,
    variable: str,
) -> dict[str, float | int]:
    values = np.asarray(dataset[variable].values)

    finite_mask = np.isfinite(values)
    finite_count = int(finite_mask.sum())
    total_count = int(values.size)

    if total_count == 0:
        raise ValueError(f"Variable {variable} contains no values")

    if finite_count == 0:
        raise ValueError(f"Variable {variable} contains no finite values")

    positive_infinity_count = int(np.isposinf(values).sum())
    negative_infinity_count = int(np.isneginf(values).sum())

    if positive_infinity_count or negative_infinity_count:
        raise ValueError(
            f"Variable {variable} contains infinite values"
        )

    finite_values = values[finite_mask]

    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    mean = float(np.mean(finite_values))

    lower_limit, upper_limit = PHYSICAL_LIMITS[variable]

    if minimum < lower_limit or maximum > upper_limit:
        raise ValueError(
            f"Variable {variable} is outside its basic physical range: "
            f"minimum={minimum:.6g}, maximum={maximum:.6g}, "
            f"accepted={lower_limit:.6g} to {upper_limit:.6g}"
        )

    return {
        "total_count": total_count,
        "finite_count": finite_count,
        "finite_percent": 100.0 * finite_count / total_count,
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
    }


def prepare_encoding(
    dataset: xr.Dataset,
    variable: str,
    time_encoding: dict,
) -> dict:
    for name in dataset.variables:
        dataset[name].encoding = {}

    encoding = {
        variable: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "_FillValue": np.float32(1.0e20),
        },
        "lat": {
            "dtype": "float64",
            "_FillValue": None,
        },
        "lon": {
            "dtype": "float64",
            "_FillValue": None,
        },
    }

    output_time_encoding = {
        "dtype": "float64",
        "_FillValue": None,
    }

    if time_encoding.get("units"):
        output_time_encoding["units"] = time_encoding["units"]

    if time_encoding.get("calendar"):
        output_time_encoding["calendar"] = time_encoding["calendar"]

    encoding["time"] = output_time_encoding

    return encoding


def validate_output_file(
    path: Path,
    experiment: str,
    variable: str,
) -> dict[str, float | int]:
    settings = EXPERIMENTS[experiment]

    with xr.open_dataset(path, decode_times=True) as dataset:
        dataset = standardize_coordinates(dataset)

        if variable not in dataset.data_vars:
            raise ValueError(
                f"Output file does not contain variable {variable}"
            )

        if dataset.sizes.get("time") != settings["expected_months"]:
            raise ValueError(
                f"Output time size is {dataset.sizes.get('time')}; "
                f"expected {settings['expected_months']}"
            )

        actual_keys = get_month_keys(dataset["time"])

        expected_keys = build_expected_month_keys(
            settings["start"],
            settings["end"],
        )

        if actual_keys != expected_keys:
            raise ValueError(
                "Output time coordinate does not match the expected period"
            )

        longitude_values = np.asarray(
            dataset["lon"].values,
            dtype=np.float64,
        )

        latitude_values = np.asarray(
            dataset["lat"].values,
            dtype=np.float64,
        )

        tolerance = 1.0e-6

        if longitude_values.min() < LON_MIN - tolerance:
            raise ValueError(
                "Output contains longitude values below the domain"
            )

        if longitude_values.max() > LON_MAX + tolerance:
            raise ValueError(
                "Output contains longitude values above the domain"
            )

        if latitude_values.min() < LAT_MIN - tolerance:
            raise ValueError(
                "Output contains latitude values below the domain"
            )

        if latitude_values.max() > LAT_MAX + tolerance:
            raise ValueError(
                "Output contains latitude values above the domain"
            )

        validate_unit(
            variable,
            dataset[variable].attrs.get("units"),
        )

        statistics = validate_data_values(dataset, variable)

        statistics["time_count"] = int(dataset.sizes["time"])
        statistics["lat_count"] = int(dataset.sizes["lat"])
        statistics["lon_count"] = int(dataset.sizes["lon"])
        statistics["lat_min"] = float(latitude_values.min())
        statistics["lat_max"] = float(latitude_values.max())
        statistics["lon_min"] = float(longitude_values.min())
        statistics["lon_max"] = float(longitude_values.max())

        return statistics


def process_file(
    source_path: Path,
    destination_path: Path,
    experiment: str,
    variable: str,
    overwrite: bool,
) -> tuple[str, dict[str, float | int]]:
    settings = EXPERIMENTS[experiment]

    if destination_path.exists() and not overwrite:
        statistics = validate_output_file(
            destination_path,
            experiment,
            variable,
        )
        return "existing", statistics

    with xr.open_dataset(source_path, decode_times=True) as source:
        source = standardize_coordinates(source)

        if variable not in source.data_vars:
            raise ValueError(
                f"Source file does not contain variable {variable}"
            )

        validate_unit(
            variable,
            source[variable].attrs.get("units"),
        )

        original_time_encoding = dict(source["time"].encoding)

        output = source[[variable]]

        output = remove_singleton_extra_dimensions(
            output,
            variable,
        )

        output = select_time_period(
            output,
            settings["start"],
            settings["end"],
            settings["expected_months"],
        )

        output = select_spatial_domain(output)

        output[variable] = output[variable].astype(np.float32)

        for attribute in (
            "_FillValue",
            "missing_value",
            "scale_factor",
            "add_offset",
        ):
            output[variable].attrs.pop(attribute, None)

        output.attrs.update(
            {
                "title": (
                    f"{MODEL} monthly data cropped to the "
                    "China analysis domain"
                ),
                "source_model": MODEL,
                "source_experiment": experiment,
                "variant_label": VARIANT,
                "table_id": TABLE,
                "grid_label": GRID,
                "processing": (
                    "Time selection, coordinate standardization, "
                    "longitude normalization, and spatial subsetting"
                ),
                "processing_time_period": (
                    f"{settings['start'][0]:04d}-"
                    f"{settings['start'][1]:02d} to "
                    f"{settings['end'][0]:04d}-"
                    f"{settings['end'][1]:02d}"
                ),
                "processing_domain": (
                    f"{LON_MIN}-{LON_MAX} degrees east, "
                    f"{LAT_MIN}-{LAT_MAX} degrees north"
                ),
                "regridded": "no",
            }
        )

        output = output.load()

    statistics = validate_data_values(output, variable)

    longitude_values = np.asarray(
        output["lon"].values,
        dtype=np.float64,
    )

    latitude_values = np.asarray(
        output["lat"].values,
        dtype=np.float64,
    )

    output.attrs["geospatial_lon_min"] = float(
        longitude_values.min()
    )
    output.attrs["geospatial_lon_max"] = float(
        longitude_values.max()
    )
    output.attrs["geospatial_lat_min"] = float(
        latitude_values.min()
    )
    output.attrs["geospatial_lat_max"] = float(
        latitude_values.max()
    )

    encoding = prepare_encoding(
        output,
        variable,
        original_time_encoding,
    )

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination_path.with_suffix(
        destination_path.suffix + ".tmp"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    try:
        output.to_netcdf(
            temporary_path,
            mode="w",
            engine="netcdf4",
            format="NETCDF4",
            encoding=encoding,
        )

        os.replace(temporary_path, destination_path)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    statistics = validate_output_file(
        destination_path,
        experiment,
        variable,
    )

    return "processed", statistics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reprocess CAS-ESM2-0 monthly CMIP6 files for "
            "the China analysis domain."
        )
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="CAS-ESM2-0 raw-data root directory.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="CAS-ESM2-0 processed output directory.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace matching output files if they already exist.",
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    raw_root = arguments.raw_root.resolve()
    output_root = arguments.output_root.resolve()

    print_separator()
    print("CAS-ESM2-0 CHINA-DOMAIN REPROCESSING")
    print_separator()
    print(f"Raw root    : {raw_root}")
    print(f"Output root : {output_root}")
    print(f"Model       : {MODEL}")
    print(f"Variant     : {VARIANT}")
    print(f"Table       : {TABLE}")
    print(f"Grid        : {GRID}")
    print(
        f"Domain      : {LON_MIN}-{LON_MAX} E, "
        f"{LAT_MIN}-{LAT_MAX} N"
    )
    print("Historical  : 1959-01 to 2014-12")
    print("Scenarios   : 2015-01 to 2100-12")
    print(f"Overwrite   : {arguments.overwrite}")
    print()

    if not raw_root.exists():
        print(f"ERROR: Raw root does not exist: {raw_root}")
        return 1

    expected_files = len(EXPERIMENTS) * len(VARIABLES)
    processed_files = 0
    existing_files = 0
    failed_files = 0
    expected_output_paths: set[Path] = set()

    item_number = 0

    for experiment, settings in EXPERIMENTS.items():
        for variable in VARIABLES:
            item_number += 1

            print_separator("-")
            print(
                f"[{item_number}/{expected_files}] "
                f"{MODEL} {experiment} {variable}"
            )
            print_separator("-")

            try:
                source_path = find_source_file(
                    raw_root,
                    experiment,
                    variable,
                )

                destination_path = make_output_path(
                    output_root,
                    experiment,
                    variable,
                    settings["output_period"],
                )

                expected_output_paths.add(
                    destination_path.resolve()
                )

                print(f"Source : {source_path}")
                print(f"Output : {destination_path}")

                status, statistics = process_file(
                    source_path=source_path,
                    destination_path=destination_path,
                    experiment=experiment,
                    variable=variable,
                    overwrite=arguments.overwrite,
                )

                if status == "processed":
                    processed_files += 1
                    print("STATUS : PROCESSED")

                else:
                    existing_files += 1
                    print("STATUS : VALID EXISTING FILE")

                print(
                    "Shape  : "
                    f"time={statistics['time_count']}, "
                    f"lat={statistics['lat_count']}, "
                    f"lon={statistics['lon_count']}"
                )

                print(
                    "Domain : "
                    f"{statistics['lon_min']:.6f} to "
                    f"{statistics['lon_max']:.6f} E, "
                    f"{statistics['lat_min']:.6f} to "
                    f"{statistics['lat_max']:.6f} N"
                )

                print(
                    "Values : "
                    f"min={statistics['minimum']:.6g}, "
                    f"mean={statistics['mean']:.6g}, "
                    f"max={statistics['maximum']:.6g}, "
                    f"finite={statistics['finite_percent']:.6f}%"
                )

            except Exception as error:
                failed_files += 1
                print(f"STATUS : FAILED")
                print(f"ERROR  : {error}")

    unexpected_files: list[Path] = []

    if output_root.exists():
        for path in output_root.rglob("*.nc"):
            resolved_path = path.resolve()

            if resolved_path not in expected_output_paths:
                unexpected_files.append(path)

    print()
    print_separator()
    print("REPROCESSING SUMMARY")
    print_separator()
    print(f"Expected files : {expected_files}")
    print(f"Processed files: {processed_files}")
    print(f"Existing files : {existing_files}")
    print(f"Failed files   : {failed_files}")
    print(f"Unexpected files: {len(unexpected_files)}")

    if unexpected_files:
        print()
        print("UNEXPECTED NETCDF FILES")
        print_separator("-")

        for path in unexpected_files:
            print(path)

        print()
        print(
            "These files were not deleted automatically. "
            "Check whether they are outputs from an earlier "
            "incorrect preprocessing run."
        )

    if failed_files == 0 and not unexpected_files:
        print()
        print(
            "ALL 12 CAS-ESM2-0 FILES WERE "
            "REPROCESSED AND VALIDATED SUCCESSFULLY"
        )
        return 0

    print()
    print(
        "CAS-ESM2-0 REPROCESSING DID NOT PASS "
        "THE FINAL CHECK"
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())