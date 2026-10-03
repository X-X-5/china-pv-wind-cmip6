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
VARIABLES = ("tas", "sfcWind", "rsds")
EXPECTED_MONTHS = 672
START_YEAR = 1959
START_MONTH = 1
END_YEAR = 2014
END_MONTH = 12
INTERPOLATION_METHOD = "linear"

DEFAULT_ERA5_FILE = get_path("data_interim_era5_reference") / "ERA5_reference_195901_201412.nc"

DEFAULT_CAS_ROOT = get_path("data_interim_cmip6_china_clipped") / "CAS-ESM2-0"

DEFAULT_OUTPUT_FILE = get_path("data_interim_era5_on_gcm_grid") / "CAS-ESM2-0" / "ERA5_CAS-ESM2-0_195901-201412.nc"

EXPECTED_UNITS = {
    "tas": {"k", "kelvin"},
    "sfcWind": {"ms-1", "ms**-1", "ms^-1"},
    "rsds": {"wm-2", "wm**-2", "wm^-2"},
}

PHYSICAL_LIMITS = {
    "tas": (100.0, 400.0),
    "sfcWind": (-0.1, 200.0),
    "rsds": (-1.0, 1500.0),
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
    return re.sub(r"\s+", "", value)


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

    for coordinate in ("time", "lat", "lon"):
        if dataset[coordinate].ndim != 1:
            raise ValueError(
                f"Coordinate {coordinate} must be one-dimensional"
            )

    return dataset


def normalize_longitude(dataset: xr.Dataset) -> xr.Dataset:
    longitude = np.asarray(dataset["lon"].values, dtype=np.float64)

    if not np.all(np.isfinite(longitude)):
        raise ValueError("Longitude coordinate contains non-finite values")

    if np.nanmax(longitude) > 180.0:
        attributes = dict(dataset["lon"].attrs)
        longitude = ((longitude + 180.0) % 360.0) - 180.0
        dataset = dataset.assign_coords(lon=("lon", longitude))
        dataset["lon"].attrs.update(attributes)

    dataset = dataset.sortby("lon")

    if np.unique(dataset["lon"].values).size != dataset.sizes["lon"]:
        raise ValueError("Longitude coordinate contains duplicate values")

    return dataset


def sort_spatial_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    dataset = normalize_longitude(dataset)
    dataset = dataset.sortby("lat")

    latitude = np.asarray(dataset["lat"].values, dtype=np.float64)

    if not np.all(np.isfinite(latitude)):
        raise ValueError("Latitude coordinate contains non-finite values")

    if np.unique(latitude).size != dataset.sizes["lat"]:
        raise ValueError("Latitude coordinate contains duplicate values")

    return dataset


def expected_month_keys() -> list[int]:
    start_key = START_YEAR * 12 + START_MONTH - 1
    end_key = END_YEAR * 12 + END_MONTH - 1
    return list(range(start_key, end_key + 1))


def actual_month_keys(time_coordinate: xr.DataArray) -> list[int]:
    years = np.asarray(time_coordinate.dt.year.values, dtype=np.int64)
    months = np.asarray(time_coordinate.dt.month.values, dtype=np.int64)
    return (years * 12 + months - 1).astype(np.int64).tolist()


def validate_time(dataset: xr.Dataset, label: str) -> None:
    if dataset.sizes.get("time") != EXPECTED_MONTHS:
        raise ValueError(
            f"{label} has {dataset.sizes.get('time')} time steps; "
            f"expected {EXPECTED_MONTHS}"
        )

    actual = actual_month_keys(dataset["time"])
    expected = expected_month_keys()

    if actual != expected:
        raise ValueError(
            f"{label} time coordinate is incomplete, duplicated, "
            "out of order, or outside 1959-01 to 2014-12"
        )


def find_cas_grid_file(cas_root: Path) -> Path:
    historical_tas_root = cas_root / "historical" / "tas"

    if not historical_tas_root.exists():
        raise FileNotFoundError(
            f"CAS historical tas directory does not exist: "
            f"{historical_tas_root}"
        )

    matches = sorted(
        path
        for path in historical_tas_root.rglob(
            "tas_Amon_CAS-ESM2-0_historical_*.nc"
        )
        if path.is_file()
    )

    if not matches:
        raise FileNotFoundError(
            f"No processed CAS historical tas file was found under "
            f"{historical_tas_root}"
        )

    if len(matches) > 1:
        formatted = "\n".join(f"    {path}" for path in matches)
        raise RuntimeError(
            "Multiple processed CAS historical tas files were found:\n"
            f"{formatted}"
        )

    return matches[0]


def load_target_grid(cas_grid_file: Path) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(cas_grid_file, decode_times=True) as dataset:
        dataset = standardize_coordinates(dataset)
        dataset = sort_spatial_coordinates(dataset)
        validate_time(dataset, "CAS historical grid file")

        target_latitude = np.asarray(
            dataset["lat"].values,
            dtype=np.float64,
        ).copy()
        target_longitude = np.asarray(
            dataset["lon"].values,
            dtype=np.float64,
        ).copy()

    if target_latitude.size < 2 or target_longitude.size < 2:
        raise ValueError("CAS target grid contains too few spatial cells")

    return target_latitude, target_longitude


def validate_source_coverage(
    source: xr.Dataset,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> None:
    source_latitude = np.asarray(source["lat"].values, dtype=np.float64)
    source_longitude = np.asarray(source["lon"].values, dtype=np.float64)
    tolerance = 1.0e-8

    if target_latitude.min() < source_latitude.min() - tolerance:
        raise ValueError("CAS latitude minimum is outside the ERA5 domain")
    if target_latitude.max() > source_latitude.max() + tolerance:
        raise ValueError("CAS latitude maximum is outside the ERA5 domain")
    if target_longitude.min() < source_longitude.min() - tolerance:
        raise ValueError("CAS longitude minimum is outside the ERA5 domain")
    if target_longitude.max() > source_longitude.max() + tolerance:
        raise ValueError("CAS longitude maximum is outside the ERA5 domain")


def validate_values(
    data_array: xr.DataArray,
    variable: str,
) -> dict[str, float | int]:
    values = np.asarray(data_array.values)
    finite_mask = np.isfinite(values)
    finite_count = int(finite_mask.sum())
    total_count = int(values.size)

    if total_count == 0:
        raise ValueError(f"Variable {variable} contains no values")

    if finite_count == 0:
        raise ValueError(f"Variable {variable} contains no finite values")

    if finite_count != total_count:
        missing_count = total_count - finite_count
        raise ValueError(
            f"Variable {variable} contains {missing_count} "
            "non-finite values after interpolation"
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


def create_regridded_dataset(
    era5_file: Path,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> tuple[xr.Dataset, dict[str, dict[str, float | int]]]:
    target_latitude_array = xr.DataArray(
        target_latitude,
        dims="lat",
        coords={"lat": target_latitude},
    )
    target_longitude_array = xr.DataArray(
        target_longitude,
        dims="lon",
        coords={"lon": target_longitude},
    )

    output_variables = {}
    statistics = {}

    with xr.open_dataset(era5_file, decode_times=True) as source:
        source = standardize_coordinates(source)
        source = sort_spatial_coordinates(source)
        validate_time(source, "ERA5 reference file")
        validate_source_coverage(
            source,
            target_latitude,
            target_longitude,
        )

        missing_variables = [
            variable for variable in VARIABLES
            if variable not in source.data_vars
        ]

        if missing_variables:
            raise ValueError(
                "ERA5 reference file is missing variables: "
                + ", ".join(missing_variables)
            )

        source_time = source["time"].load().copy(deep=True)
        source_time_encoding = dict(source["time"].encoding)

        for variable in VARIABLES:
            validate_unit(variable, source[variable].attrs.get("units"))

            print(f"  Interpolating {variable} with method={INTERPOLATION_METHOD}")

            interpolated = source[variable].interp(
                lat=target_latitude_array,
                lon=target_longitude_array,
                method=INTERPOLATION_METHOD,
            )

            extra_dimensions = [
                dimension
                for dimension in interpolated.dims
                if dimension not in {"time", "lat", "lon"}
            ]

            for dimension in extra_dimensions:
                if interpolated.sizes[dimension] != 1:
                    raise ValueError(
                        f"Variable {variable} has unsupported dimension "
                        f"{dimension!r} with size "
                        f"{interpolated.sizes[dimension]}"
                    )

            if extra_dimensions:
                interpolated = interpolated.squeeze(
                    dim=extra_dimensions,
                    drop=True,
                )

            interpolated = interpolated.transpose("time", "lat", "lon")
            interpolated = interpolated.astype(np.float32).load()

            for attribute in (
                "_FillValue",
                "missing_value",
                "scale_factor",
                "add_offset",
            ):
                interpolated.attrs.pop(attribute, None)

            interpolated.attrs.update(
                {
                    "regridded_to": MODEL,
                    "interpolation_method": INTERPOLATION_METHOD,
                }
            )

            statistics[variable] = validate_values(
                interpolated,
                variable,
            )
            output_variables[variable] = interpolated

    output = xr.Dataset(output_variables)
    output = output.assign_coords(time=source_time)
    output["time"].encoding = source_time_encoding

    output["lat"].attrs.update(
        {
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        }
    )
    output["lon"].attrs.update(
        {
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        }
    )

    output.attrs.update(
        {
            "title": "ERA5 monthly reference data on the CAS-ESM2-0 grid",
            "source": "ERA5 monthly reference dataset",
            "target_model": MODEL,
            "period": "1959-01 to 2014-12",
            "variables": ", ".join(VARIABLES),
            "spatial_regridding": (
                "Rectilinear latitude-longitude interpolation using "
                f"xarray.interp with method={INTERPOLATION_METHOD}"
            ),
            "temporal_regridding": "none",
        }
    )

    return output, statistics


def build_encoding(dataset: xr.Dataset) -> dict:
    encoding = {}

    for variable in VARIABLES:
        dataset[variable].encoding = {}
        encoding[variable] = {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "_FillValue": np.float32(1.0e20),
        }

    encoding["lat"] = {
        "dtype": "float64",
        "_FillValue": None,
    }
    encoding["lon"] = {
        "dtype": "float64",
        "_FillValue": None,
    }

    return encoding


def validate_output(
    output_file: Path,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    statistics = {}

    with xr.open_dataset(output_file, decode_times=True) as dataset:
        dataset = standardize_coordinates(dataset)
        validate_time(dataset, "ERA5-on-CAS output file")

        for variable in VARIABLES:
            if variable not in dataset.data_vars:
                raise ValueError(
                    f"Output file is missing variable {variable}"
                )

            if dataset[variable].dims != ("time", "lat", "lon"):
                raise ValueError(
                    f"Output variable {variable} has dimensions "
                    f"{dataset[variable].dims}; expected "
                    "('time', 'lat', 'lon')"
                )

            validate_unit(variable, dataset[variable].attrs.get("units"))
            statistics[variable] = validate_values(
                dataset[variable],
                variable,
            )

        output_latitude = np.asarray(
            dataset["lat"].values,
            dtype=np.float64,
        )
        output_longitude = np.asarray(
            dataset["lon"].values,
            dtype=np.float64,
        )

        if not np.array_equal(output_latitude, target_latitude):
            raise ValueError(
                "Output latitude coordinate does not exactly match "
                "the CAS target grid"
            )

        if not np.array_equal(output_longitude, target_longitude):
            raise ValueError(
                "Output longitude coordinate does not exactly match "
                "the CAS target grid"
            )

    return statistics


def write_output(
    dataset: xr.Dataset,
    output_file: Path,
    overwrite: bool,
) -> None:
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}. "
            "Use --overwrite to replace it."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")

    if temporary_file.exists():
        temporary_file.unlink()

    try:
        dataset.to_netcdf(
            temporary_file,
            mode="w",
            engine="netcdf4",
            format="NETCDF4",
            encoding=build_encoding(dataset),
        )
        os.replace(temporary_file, output_file)

    except Exception:
        if temporary_file.exists():
            temporary_file.unlink()
        raise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate the 1959-2014 ERA5 monthly reference dataset "
            "to the processed CAS-ESM2-0 native grid."
        )
    )
    parser.add_argument(
        "--era5-file",
        type=Path,
        default=DEFAULT_ERA5_FILE,
        help="Combined ERA5 monthly reference NetCDF file.",
    )
    parser.add_argument(
        "--cas-root",
        type=Path,
        default=DEFAULT_CAS_ROOT,
        help="Processed CAS-ESM2-0 root directory.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="ERA5-on-CAS output NetCDF file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    era5_file = arguments.era5_file.resolve()
    cas_root = arguments.cas_root.resolve()
    output_file = arguments.output_file.resolve()

    print_separator()
    print("ERA5 TO CAS-ESM2-0 GRID REGRIDDING")
    print_separator()
    print(f"ERA5 source : {era5_file}")
    print(f"CAS root    : {cas_root}")
    print(f"Output file : {output_file}")
    print(f"Period      : 1959-01 to 2014-12 ({EXPECTED_MONTHS} months)")
    print(f"Variables   : {', '.join(VARIABLES)}")
    print(f"Method      : {INTERPOLATION_METHOD}")
    print(f"Overwrite   : {arguments.overwrite}")
    print()

    if not era5_file.is_file():
        print(f"ERROR: ERA5 reference file does not exist: {era5_file}")
        return 1

    if not cas_root.is_dir():
        print(f"ERROR: CAS processed root does not exist: {cas_root}")
        return 1

    try:
        cas_grid_file = find_cas_grid_file(cas_root)
        print(f"CAS grid file: {cas_grid_file}")

        target_latitude, target_longitude = load_target_grid(cas_grid_file)

        print(
            "Target grid  : "
            f"lat={target_latitude.size}, lon={target_longitude.size}"
        )
        print(
            "Target domain: "
            f"{target_longitude.min():.6f} to "
            f"{target_longitude.max():.6f} E, "
            f"{target_latitude.min():.6f} to "
            f"{target_latitude.max():.6f} N"
        )
        print()

        if output_file.exists() and not arguments.overwrite:
            print("Output file already exists; validating it without rewriting.")
            statistics = validate_output(
                output_file,
                target_latitude,
                target_longitude,
            )
            status = "VALID EXISTING FILE"

        else:
            output, statistics = create_regridded_dataset(
                era5_file,
                target_latitude,
                target_longitude,
            )
            write_output(output, output_file, arguments.overwrite)
            statistics = validate_output(
                output_file,
                target_latitude,
                target_longitude,
            )
            status = "CREATED AND VALIDATED"

        print()
        print_separator()
        print("REGRIDDING SUMMARY")
        print_separator()
        print(f"Status      : {status}")
        print(f"Output file : {output_file}")
        print(f"Time steps  : {EXPECTED_MONTHS}")
        print(
            "Target grid : "
            f"lat={target_latitude.size}, lon={target_longitude.size}"
        )

        for variable in VARIABLES:
            item = statistics[variable]
            print(
                f"{variable:8s}: "
                f"min={item['minimum']:.6g}, "
                f"mean={item['mean']:.6g}, "
                f"max={item['maximum']:.6g}, "
                f"finite={item['finite_percent']:.6f}%"
            )

        print()
        print("ERA5 WAS SUCCESSFULLY REGRIDDED TO THE CAS-ESM2-0 GRID")
        return 0

    except Exception as error:
        print()
        print_separator()
        print("REGRIDDING FAILED")
        print_separator()
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
