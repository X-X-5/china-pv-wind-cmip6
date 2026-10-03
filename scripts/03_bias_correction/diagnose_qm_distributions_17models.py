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
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = PROJECT_ROOT

CMIP6_ROOT = get_path("data_interim_cmip6_china_clipped")
ERA5_GRID_ROOT = (
    get_path("data_interim_era5_on_gcm_grid")
)
OUTPUT_ROOT = get_path("results_audits") / "qm_distribution_diagnostics_17models"

MODELS = (
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "AWI-CM-1-1-MR",
    "BCC-CSM2-MR",
    "CAS-ESM2-0",
    "CESM2-WACCM",
    "CMCC-CM2-SR5",
    "CMCC-ESM2",
    "CanESM5",
    "FGOALS-f3-L",
    "FIO-ESM-2-0",
    "IPSL-CM6A-LR",
    "KACE-1-0-G",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "TaiESM1",
)

VARIABLES = ("tas", "rsds", "sfcWind")
MONTHS = tuple(range(1, 13))
EXPECTED_TIME = 672
EXPECTED_YEARS_PER_MONTH = 56
START_YEAR = 1959
END_YEAR = 2014
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

NEAR_ZERO_THRESHOLDS = {
    "tas": 1.0,
    "rsds": 1.0,
    "sfcWind": 0.1,
}

EXPECTED_UNITS = {
    "tas": {"k", "kelvin"},
    "rsds": {"wm-2", "wm**-2", "wm^-2", "w/m2"},
    "sfcWind": {"ms-1", "ms**-1", "ms^-1", "m/s"},
}

MONTHLY_REPORT_NAME = "qm_monthly_distribution_diagnostics.csv"
MODEL_VARIABLE_REPORT_NAME = "qm_model_variable_recommendations.csv"
RUN_SUMMARY_NAME = "qm_distribution_diagnostics_summary.txt"


def print_separator(character: str = "=", width: int = 110) -> None:
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
    return "".join(value.split())


def validate_unit(variable: str, unit: str | None, label: str) -> None:
    normalized = normalize_unit(unit)
    if normalized not in EXPECTED_UNITS[variable]:
        raise ValueError(
            f"Unexpected units for {label} {variable}: {unit!r}"
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
        elif coordinate_type == "latitude":
            if standard_name == "latitude" or axis == "Y":
                return name
        elif coordinate_type == "time":
            if standard_name == "time" or axis == "T":
                return name

    raise ValueError(f"Could not identify the {coordinate_type} coordinate")


def standardize_dataset(dataset: xr.Dataset) -> xr.Dataset:
    lon_name = find_coordinate_name(
        dataset, ("lon", "longitude", "x"), "longitude"
    )
    lat_name = find_coordinate_name(
        dataset, ("lat", "latitude", "y"), "latitude"
    )
    time_name = find_coordinate_name(dataset, ("time",), "time")

    rename_map = {}
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if time_name != "time":
        rename_map[time_name] = "time"
    if rename_map:
        dataset = dataset.rename(rename_map)

    for coordinate in ("time", "lat", "lon"):
        if dataset[coordinate].ndim != 1:
            raise ValueError(
                f"Coordinate {coordinate} must be one-dimensional"
            )

    lon = np.asarray(dataset["lon"].values, dtype=np.float64)
    if np.nanmax(lon) > 180.0:
        lon_attrs = dict(dataset["lon"].attrs)
        lon = ((lon + 180.0) % 360.0) - 180.0
        dataset = dataset.assign_coords(lon=("lon", lon))
        dataset["lon"].attrs.update(lon_attrs)

    return dataset.sortby("lat").sortby("lon")


def month_keys(time_coordinate: xr.DataArray) -> list[int]:
    years = np.asarray(time_coordinate.dt.year.values, dtype=np.int64)
    months = np.asarray(time_coordinate.dt.month.values, dtype=np.int64)
    return (years * 12 + months - 1).astype(np.int64).tolist()


def expected_month_keys() -> list[int]:
    return list(
        range(
            START_YEAR * 12,
            END_YEAR * 12 + 12,
        )
    )


def validate_time(dataset: xr.Dataset, label: str) -> None:
    if dataset.sizes.get("time") != EXPECTED_TIME:
        raise ValueError(
            f"{label} has {dataset.sizes.get('time')} time steps; "
            f"expected {EXPECTED_TIME}"
        )
    if month_keys(dataset["time"]) != expected_month_keys():
        raise ValueError(
            f"{label} does not contain a complete monthly sequence "
            f"from {START_YEAR}-01 to {END_YEAR}-12"
        )


def find_cmip6_file(model: str, variable: str) -> Path:
    variable_root = CMIP6_ROOT / model / "historical" / variable
    if not variable_root.is_dir():
        raise FileNotFoundError(
            f"CMIP6 directory does not exist: {variable_root}"
        )

    matches = sorted(
        path
        for path in variable_root.rglob("*.nc")
        if path.is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one CMIP6 file for {model}/{variable}, "
            f"but found {len(matches)} under {variable_root}"
        )
    return matches[0]


def find_era5_file(model: str) -> Path:
    model_root = ERA5_GRID_ROOT / model
    preferred = model_root / f"ERA5_{model}_195901-201412.nc"
    if preferred.is_file():
        return preferred

    matches = sorted(
        path for path in model_root.glob("*.nc") if path.is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one ERA5 file for {model}, but found "
            f"{len(matches)} under {model_root}"
        )
    return matches[0]


def extract_variable(
    dataset: xr.Dataset,
    variable: str,
    label: str,
) -> xr.DataArray:
    if variable not in dataset.data_vars:
        raise ValueError(f"{label} is missing variable {variable}")

    data = dataset[variable]
    extra_dims = [
        dimension
        for dimension in data.dims
        if dimension not in {"time", "lat", "lon"}
    ]

    for dimension in extra_dims:
        if data.sizes[dimension] != 1:
            raise ValueError(
                f"{label} {variable} has unsupported dimension "
                f"{dimension!r} with size {data.sizes[dimension]}"
            )

    if extra_dims:
        data = data.squeeze(dim=extra_dims, drop=True)

    if set(data.dims) != {"time", "lat", "lon"}:
        raise ValueError(
            f"{label} {variable} dimensions are {data.dims}; "
            "expected time, lat, and lon"
        )

    validate_unit(variable, data.attrs.get("units"), label)
    return data.transpose("time", "lat", "lon")


def validate_grid_pair(
    gcm: xr.Dataset,
    era5: xr.Dataset,
    model: str,
) -> None:
    for coordinate in ("lat", "lon"):
        gcm_values = np.asarray(gcm[coordinate].values, dtype=np.float64)
        era5_values = np.asarray(era5[coordinate].values, dtype=np.float64)

        if gcm_values.shape != era5_values.shape:
            raise ValueError(
                f"{model} {coordinate} shape mismatch: "
                f"CMIP6={gcm_values.shape}, ERA5={era5_values.shape}"
            )

        if not np.allclose(
            gcm_values,
            era5_values,
            rtol=0.0,
            atol=1.0e-10,
            equal_nan=False,
        ):
            maximum_difference = float(
                np.max(np.abs(gcm_values - era5_values))
            )
            raise ValueError(
                f"{model} {coordinate} values do not match; "
                f"maximum difference={maximum_difference:.12g}"
            )


def nanquantile(values: np.ndarray, quantile: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanquantile(values, quantile, axis=0)


def nanmean(values: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def nanstd(values: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(values, axis=axis, ddof=1)


def finite_values(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float64).ravel()
    return flattened[np.isfinite(flattened)]


def add_spatial_summary(
    row: dict,
    prefix: str,
    values: np.ndarray,
) -> None:
    valid = finite_values(values)
    if valid.size == 0:
        row[f"{prefix}_min"] = np.nan
        row[f"{prefix}_p10"] = np.nan
        row[f"{prefix}_median"] = np.nan
        row[f"{prefix}_mean"] = np.nan
        row[f"{prefix}_p90"] = np.nan
        row[f"{prefix}_max"] = np.nan
        return

    row[f"{prefix}_min"] = float(np.min(valid))
    row[f"{prefix}_p10"] = float(np.quantile(valid, 0.10))
    row[f"{prefix}_median"] = float(np.median(valid))
    row[f"{prefix}_mean"] = float(np.mean(valid))
    row[f"{prefix}_p90"] = float(np.quantile(valid, 0.90))
    row[f"{prefix}_max"] = float(np.max(valid))


def safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    minimum_denominator: float,
) -> tuple[np.ndarray, float]:
    valid_numerator = np.isfinite(numerator)
    valid_denominator = np.isfinite(denominator)
    valid_pair = valid_numerator & valid_denominator
    safe = valid_pair & (np.abs(denominator) > minimum_denominator)

    result = np.full(np.shape(denominator), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=safe)

    pair_count = int(valid_pair.sum())
    invalid_percent = (
        100.0 * int((valid_pair & ~safe).sum()) / pair_count
        if pair_count else 100.0
    )
    return result, invalid_percent


def calculate_skewness(values: np.ndarray) -> np.ndarray:
    mean = nanmean(values, axis=0)
    standard_deviation = nanstd(values, axis=0)
    valid_scale = np.isfinite(standard_deviation) & (
        standard_deviation > 0.0
    )

    standardized = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(
        values - mean[None, :, :],
        standard_deviation[None, :, :],
        out=standardized,
        where=valid_scale[None, :, :],
    )
    return nanmean(standardized ** 3, axis=0)


def q99_leave_max_sensitivity(
    values: np.ndarray,
    denominator_floor: float,
) -> np.ndarray:
    time_size, lat_size, lon_size = values.shape
    flattened = values.reshape(time_size, lat_size * lon_size).copy()
    has_finite = np.any(np.isfinite(flattened), axis=0)

    filled = np.where(np.isfinite(flattened), flattened, -np.inf)
    maximum_indices = np.argmax(filled, axis=0)
    columns = np.flatnonzero(has_finite)
    flattened[maximum_indices[columns], columns] = np.nan

    full_q99 = nanquantile(values, 0.99)
    leave_max_q99 = nanquantile(
        flattened.reshape(time_size, lat_size, lon_size),
        0.99,
    )
    denominator = np.maximum(np.abs(full_q99), denominator_floor)
    sensitivity = 100.0 * np.abs(full_q99 - leave_max_q99) / denominator
    sensitivity[~np.isfinite(full_q99)] = np.nan
    return sensitivity


def percent_of_values(
    values: np.ndarray,
    condition: np.ndarray,
) -> float:
    finite = np.isfinite(values)
    count = int(finite.sum())
    if count == 0:
        return np.nan
    return 100.0 * int((finite & condition).sum()) / count


def diagnose_month(
    model: str,
    variable: str,
    month: int,
    gcm_values: np.ndarray,
    era5_values: np.ndarray,
) -> dict:
    if gcm_values.shape != era5_values.shape:
        raise ValueError(
            f"Array shape mismatch for {model}/{variable}/month {month}"
        )
    if gcm_values.shape[0] != EXPECTED_YEARS_PER_MONTH:
        raise ValueError(
            f"Expected {EXPECTED_YEARS_PER_MONTH} samples for "
            f"{model}/{variable}/month {month}, but found "
            f"{gcm_values.shape[0]}"
        )

    finite_pair = np.isfinite(gcm_values) & np.isfinite(era5_values)
    complete_cells = np.all(finite_pair, axis=0)
    grid_cells = int(complete_cells.size)
    complete_count = int(complete_cells.sum())

    gcm = np.where(complete_cells[None, :, :], gcm_values, np.nan)
    era5 = np.where(complete_cells[None, :, :], era5_values, np.nan)

    threshold = NEAR_ZERO_THRESHOLDS[variable]
    row = {
        "model": model,
        "variable": variable,
        "month": month,
        "years_per_month": gcm_values.shape[0],
        "lat_cells": gcm_values.shape[1],
        "lon_cells": gcm_values.shape[2],
        "grid_cells": grid_cells,
        "complete_grid_cells": complete_count,
        "complete_grid_cell_percent": (
            100.0 * complete_count / grid_cells if grid_cells else 0.0
        ),
        "paired_finite_value_percent": (
            100.0 * int(finite_pair.sum()) / int(finite_pair.size)
        ),
        "near_zero_threshold": threshold,
        "gcm_near_zero_value_percent": percent_of_values(
            gcm, np.abs(gcm) <= threshold
        ),
        "era5_near_zero_value_percent": percent_of_values(
            era5, np.abs(era5) <= threshold
        ),
        "gcm_negative_value_percent": percent_of_values(gcm, gcm < 0.0),
        "era5_negative_value_percent": percent_of_values(
            era5, era5 < 0.0
        ),
        "gcm_all_value_mean": float(nanmean(gcm)),
        "era5_all_value_mean": float(nanmean(era5)),
    }

    gcm_mean = nanmean(gcm, axis=0)
    era5_mean = nanmean(era5, axis=0)
    mean_bias = gcm_mean - era5_mean
    mean_ratio, mean_ratio_invalid = safe_ratio(
        era5_mean, gcm_mean, threshold
    )

    gcm_std = nanstd(gcm, axis=0)
    era5_std = nanstd(era5, axis=0)
    std_ratio, std_ratio_invalid = safe_ratio(
        era5_std, gcm_std, 1.0e-12
    )

    row["mean_ratio_invalid_percent"] = mean_ratio_invalid
    row["std_ratio_invalid_percent"] = std_ratio_invalid
    add_spatial_summary(row, "gcm_mean", gcm_mean)
    add_spatial_summary(row, "era5_mean", era5_mean)
    add_spatial_summary(row, "mean_bias_gcm_minus_era5", mean_bias)
    add_spatial_summary(row, "mean_multiplicative_factor", mean_ratio)
    add_spatial_summary(row, "gcm_std", gcm_std)
    add_spatial_summary(row, "era5_std", era5_std)
    add_spatial_summary(row, "std_multiplicative_factor", std_ratio)
    add_spatial_summary(row, "gcm_skewness", calculate_skewness(gcm))
    add_spatial_summary(row, "era5_skewness", calculate_skewness(era5))

    for quantile in QUANTILES:
        label = f"q{int(round(quantile * 100)):02d}"
        gcm_quantile = nanquantile(gcm, quantile)
        era5_quantile = nanquantile(era5, quantile)
        additive_adjustment = era5_quantile - gcm_quantile
        multiplicative_factor, invalid_percent = safe_ratio(
            era5_quantile,
            gcm_quantile,
            threshold,
        )

        row[f"{label}_multiplicative_invalid_percent"] = invalid_percent
        add_spatial_summary(row, f"gcm_{label}", gcm_quantile)
        add_spatial_summary(row, f"era5_{label}", era5_quantile)
        add_spatial_summary(
            row,
            f"additive_adjustment_{label}",
            additive_adjustment,
        )
        add_spatial_summary(
            row,
            f"multiplicative_factor_{label}",
            multiplicative_factor,
        )

    gcm_tail = q99_leave_max_sensitivity(gcm, threshold)
    era5_tail = q99_leave_max_sensitivity(era5, threshold)
    add_spatial_summary(row, "gcm_q99_leave_max_change_percent", gcm_tail)
    add_spatial_summary(
        row,
        "era5_q99_leave_max_change_percent",
        era5_tail,
    )
    row["q99_sensitivity_worst_p90_percent"] = max(
        row["gcm_q99_leave_max_change_percent_p90"],
        row["era5_q99_leave_max_change_percent_p90"],
    )
    return row


def diagnose_model(model: str) -> list[dict]:
    era5_file = find_era5_file(model)
    rows = []

    with xr.open_dataset(era5_file, decode_times=True) as era5_source:
        era5 = standardize_dataset(era5_source)
        validate_time(era5, f"ERA5/{model}")

        for variable in VARIABLES:
            cmip6_file = find_cmip6_file(model, variable)

            with xr.open_dataset(cmip6_file, decode_times=True) as gcm_source:
                gcm = standardize_dataset(gcm_source)
                validate_time(gcm, f"CMIP6/{model}/{variable}")
                validate_grid_pair(gcm, era5, model)

                gcm_data = extract_variable(
                    gcm, variable, f"CMIP6/{model}"
                )
                era5_data = extract_variable(
                    era5, variable, f"ERA5/{model}"
                )

                gcm_months = np.asarray(
                    gcm["time"].dt.month.values,
                    dtype=np.int64,
                )
                era5_months = np.asarray(
                    era5["time"].dt.month.values,
                    dtype=np.int64,
                )

                if not np.array_equal(gcm_months, era5_months):
                    raise ValueError(
                        f"Month order mismatch for {model}/{variable}"
                    )

                for month in MONTHS:
                    indices = np.flatnonzero(gcm_months == month)
                    gcm_values = np.asarray(
                        gcm_data.isel(time=indices).values,
                        dtype=np.float64,
                    )
                    era5_values = np.asarray(
                        era5_data.isel(time=indices).values,
                        dtype=np.float64,
                    )
                    rows.append(
                        diagnose_month(
                            model,
                            variable,
                            month,
                            gcm_values,
                            era5_values,
                        )
                    )

    return rows


def finite_max(values: list[float]) -> float:
    array = finite_values(np.asarray(values, dtype=np.float64))
    return float(np.max(array)) if array.size else np.nan


def finite_median(values: list[float]) -> float:
    array = finite_values(np.asarray(values, dtype=np.float64))
    return float(np.median(array)) if array.size else np.nan


def build_recommendation_rows(monthly_rows: list[dict]) -> list[dict]:
    recommendation_rows = []

    for model in MODELS:
        for variable in VARIABLES:
            selected = [
                row
                for row in monthly_rows
                if row["model"] == model and row["variable"] == variable
            ]
            if len(selected) != 12:
                raise ValueError(
                    f"Expected 12 monthly rows for {model}/{variable}, "
                    f"but found {len(selected)}"
                )

            worst_tail = finite_max(
                [
                    row["q99_sensitivity_worst_p90_percent"]
                    for row in selected
                ]
            )
            median_tail = finite_median(
                [
                    row["q99_sensitivity_worst_p90_percent"]
                    for row in selected
                ]
            )
            maximum_near_zero = finite_max(
                [
                    max(
                        row["gcm_near_zero_value_percent"],
                        row["era5_near_zero_value_percent"],
                    )
                    for row in selected
                ]
            )
            maximum_q01_invalid = finite_max(
                [
                    row["q01_multiplicative_invalid_percent"]
                    for row in selected
                ]
            )
            median_absolute_bias = finite_median(
                [
                    abs(row["mean_bias_gcm_minus_era5_median"])
                    for row in selected
                ]
            )

            if worst_tail > 10.0:
                suggested_bounds = "0.05-0.95"
                tail_flag = "HIGH"
            else:
                suggested_bounds = "0.02-0.98"
                tail_flag = "MODERATE_OR_LOW"

            if variable == "tas":
                lower_tail_action = (
                    "Multiplicative factors are numerically stable in "
                    "kelvin; compare multiplicative and additive diagnostics"
                )
            elif maximum_near_zero > 1.0 or maximum_q01_invalid > 1.0:
                lower_tail_action = (
                    "Use a positive floor and bounded or hybrid lower-tail "
                    "treatment before multiplicative QM"
                )
            else:
                lower_tail_action = (
                    "Bounded multiplicative QM is numerically feasible; "
                    "retain a physical lower bound of zero"
                )

            recommendation_rows.append(
                {
                    "model": model,
                    "variable": variable,
                    "monthly_sample_size": EXPECTED_YEARS_PER_MONTH,
                    "median_absolute_mean_bias": median_absolute_bias,
                    "median_q99_sensitivity_p90_percent": median_tail,
                    "worst_month_q99_sensitivity_p90_percent": worst_tail,
                    "maximum_near_zero_value_percent": maximum_near_zero,
                    "maximum_q01_multiplicative_invalid_percent": (
                        maximum_q01_invalid
                    ),
                    "tail_sensitivity_flag": tail_flag,
                    "diagnostic_suggested_quantile_bounds": suggested_bounds,
                    "lower_tail_action": lower_tail_action,
                    "final_method_status": (
                        "Diagnostic suggestion only; confirm after reviewing "
                        "monthly CSV and heatmaps"
                    ),
                }
            )

    return recommendation_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows are available for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_heatmaps(monthly_rows: list[dict], output_root: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib is unavailable; heatmaps were not created.")
        return []

    metrics = (
        (
            "mean_bias_gcm_minus_era5_median",
            "Median GCM minus ERA5 mean bias",
            "mean_bias",
            "coolwarm",
        ),
        (
            "q99_sensitivity_worst_p90_percent",
            "P90 leave-maximum-out Q99 sensitivity (%)",
            "q99_sensitivity",
            "magma",
        ),
        (
            "gcm_near_zero_value_percent",
            "GCM near-zero value percentage",
            "near_zero",
            "viridis",
        ),
    )

    created = []
    for variable in VARIABLES:
        variable_rows = [
            row for row in monthly_rows if row["variable"] == variable
        ]
        row_lookup = {
            (row["model"], row["month"]): row
            for row in variable_rows
        }

        for metric, title, filename_label, color_map in metrics:
            matrix = np.full((len(MODELS), 12), np.nan, dtype=np.float64)
            for model_index, model in enumerate(MODELS):
                for month_index, month in enumerate(MONTHS):
                    matrix[model_index, month_index] = row_lookup[
                        (model, month)
                    ][metric]

            figure_height = max(7.0, 0.42 * len(MODELS) + 2.0)
            figure, axis = plt.subplots(
                figsize=(12.5, figure_height),
                constrained_layout=True,
            )

            if color_map == "coolwarm":
                finite = finite_values(matrix)
                limit = float(np.max(np.abs(finite))) if finite.size else 1.0
                if limit == 0.0:
                    limit = 1.0
                image = axis.imshow(
                    matrix,
                    aspect="auto",
                    cmap=color_map,
                    vmin=-limit,
                    vmax=limit,
                )
            else:
                image = axis.imshow(
                    matrix,
                    aspect="auto",
                    cmap=color_map,
                )

            axis.set_xticks(np.arange(12), labels=np.arange(1, 13))
            axis.set_yticks(np.arange(len(MODELS)), labels=MODELS)
            axis.set_xlabel("Calendar month")
            axis.set_ylabel("CMIP6 model")
            axis.set_title(f"{variable}: {title}")
            colorbar = figure.colorbar(image, ax=axis, shrink=0.9)
            colorbar.set_label(title)

            output_path = output_root / (
                f"heatmap_{variable}_{filename_label}.png"
            )
            figure.savefig(output_path, dpi=180, bbox_inches="tight")
            plt.close(figure)
            created.append(output_path)

    return created


def write_run_summary(
    path: Path,
    monthly_rows: list[dict],
    recommendation_rows: list[dict],
    failed_models: list[tuple[str, str]],
    figures: list[Path],
) -> None:
    expected_rows = len(MODELS) * len(VARIABLES) * len(MONTHS)
    high_tail = [
        row
        for row in recommendation_rows
        if row["tail_sensitivity_flag"] == "HIGH"
    ]
    near_zero_risk = [
        row
        for row in recommendation_rows
        if row["variable"] != "tas"
        and (
            row["maximum_near_zero_value_percent"] > 1.0
            or row["maximum_q01_multiplicative_invalid_percent"] > 1.0
        )
    ]

    lines = [
        "QM DISTRIBUTION DIAGNOSTICS SUMMARY",
        "=" * 80,
        f"Models expected: {len(MODELS)}",
        f"Models failed: {len(failed_models)}",
        f"Monthly rows expected: {expected_rows}",
        f"Monthly rows created: {len(monthly_rows)}",
        f"Model-variable recommendation rows: {len(recommendation_rows)}",
        f"Figures created: {len(figures)}",
        f"High-tail model-variable cases: {len(high_tail)}",
        f"Near-zero multiplicative-risk cases: {len(near_zero_risk)}",
        "",
        "INTERPRETATION NOTES",
        "- Each calendar month uses 56 annual samples at each grid cell.",
        "- Spatial grid cells are summarized after grid-cell diagnostics.",
        "- Grid cells are not treated as independent temporal samples.",
        "- Q99 sensitivity removes the single largest annual value at each grid cell.",
        "- Suggested quantile bounds are diagnostics, not the final QM decision.",
        "- Future scenario data are not read or modified by this program.",
    ]

    if failed_models:
        lines.extend(["", "FAILED MODELS"])
        for model, error in failed_models:
            lines.append(f"- {model}: {error}")

    if high_tail:
        lines.extend(["", "HIGH-TAIL CASES"])
        for row in high_tail:
            lines.append(
                f"- {row['model']} {row['variable']}: "
                f"worst P90 sensitivity="
                f"{row['worst_month_q99_sensitivity_p90_percent']:.6g}%"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose monthly CMIP6 and ERA5 distributions before "
            "quantile mapping for 17 models."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for CSV reports, summary text, and heatmaps.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Create CSV and text reports without PNG heatmaps.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output_root = arguments.output_root.resolve()

    print_separator()
    print("PRE-QM MONTHLY DISTRIBUTION DIAGNOSTICS FOR 17 MODELS")
    print_separator()
    print(f"CMIP6 root : {CMIP6_ROOT}")
    print(f"ERA5 root  : {ERA5_GRID_ROOT}")
    print(f"Output root: {output_root}")
    print(f"Models     : {len(MODELS)}")
    print(f"Variables  : {', '.join(VARIABLES)}")
    print(f"Period     : {START_YEAR}-01 to {END_YEAR}-12")
    print(f"Samples    : {EXPECTED_YEARS_PER_MONTH} per calendar month")
    print("Input mode : read only")
    print()

    if not CMIP6_ROOT.is_dir():
        print(f"ERROR: CMIP6 root does not exist: {CMIP6_ROOT}")
        return 1
    if not ERA5_GRID_ROOT.is_dir():
        print(f"ERROR: ERA5 grid root does not exist: {ERA5_GRID_ROOT}")
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    monthly_rows = []
    failed_models = []

    for index, model in enumerate(MODELS, start=1):
        print_separator("-")
        print(f"[{index}/{len(MODELS)}] {model}")
        print_separator("-")
        try:
            model_rows = diagnose_model(model)
            monthly_rows.extend(model_rows)
            print(f"PASS: {len(model_rows)} monthly diagnostic rows created")
        except Exception as error:
            failed_models.append((model, str(error)))
            print(f"FAILED: {error}")

    if monthly_rows:
        monthly_path = output_root / MONTHLY_REPORT_NAME
        write_csv(monthly_path, monthly_rows)
        recommendation_path = output_root / MODEL_VARIABLE_REPORT_NAME

        if len(monthly_rows) == len(MODELS) * len(VARIABLES) * len(MONTHS):
            recommendation_rows = build_recommendation_rows(monthly_rows)
            write_csv(recommendation_path, recommendation_rows)
            figures = (
                []
                if arguments.skip_figures
                else create_heatmaps(monthly_rows, output_root)
            )
        else:
            recommendation_rows = []
            figures = []
    else:
        monthly_path = output_root / MONTHLY_REPORT_NAME
        recommendation_path = output_root / MODEL_VARIABLE_REPORT_NAME
        recommendation_rows = []
        figures = []

    summary_path = output_root / RUN_SUMMARY_NAME
    write_run_summary(
        summary_path,
        monthly_rows,
        recommendation_rows,
        failed_models,
        figures,
    )

    expected_rows = len(MODELS) * len(VARIABLES) * len(MONTHS)
    print()
    print_separator()
    print("DIAGNOSTIC SUMMARY")
    print_separator()
    print(f"Expected models : {len(MODELS)}")
    print(f"Failed models   : {len(failed_models)}")
    print(f"Expected rows   : {expected_rows}")
    print(f"Created rows    : {len(monthly_rows)}")
    print(f"Figures created : {len(figures)}")
    print(f"Monthly report  : {monthly_path}")
    print(f"Method report   : {recommendation_path}")
    print(f"Run summary     : {summary_path}")

    if failed_models or len(monthly_rows) != expected_rows:
        print()
        print("PRE-QM DISTRIBUTION DIAGNOSTICS FAILED")
        return 1

    print()
    print("ALL 17 MODELS PASSED THE PRE-QM DISTRIBUTION DIAGNOSTICS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
