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
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError:  # Allows the numerical --self-test in a minimal environment.
    xr = None


# PROJECT_ROOT is provided by scripts/common/project_paths.py
SCRIPT_ROOT = Path(__file__).resolve().parent
CMIP6_ROOT = get_path("data_interim_cmip6_china_clipped")
ERA5_ROOT = get_path("data_interim_era5_on_gcm_grid")
OUTPUT_ROOT = get_path("data_processed_validation") / "qm_qdm_holdout_smoke_test"

DEFAULT_MODEL = "CAS-ESM2-0"
DEFAULT_VARIABLE = "sfcWind"
FULL_START_YEAR = 1959
FULL_END_YEAR = 2014
FIT_START_YEAR = 1959
FIT_END_YEAR = 1988
EVAL_START_YEAR = 1989
EVAL_END_YEAR = 2014
MONTHS = tuple(range(1, 13))

EXPECTED_UNITS = {
    "tas": {"k", "kelvin"},
    "rsds": {"wm-2", "wm**-2", "wm^-2", "w/m2"},
    "sfcWind": {"ms-1", "ms**-1", "ms^-1", "m/s"},
}

PHYSICAL_MINIMUM = {
    "tas": None,
    "rsds": 0.0,
    "sfcWind": 0.0,
}

RATIO_DENOMINATOR_FLOOR = {
    "tas": 1.0,
    "rsds": 1.0,
    "sfcWind": 0.1,
}


@dataclass(frozen=True)
class CorrectionResult:
    qm: np.ndarray
    qdm: np.ndarray
    qm_tail_percent: float
    qdm_tail_percent: float
    qm_fallback_percent: float
    qdm_fallback_percent: float


def normalize_unit(unit: str | None) -> str:
    if unit is None:
        return ""
    value = str(unit).strip().lower()
    replacements = {
        "watts": "w",
        "watt": "w",
        "metres": "m",
        "metre": "m",
        "meters": "m",
        "meter": "m",
        "seconds": "s",
        "second": "s",
        "degrees_kelvin": "kelvin",
        "degree_kelvin": "kelvin",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return "".join(value.split())


def validate_unit(variable: str, unit: str | None, label: str) -> None:
    if normalize_unit(unit) not in EXPECTED_UNITS[variable]:
        raise ValueError(
            f"Unexpected units for {label}/{variable}: {unit!r}; "
            f"expected one of {sorted(EXPECTED_UNITS[variable])}"
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
        standard_name = str(dataset[name].attrs.get("standard_name", "")).lower()
        axis = str(dataset[name].attrs.get("axis", "")).upper()
        if coordinate_type == "longitude" and (
            standard_name == "longitude" or axis == "X"
        ):
            return name
        if coordinate_type == "latitude" and (
            standard_name == "latitude" or axis == "Y"
        ):
            return name
        if coordinate_type == "time" and (
            standard_name == "time" or axis == "T"
        ):
            return name
    raise ValueError(f"Could not identify the {coordinate_type} coordinate")


def standardize_dataset(dataset: xr.Dataset) -> xr.Dataset:
    names = {
        find_coordinate_name(dataset, ("lon", "longitude", "x"), "longitude"): "lon",
        find_coordinate_name(dataset, ("lat", "latitude", "y"), "latitude"): "lat",
        find_coordinate_name(dataset, ("time",), "time"): "time",
    }
    rename = {old: new for old, new in names.items() if old != new}
    if rename:
        dataset = dataset.rename(rename)
    for coordinate in ("time", "lat", "lon"):
        if dataset[coordinate].ndim != 1:
            raise ValueError(f"Coordinate {coordinate} must be one-dimensional")
    lon = np.asarray(dataset["lon"].values, dtype=np.float64)
    if np.nanmax(lon) > 180.0:
        attrs = dict(dataset["lon"].attrs)
        lon = ((lon + 180.0) % 360.0) - 180.0
        dataset = dataset.assign_coords(lon=("lon", lon))
        dataset["lon"].attrs.update(attrs)
    return dataset.sortby("lat").sortby("lon")


def expected_month_keys(start_year: int, end_year: int) -> list[int]:
    return list(range(start_year * 12, end_year * 12 + 12))


def month_keys(time: xr.DataArray) -> list[int]:
    years = np.asarray(time.dt.year.values, dtype=np.int64)
    months = np.asarray(time.dt.month.values, dtype=np.int64)
    return (years * 12 + months - 1).tolist()


def validate_full_time(dataset: xr.Dataset, label: str) -> None:
    expected = expected_month_keys(FULL_START_YEAR, FULL_END_YEAR)
    actual = month_keys(dataset["time"])
    if actual != expected:
        raise ValueError(
            f"{label} must contain the complete monthly sequence "
            f"{FULL_START_YEAR}-01 to {FULL_END_YEAR}-12; "
            f"found {len(actual)} time steps"
        )


def find_cmip6_file(root: Path, model: str, variable: str) -> Path:
    variable_root = root / model / "historical" / variable
    matches = sorted(path for path in variable_root.rglob("*.nc") if path.is_file())
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one CMIP6 file for {model}/{variable}, found "
            f"{len(matches)} under {variable_root}"
        )
    return matches[0]


def find_era5_file(root: Path, model: str) -> Path:
    model_root = root / model
    preferred = model_root / f"ERA5_{model}_195901-201412.nc"
    if preferred.is_file():
        return preferred
    matches = sorted(path for path in model_root.glob("*.nc") if path.is_file())
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one ERA5 file for {model}, found {len(matches)} "
            f"under {model_root}"
        )
    return matches[0]


def extract_variable(dataset: xr.Dataset, variable: str, label: str) -> xr.DataArray:
    if variable not in dataset.data_vars:
        raise ValueError(f"{label} is missing variable {variable}")
    data = dataset[variable]
    extra_dims = [d for d in data.dims if d not in {"time", "lat", "lon"}]
    for dim in extra_dims:
        if data.sizes[dim] != 1:
            raise ValueError(
                f"{label}/{variable} has unsupported dimension {dim!r} "
                f"with size {data.sizes[dim]}"
            )
    if extra_dims:
        data = data.squeeze(dim=extra_dims, drop=True)
    if set(data.dims) != {"time", "lat", "lon"}:
        raise ValueError(
            f"{label}/{variable} dimensions are {data.dims}; expected time, lat, lon"
        )
    validate_unit(variable, data.attrs.get("units"), label)
    return data.transpose("time", "lat", "lon")


def validate_grid_pair(gcm: xr.Dataset, era5: xr.Dataset, model: str) -> None:
    for coordinate in ("lat", "lon"):
        left = np.asarray(gcm[coordinate].values, dtype=np.float64)
        right = np.asarray(era5[coordinate].values, dtype=np.float64)
        if left.shape != right.shape or not np.allclose(
            left, right, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError(f"CMIP6 and ERA5 {coordinate} grids differ for {model}")


def nanquantile(values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanquantile(values, probabilities, axis=0)


def empirical_probabilities_against_sample(
    values: np.ndarray,
    sample: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Return mid-rank empirical CDF probabilities for each grid-cell series."""
    n_values, n_cells = values.shape
    probabilities = np.full((n_values, n_cells), np.nan, dtype=np.float64)
    for start in range(0, n_cells, chunk_size):
        stop = min(start + chunk_size, n_cells)
        target = values[:, start:stop]
        reference = sample[:, start:stop]
        finite_target = np.isfinite(target)
        finite_reference = np.isfinite(reference)
        less = (
            (reference[None, :, :] < target[:, None, :])
            & finite_reference[None, :, :]
        ).sum(axis=1)
        equal = (
            (reference[None, :, :] == target[:, None, :])
            & finite_reference[None, :, :]
        ).sum(axis=1)
        count = finite_reference.sum(axis=0)[None, :]
        valid = finite_target & (count > 0)
        block = np.full(target.shape, np.nan, dtype=np.float64)
        np.divide(less + 0.5 * equal, count, out=block, where=valid)
        probabilities[:, start:stop] = block
    return probabilities


def interpolate_uniform_quantiles(
    quantiles: np.ndarray,
    probabilities: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    n_quantiles, n_cells = quantiles.shape
    clipped = np.clip(probabilities, lower, upper)
    position = (clipped - lower) / (upper - lower) * (n_quantiles - 1)
    lower_index = np.floor(position).astype(np.int64)
    upper_index = np.minimum(lower_index + 1, n_quantiles - 1)
    weight = position - lower_index
    columns = np.broadcast_to(np.arange(n_cells), probabilities.shape)
    low_values = quantiles[lower_index, columns]
    high_values = quantiles[upper_index, columns]
    return low_values + weight * (high_values - low_values)


def correct_month(
    model_fit: np.ndarray,
    reference_fit: np.ndarray,
    model_eval: np.ndarray,
    variable: str,
    lower: float,
    upper: float,
    n_quantiles: int,
    correction_mode: str | None = None,
) -> CorrectionResult:
    if correction_mode is None or correction_mode == "auto":
        correction_mode = "additive" if variable == "tas" else "ratio"
    if correction_mode not in {"additive", "ratio"}:
        raise ValueError(
            f"Unsupported correction mode {correction_mode!r}; "
            "expected 'additive' or 'ratio'"
        )
    original_shape = model_eval.shape
    fit = model_fit.reshape(model_fit.shape[0], -1).astype(np.float64)
    obs = reference_fit.reshape(reference_fit.shape[0], -1).astype(np.float64)
    target = model_eval.reshape(model_eval.shape[0], -1).astype(np.float64)

    complete = (
        np.all(np.isfinite(fit), axis=0)
        & np.all(np.isfinite(obs), axis=0)
        & np.all(np.isfinite(target), axis=0)
    )
    fit[:, ~complete] = np.nan
    obs[:, ~complete] = np.nan
    target[:, ~complete] = np.nan

    quantile_probabilities = np.linspace(lower, upper, n_quantiles)
    model_quantiles = nanquantile(fit, quantile_probabilities)
    reference_quantiles = nanquantile(obs, quantile_probabilities)

    qm_probabilities = empirical_probabilities_against_sample(target, fit)
    qdm_probabilities = empirical_probabilities_against_sample(target, target)

    qm_tail = np.isfinite(qm_probabilities) & (
        (qm_probabilities < lower) | (qm_probabilities > upper)
    )
    qdm_tail = np.isfinite(qdm_probabilities) & (
        (qdm_probabilities < lower) | (qdm_probabilities > upper)
    )

    qm_model_q = interpolate_uniform_quantiles(
        model_quantiles, qm_probabilities, lower, upper
    )
    qm_reference_q = interpolate_uniform_quantiles(
        reference_quantiles, qm_probabilities, lower, upper
    )
    qdm_model_q = interpolate_uniform_quantiles(
        model_quantiles, qdm_probabilities, lower, upper
    )
    qdm_reference_q = interpolate_uniform_quantiles(
        reference_quantiles, qdm_probabilities, lower, upper
    )

    if correction_mode == "additive":
        qm = target + (qm_reference_q - qm_model_q)
        qdm = target + (qdm_reference_q - qdm_model_q)
        qm_fallback = np.zeros(target.shape, dtype=bool)
        qdm_fallback = np.zeros(target.shape, dtype=bool)
    else:
        floor = RATIO_DENOMINATOR_FLOOR[variable]
        qm_safe = np.isfinite(qm_model_q) & (np.abs(qm_model_q) > floor)
        qdm_safe = np.isfinite(qdm_model_q) & (np.abs(qdm_model_q) > floor)
        qm = np.full(target.shape, np.nan, dtype=np.float64)
        qdm = np.full(target.shape, np.nan, dtype=np.float64)
        np.multiply(
            target,
            qm_reference_q / np.where(qm_safe, qm_model_q, 1.0),
            out=qm,
            where=qm_safe,
        )
        np.multiply(
            target,
            qdm_reference_q / np.where(qdm_safe, qdm_model_q, 1.0),
            out=qdm,
            where=qdm_safe,
        )
        qm_fallback = np.isfinite(target) & ~qm_safe
        qdm_fallback = np.isfinite(target) & ~qdm_safe
        # A rare unsafe ratio uses the corresponding additive quantile delta.
        qm[qm_fallback] = (
            target + qm_reference_q - qm_model_q
        )[qm_fallback]
        qdm[qdm_fallback] = (
            target + qdm_reference_q - qdm_model_q
        )[qdm_fallback]

    minimum = PHYSICAL_MINIMUM[variable]
    if minimum is not None:
        qm = np.maximum(qm, minimum)
        qdm = np.maximum(qdm, minimum)

    finite_count = int(np.isfinite(target).sum())

    def percentage(mask: np.ndarray) -> float:
        return 100.0 * int(mask.sum()) / finite_count if finite_count else np.nan

    return CorrectionResult(
        qm=qm.reshape(original_shape),
        qdm=qdm.reshape(original_shape),
        qm_tail_percent=percentage(qm_tail),
        qdm_tail_percent=percentage(qdm_tail),
        qm_fallback_percent=percentage(qm_fallback),
        qdm_fallback_percent=percentage(qdm_fallback),
    )


def finite_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {f"{prefix}_{name}": np.nan for name in ("median", "mean", "p90", "max")}
    return {
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_p90": float(np.quantile(finite, 0.90)),
        f"{prefix}_max": float(np.max(finite)),
    }


def finite_spatial_summary(
    values: np.ndarray,
    prefix: str,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> dict[str, float]:
    summary = finite_summary(values, prefix)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{prefix} must be a two-dimensional spatial field")
    if array.shape != (latitudes.size, longitudes.size):
        raise ValueError(
            f"{prefix} shape {array.shape} does not match "
            f"lat/lon sizes {(latitudes.size, longitudes.size)}"
        )
    finite = np.isfinite(array)
    if not finite.any():
        summary[f"{prefix}_max_lat"] = np.nan
        summary[f"{prefix}_max_lon"] = np.nan
        return summary
    flat_index = int(np.nanargmax(array))
    lat_index, lon_index = np.unravel_index(flat_index, array.shape)
    summary[f"{prefix}_max_lat"] = float(latitudes[lat_index])
    summary[f"{prefix}_max_lon"] = float(longitudes[lon_index])
    return summary


def evaluation_rows(
    model: str,
    variable: str,
    bounds_label: str,
    month: int,
    reference: np.ndarray,
    raw: np.ndarray,
    qm: np.ndarray,
    qdm: np.ndarray,
    diagnostics: CorrectionResult,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    correction_mode: str,
) -> list[dict]:
    rows: list[dict] = []
    for method, values in (("RAW", raw), ("QM", qm), ("QDM", qdm)):
        row: dict[str, object] = {
            "model": model,
            "variable": variable,
            "bounds": bounds_label,
            "month": month,
            "method": method,
            "correction_mode": correction_mode,
            "fit_period": f"{FIT_START_YEAR}-{FIT_END_YEAR}",
            "evaluation_period": f"{EVAL_START_YEAR}-{EVAL_END_YEAR}",
            "fit_samples_per_month": FIT_END_YEAR - FIT_START_YEAR + 1,
            "evaluation_samples_per_month": EVAL_END_YEAR - EVAL_START_YEAR + 1,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_error = np.abs(np.nanmean(values, axis=0) - np.nanmean(reference, axis=0))
            std_error = np.abs(
                np.nanstd(values, axis=0, ddof=1)
                - np.nanstd(reference, axis=0, ddof=1)
            )
            for probability, label in ((0.10, "q10"), (0.50, "q50"), (0.90, "q90")):
                error = np.abs(
                    np.nanquantile(values, probability, axis=0)
                    - np.nanquantile(reference, probability, axis=0)
                )
                row.update(
                    finite_spatial_summary(
                        error,
                        f"absolute_{label}_error",
                        latitudes,
                        longitudes,
                    )
                )
        row.update(
            finite_spatial_summary(
                mean_error, "absolute_mean_error", latitudes, longitudes
            )
        )
        row.update(
            finite_spatial_summary(
                std_error, "absolute_std_error", latitudes, longitudes
            )
        )
        row["negative_value_percent"] = (
            100.0 * int((np.isfinite(values) & (values < 0.0)).sum())
            / int(np.isfinite(values).sum())
            if np.isfinite(values).any()
            else np.nan
        )
        row["tail_rule_trigger_percent"] = (
            0.0
            if method == "RAW"
            else diagnostics.qm_tail_percent
            if method == "QM"
            else diagnostics.qdm_tail_percent
        )
        row["ratio_fallback_percent"] = (
            0.0
            if method == "RAW"
            else diagnostics.qm_fallback_percent
            if method == "QM"
            else diagnostics.qdm_fallback_percent
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for bounds in sorted({str(row["bounds"]) for row in rows}):
        for method in ("RAW", "QM", "QDM"):
            selected = [
                row for row in rows
                if row["bounds"] == bounds and row["method"] == method
            ]
            summary.append(
                {
                    "bounds": bounds,
                    "method": method,
                    "months": len(selected),
                    "median_monthly_spatial_median_absolute_mean_error": float(
                        np.median([row["absolute_mean_error_median"] for row in selected])
                    ),
                    "median_monthly_spatial_median_absolute_std_error": float(
                        np.median([row["absolute_std_error_median"] for row in selected])
                    ),
                    "median_monthly_spatial_median_absolute_q10_error": float(
                        np.median([row["absolute_q10_error_median"] for row in selected])
                    ),
                    "median_monthly_spatial_median_absolute_q50_error": float(
                        np.median([row["absolute_q50_error_median"] for row in selected])
                    ),
                    "median_monthly_spatial_median_absolute_q90_error": float(
                        np.median([row["absolute_q90_error_median"] for row in selected])
                    ),
                    "maximum_monthly_tail_rule_trigger_percent": float(
                        np.max([row["tail_rule_trigger_percent"] for row in selected])
                    ),
                    "maximum_monthly_ratio_fallback_percent": float(
                        np.max([row["ratio_fallback_percent"] for row in selected])
                    ),
                }
            )
    return summary


def run_validation(arguments: argparse.Namespace) -> int:
    if xr is None:
        raise RuntimeError(
            "xarray is required for NetCDF input/output. Activate the pvwind "
            "environment before running the real-data smoke test."
        )
    cmip6_file = find_cmip6_file(arguments.cmip6_root, arguments.model, arguments.variable)
    era5_file = find_era5_file(arguments.era5_root, arguments.model)
    requested_mode = getattr(arguments, "correction_mode", "auto")
    correction_mode = (
        "additive"
        if requested_mode == "auto" and arguments.variable == "tas"
        else "ratio"
        if requested_mode == "auto"
        else requested_mode
    )
    if correction_mode not in {"additive", "ratio"}:
        raise ValueError(f"Unsupported correction mode: {correction_mode}")
    variable_directory = (
        f"{arguments.variable}_{correction_mode}"
        if arguments.variable == "tas"
        else arguments.variable
    )
    output_root = arguments.output_root / arguments.model / variable_directory
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"CMIP6 file : {cmip6_file}")
    print(f"ERA5 file  : {era5_file}")
    print(f"Output root: {output_root}")
    print(f"Test        : {FIT_START_YEAR}-{FIT_END_YEAR} fit -> "
          f"{EVAL_START_YEAR}-{EVAL_END_YEAR} evaluation")
    print(f"Correction  : {correction_mode}")

    with xr.open_dataset(cmip6_file, decode_times=True) as gcm_source, xr.open_dataset(
        era5_file, decode_times=True
    ) as era5_source:
        gcm = standardize_dataset(gcm_source)
        era5 = standardize_dataset(era5_source)
        validate_full_time(gcm, f"CMIP6/{arguments.model}")
        validate_full_time(era5, f"ERA5/{arguments.model}")
        validate_grid_pair(gcm, era5, arguments.model)
        gcm_data = extract_variable(gcm, arguments.variable, f"CMIP6/{arguments.model}").load()
        era5_data = extract_variable(era5, arguments.variable, f"ERA5/{arguments.model}").load()

    # Select each dataset by its own year values.  CMIP6 may use a no-leap
    # cftime calendar while ERA5 uses pandas/numpy datetimes, so a boolean
    # index carrying one dataset's time coordinate must not index the other.
    gcm_years = np.asarray(gcm_data.time.dt.year.values, dtype=np.int64)
    era5_years = np.asarray(era5_data.time.dt.year.values, dtype=np.int64)
    gcm_fit_indices = np.flatnonzero(
        (gcm_years >= FIT_START_YEAR) & (gcm_years <= FIT_END_YEAR)
    )
    era5_fit_indices = np.flatnonzero(
        (era5_years >= FIT_START_YEAR) & (era5_years <= FIT_END_YEAR)
    )
    gcm_eval_indices = np.flatnonzero(
        (gcm_years >= EVAL_START_YEAR) & (gcm_years <= EVAL_END_YEAR)
    )
    era5_eval_indices = np.flatnonzero(
        (era5_years >= EVAL_START_YEAR) & (era5_years <= EVAL_END_YEAR)
    )
    gcm_fit = gcm_data.isel(time=gcm_fit_indices)
    era5_fit = era5_data.isel(time=era5_fit_indices)
    gcm_eval = gcm_data.isel(time=gcm_eval_indices)
    era5_eval = era5_data.isel(time=era5_eval_indices)

    if gcm_fit.sizes["time"] != 360 or gcm_eval.sizes["time"] != 312:
        raise ValueError("Unexpected fit/evaluation time length after splitting")
    if era5_fit.sizes["time"] != 360 or era5_eval.sizes["time"] != 312:
        raise ValueError("Unexpected ERA5 fit/evaluation time length after splitting")
    if month_keys(gcm_fit.time) != month_keys(era5_fit.time):
        raise ValueError("CMIP6 and ERA5 fit-period year/month sequences differ")
    if month_keys(gcm_eval.time) != month_keys(era5_eval.time):
        raise ValueError("CMIP6 and ERA5 evaluation-period year/month sequences differ")

    # The observations and model represent the same months but can use
    # different calendar classes and different nominal days within a month.
    # Use the CMIP6 time coordinate in the combined diagnostic output so
    # xarray does not form the union of two incompatible time indexes.
    era5_eval_output = era5_eval.assign_coords(time=gcm_eval.time)

    all_rows: list[dict] = []
    manifest: dict[str, object] = {
        "model": arguments.model,
        "variable": arguments.variable,
        "fit_period": [FIT_START_YEAR, FIT_END_YEAR],
        "evaluation_period": [EVAL_START_YEAR, EVAL_END_YEAR],
        "cmip6_file": str(cmip6_file),
        "era5_file": str(era5_file),
        "n_quantiles": arguments.n_quantiles,
        "correction_mode": correction_mode,
        "bounds": [],
        "netcdf_written": not arguments.skip_netcdf,
    }

    latitudes = np.asarray(gcm_eval.lat.values, dtype=np.float64)
    longitudes = np.asarray(gcm_eval.lon.values, dtype=np.float64)

    for lower, upper in arguments.bounds:
        bounds_label = f"Q{round(lower * 100):02d}-Q{round(upper * 100):02d}"
        print(f"Running {bounds_label}")
        qm_output = np.full(gcm_eval.shape, np.nan, dtype=np.float32)
        qdm_output = np.full(gcm_eval.shape, np.nan, dtype=np.float32)
        eval_months = np.asarray(gcm_eval.time.dt.month.values, dtype=np.int64)
        fit_months = np.asarray(gcm_fit.time.dt.month.values, dtype=np.int64)

        for month in MONTHS:
            fit_indices = np.flatnonzero(fit_months == month)
            eval_indices = np.flatnonzero(eval_months == month)
            result = correct_month(
                np.asarray(gcm_fit.isel(time=fit_indices).values),
                np.asarray(era5_fit.isel(time=fit_indices).values),
                np.asarray(gcm_eval.isel(time=eval_indices).values),
                arguments.variable,
                lower,
                upper,
                arguments.n_quantiles,
                correction_mode,
            )
            qm_output[eval_indices] = result.qm.astype(np.float32)
            qdm_output[eval_indices] = result.qdm.astype(np.float32)
            all_rows.extend(
                evaluation_rows(
                    arguments.model,
                    arguments.variable,
                    bounds_label,
                    month,
                    np.asarray(era5_eval.isel(time=eval_indices).values),
                    np.asarray(gcm_eval.isel(time=eval_indices).values),
                    result.qm,
                    result.qdm,
                    result,
                    latitudes,
                    longitudes,
                    correction_mode,
                )
            )

        corrected = xr.Dataset(
            {
                "raw": gcm_eval.astype(np.float32),
                "qm": xr.DataArray(qm_output, coords=gcm_eval.coords, dims=gcm_eval.dims),
                "qdm": xr.DataArray(qdm_output, coords=gcm_eval.coords, dims=gcm_eval.dims),
                "era5_reference": era5_eval_output.astype(np.float32),
            }
        )
        corrected.attrs.update(
            {
                "purpose": "Historical holdout smoke test; not final future projection",
                "fit_period": f"{FIT_START_YEAR}-{FIT_END_YEAR}",
                "evaluation_period": f"{EVAL_START_YEAR}-{EVAL_END_YEAR}",
                "quantile_bounds": bounds_label,
                "correction_mode": correction_mode,
                "qm_probability_source": "historical model empirical CDF",
                "qdm_probability_source": "evaluation-period model empirical CDF",
            }
        )
        for name in corrected.data_vars:
            corrected[name].attrs["units"] = gcm_data.attrs.get("units", "")
        bounds_manifest = {
            "label": bounds_label,
            "lower": lower,
            "upper": upper,
            "netcdf": None,
        }
        if not arguments.skip_netcdf:
            netcdf_path = output_root / (
                f"holdout_{arguments.model}_{arguments.variable}_"
                f"{correction_mode}_{bounds_label}.nc"
            )
            encoding = {
                name: {"zlib": True, "complevel": 3}
                for name in corrected.data_vars
            }
            try:
                corrected.to_netcdf(netcdf_path, encoding=encoding)
            except ValueError:
                corrected.to_netcdf(netcdf_path)
            bounds_manifest["netcdf"] = str(netcdf_path)
        manifest["bounds"].append(bounds_manifest)

    monthly_path = output_root / "holdout_monthly_metrics.csv"
    summary_path = output_root / "holdout_method_summary.csv"
    manifest_path = output_root / "holdout_run_manifest.json"
    write_csv(monthly_path, all_rows)
    write_csv(summary_path, build_summary(all_rows))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Monthly metrics: {monthly_path}")
    print(f"Method summary : {summary_path}")
    print(f"Run manifest   : {manifest_path}")
    print("SMOKE TEST COMPLETED")
    return 0


def run_self_test() -> int:
    rng = np.random.default_rng(20260918)
    reference_fit = rng.uniform(2.0, 12.0, size=(30, 2, 3))
    model_fit = reference_fit * 2.0
    model_eval = rng.uniform(4.0, 24.0, size=(26, 2, 3))
    result = correct_month(
        model_fit, reference_fit, model_eval, "sfcWind", 0.02, 0.98, 99
    )
    expected = model_eval * 0.5
    if not np.allclose(result.qm, expected, rtol=1e-10, atol=1e-10):
        raise AssertionError("QM failed the known factor-of-two synthetic case")
    if not np.allclose(result.qdm, expected, rtol=1e-10, atol=1e-10):
        raise AssertionError("QDM failed the known factor-of-two synthetic case")
    if result.qm_fallback_percent != 0.0 or result.qdm_fallback_percent != 0.0:
        raise AssertionError("Unexpected ratio fallback in synthetic case")

    temperature_reference = rng.uniform(270.0, 300.0, size=(30, 2, 3))
    temperature_model = temperature_reference * 1.02
    temperature_eval = rng.uniform(275.0, 305.0, size=(26, 2, 3))
    temperature_ratio = correct_month(
        temperature_model,
        temperature_reference,
        temperature_eval,
        "tas",
        0.02,
        0.98,
        99,
        "ratio",
    )
    expected_temperature = temperature_eval / 1.02
    if not np.allclose(
        temperature_ratio.qm,
        expected_temperature,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise AssertionError("Ratio temperature QM failed the Kelvin test")
    if not np.allclose(
        temperature_ratio.qdm,
        expected_temperature,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise AssertionError("Ratio temperature QDM failed the Kelvin test")
    if (
        temperature_ratio.qm_fallback_percent != 0.0
        or temperature_ratio.qdm_fallback_percent != 0.0
    ):
        raise AssertionError("Unexpected Kelvin temperature ratio fallback")
    print(
        "SELF-TEST PASSED: ratio wind and ratio Kelvin temperature cases "
        "recovered their known factors"
    )
    return 0


def parse_bounds(values: list[str]) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    for value in values:
        try:
            lower_text, upper_text = value.split(",", maxsplit=1)
            lower, upper = float(lower_text), float(upper_text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"Invalid --bounds {value!r}; use LOWER,UPPER such as 0.02,0.98"
            ) from error
        if not (0.0 < lower < upper < 1.0):
            raise argparse.ArgumentTypeError(
                f"Invalid --bounds {value!r}; require 0 < LOWER < UPPER < 1"
            )
        parsed.append((lower, upper))
    return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a historical holdout smoke test for monthly QM and QDM."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--variable", choices=tuple(EXPECTED_UNITS), default=DEFAULT_VARIABLE
    )
    parser.add_argument("--cmip6-root", type=Path, default=CMIP6_ROOT)
    parser.add_argument("--era5-root", type=Path, default=ERA5_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--bounds",
        action="append",
        default=None,
        metavar="LOWER,UPPER",
        help="Repeat to compare bounds; defaults to 0.02,0.98 and 0.05,0.95.",
    )
    parser.add_argument("--n-quantiles", type=int, default=99)
    parser.add_argument(
        "--correction-mode",
        choices=("auto", "additive", "ratio"),
        default="auto",
        help=(
            "Correction scaling. Auto uses additive for tas and ratio for "
            "rsds/sfcWind. Use ratio for the paper-style tas reproduction."
        ),
    )
    parser.add_argument(
        "--skip-netcdf",
        action="store_true",
        help="Write CSV diagnostics without the large corrected NetCDF files.",
    )
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.n_quantiles < 3:
        parser.error("--n-quantiles must be at least 3")
    try:
        arguments.bounds = parse_bounds(
            arguments.bounds or ["0.02,0.98", "0.05,0.95"]
        )
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    return arguments


def main() -> int:
    arguments = parse_arguments()
    if arguments.self_test:
        return run_self_test()
    return run_validation(arguments)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
