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

"""Single-model future QM/QDM production pilot (version 2).

Expected layout::

    scripts/
    |-- 03_bias_correction/qm_qdm_future_production_test.py
    `-- 03_bias_correction/qm_qdm_holdout_smoke_test.py

The tested QM/QDM numerical core is imported automatically from the sibling
``03_bias_correction`` directory.  Keeping both files in the same directory is
also supported.

Default workflow
----------------
1. Fit monthly correction relationships with CMIP6 historical and ERA5 data
   from 1959-2014 (56 samples per calendar month).
2. Correct one CMIP6 scenario from 2015-2100.
3. Write paper-style multiplicative QM and a 30-year moving-window QDM branch:
      tas     : multiplicative QM + additive QDM
      rsds    : multiplicative QM + multiplicative QDM
      sfcWind : multiplicative QM + multiplicative QDM
4. Use Q02-Q98 and 99 quantile nodes by default.

Examples
--------
    python qm_qdm_future_production_test.py --dry-run
    python qm_qdm_future_production_test.py
    python qm_qdm_future_production_test.py --model ACCESS-CM2 --scenario ssp245
"""

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
except ImportError:
    xr = None

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_SCRIPT_NAME = "qm_qdm_holdout_smoke_test.py"
CORE_SEARCH_DIRECTORIES = (
    SCRIPT_DIR,
    SCRIPT_DIR.parent / "03_bias_correction",
)
for candidate_directory in CORE_SEARCH_DIRECTORIES:
    if (candidate_directory / CORE_SCRIPT_NAME).is_file():
        candidate_text = str(candidate_directory)
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)
        break
else:
    searched = ", ".join(str(path / CORE_SCRIPT_NAME) for path in CORE_SEARCH_DIRECTORIES)
    raise FileNotFoundError(
        f"Cannot find the tested QM/QDM core script. Searched: {searched}"
    )

import qm_qdm_holdout_smoke_test as core

from area_weights import china_intersection_weights, geodesic_gridcell_areas


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "archive" / "abandoned_pilots" / "qm_qdm_future_production_pilot_v2"
DEFAULT_MODEL = "CAS-ESM2-0"
DEFAULT_SCENARIO = "ssp245"
DEFAULT_VARIABLES = ("tas", "rsds", "sfcWind")
HISTORICAL_START_YEAR = 1959
HISTORICAL_END_YEAR = 2014
FUTURE_START_YEAR = 2015
FUTURE_END_YEAR = 2100
MONTHS = tuple(range(1, 13))
DEFAULT_QDM_WINDOW_YEARS = 30
SIGNAL_PERIODS = (
    ("near", 2020, 2039),
    ("mid", 2040, 2069),
    ("late", 2070, 2099),
)

SCENARIO_ALIASES = {
    "ssp126": ("ssp126", "SSP1-2.6", "ssp1-2.6"),
    "ssp245": ("ssp245", "SSP2-4.5", "ssp2-4.5"),
    "ssp585": ("ssp585", "SSP5-8.5", "ssp5-8.5"),
}


@dataclass(frozen=True)
class SeriesInspection:
    files: tuple[Path, ...]
    month_keys: tuple[int, ...]
    latitudes: np.ndarray
    longitudes: np.ndarray
    unit: str
    calendar: str


@dataclass(frozen=True)
class MovingQDMResult:
    corrected: np.ndarray
    tail_percent: float
    fallback_percent: float
    tail_mask: np.ndarray
    fallback_mask: np.ndarray


@dataclass(frozen=True)
class SpatialWeights:
    """QC spatial weighting over a native rectilinear grid.

    ``intersects`` is the boolean China selection mask and ``area_km2`` the
    per-cell geodesic intersection area (0 outside the boundary).  Both have
    shape ``(lat, lon)``.  ``region``/``weighting`` document the aggregation so
    downstream CSVs are self-describing.  The corrected NetCDF is never clipped
    or re-weighted by this object.
    """

    intersects: np.ndarray
    area_km2: np.ndarray
    region: str
    weighting: str


def expected_month_keys(start_year: int, end_year: int) -> list[int]:
    return list(range(start_year * 12, end_year * 12 + 12))


def validate_month_sequence(
    actual: list[int] | tuple[int, ...],
    start_year: int,
    end_year: int,
    label: str,
) -> None:
    expected = expected_month_keys(start_year, end_year)
    actual_list = list(actual)
    if len(actual_list) != len(set(actual_list)):
        raise ValueError(f"{label} contains duplicate year-month entries")
    if actual_list != expected:
        actual_set = set(actual_list)
        missing = [key for key in expected if key not in actual_set]
        extra = [key for key in actual_list if key not in set(expected)]

        def display(keys: list[int]) -> list[str]:
            return [f"{key // 12:04d}-{key % 12 + 1:02d}" for key in keys[:6]]

        raise ValueError(
            f"{label} must contain every month from {start_year}-01 to "
            f"{end_year}-12; found {len(actual_list)} steps, "
            f"missing={display(missing)}, extra={display(extra)}"
        )


def same_grid(left: SeriesInspection, right: SeriesInspection) -> bool:
    return (
        left.latitudes.shape == right.latitudes.shape
        and left.longitudes.shape == right.longitudes.shape
        and np.allclose(left.latitudes, right.latitudes, rtol=0.0, atol=1.0e-10)
        and np.allclose(left.longitudes, right.longitudes, rtol=0.0, atol=1.0e-10)
    )


def calendar_name(time: xr.DataArray) -> str:
    try:
        return str(time.dt.calendar)
    except Exception:
        return str(time.encoding.get("calendar", "standard"))


def inspect_series(
    files: list[Path],
    variable: str,
    label: str,
) -> SeriesInspection:
    if not files:
        raise ValueError(f"No files supplied for {label}")
    all_keys: list[int] = []
    first_lat: np.ndarray | None = None
    first_lon: np.ndarray | None = None
    unit = ""
    calendars: set[str] = set()

    for path in files:
        with xr.open_dataset(path, decode_times=True) as source:
            dataset = core.standardize_dataset(source)
            data = core.extract_variable(dataset, variable, label)
            lat = np.asarray(data.lat.values, dtype=np.float64)
            lon = np.asarray(data.lon.values, dtype=np.float64)
            if first_lat is None:
                first_lat, first_lon = lat, lon
                unit = str(data.attrs.get("units", ""))
            elif (
                lat.shape != first_lat.shape
                or lon.shape != first_lon.shape
                or not np.allclose(lat, first_lat, rtol=0.0, atol=1.0e-10)
                or not np.allclose(lon, first_lon, rtol=0.0, atol=1.0e-10)
            ):
                raise ValueError(f"Grid changes between files for {label}: {path}")
            all_keys.extend(core.month_keys(data.time))
            calendars.add(calendar_name(data.time))

    if len(calendars) != 1:
        raise ValueError(f"Multiple calendars found for {label}: {sorted(calendars)}")
    assert first_lat is not None and first_lon is not None
    return SeriesInspection(
        files=tuple(files),
        month_keys=tuple(sorted(all_keys)),
        latitudes=first_lat,
        longitudes=first_lon,
        unit=unit,
        calendar=next(iter(calendars)),
    )


def load_series(files: list[Path], variable: str, label: str) -> xr.DataArray:
    arrays: list[xr.DataArray] = []
    for path in files:
        with xr.open_dataset(path, decode_times=True) as source:
            dataset = core.standardize_dataset(source)
            arrays.append(core.extract_variable(dataset, variable, label).load())
    if len(arrays) == 1:
        result = arrays[0]
    else:
        result = xr.concat(
            arrays,
            dim="time",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        )
    return result.sortby("time").transpose("time", "lat", "lon")


def find_future_files(
    cmip6_root: Path,
    model: str,
    scenario: str,
    variable: str,
) -> list[Path]:
    aliases = SCENARIO_ALIASES.get(scenario, (scenario,))
    populated: list[tuple[Path, list[Path]]] = []
    for alias in aliases:
        root = cmip6_root / model / alias / variable
        files = sorted(path for path in root.rglob("*.nc") if path.is_file()) if root.is_dir() else []
        if files:
            populated.append((root, files))
    if not populated:
        tried = ", ".join(str(cmip6_root / model / alias / variable) for alias in aliases)
        raise FileNotFoundError(f"No future NetCDF files for {model}/{scenario}/{variable}; tried {tried}")
    if len(populated) > 1:
        roots = ", ".join(str(item[0]) for item in populated)
        raise RuntimeError(f"More than one scenario directory matched: {roots}")
    return populated[0][1]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Weighted quantile over the pooled finite cells of ``values``.

    ``weights`` is broadcast against ``values``; cells with NaN values or
    non-positive weight are excluded.  Generalises ``area_weighted_median`` to
    an arbitrary quantile.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), values.shape)
    flat_values = values.ravel()
    flat_weights = weights.ravel()
    keep = np.isfinite(flat_values) & (flat_weights > 0.0)
    ordered_values = flat_values[keep]
    ordered_weights = flat_weights[keep]
    if ordered_values.size == 0:
        return float("nan")
    order = np.argsort(ordered_values)
    ordered_values = ordered_values[order]
    ordered_weights = ordered_weights[order]
    cumulative = np.cumsum(ordered_weights)
    total = cumulative[-1]
    if total <= 0.0:
        return float("nan")
    index = int(np.searchsorted(cumulative, quantile * total))
    return float(ordered_values[index])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), values.shape)
    flat_values = values.ravel()
    flat_weights = weights.ravel()
    keep = np.isfinite(flat_values) & (flat_weights > 0.0)
    if not keep.any():
        return float("nan")
    return float(np.sum(flat_values[keep] * flat_weights[keep]) / np.sum(flat_weights[keep]))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), values.shape)
    flat_values = values.ravel()
    flat_weights = weights.ravel()
    keep = np.isfinite(flat_values) & (flat_weights > 0.0)
    if keep.sum() < 2:
        return float("nan")
    selected = flat_values[keep]
    selected_weights = flat_weights[keep]
    mean = float(np.sum(selected * selected_weights) / np.sum(selected_weights))
    variance = float(np.sum(selected_weights * (selected - mean) ** 2) / np.sum(selected_weights))
    return float(np.sqrt(variance))


def finite_statistics(
    values: np.ndarray,
    spatial_weights: SpatialWeights,
) -> dict[str, float | int]:
    """Intersection-area-weighted pooled diagnostics over ``(time, lat, lon)``.

    Extrema (min/max) are taken over every China-intersecting finite cell;
    the mean/median/quantiles/std are area-weighted by the cell's China
    intersection area, and the missing/negative proportions are area-weighted
    proportions.  Variable values are never multiplied by the area fraction.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-2:] != spatial_weights.intersects.shape:
        raise ValueError(
            f"Spatial weights shape {spatial_weights.intersects.shape} does not "
            f"match data {array.shape[-2:]}"
        )
    keep = np.broadcast_to(spatial_weights.intersects, array.shape)
    area = np.broadcast_to(spatial_weights.area_km2, array.shape)
    finite = np.isfinite(array)
    selected = finite & keep
    finite_count = int(selected.sum())
    total_area = float(np.sum(area[keep]))
    finite_area = float(np.sum(area[selected]))
    if finite_count == 0:
        return {
            "finite_count": 0,
            "missing_percent": 100.0 if total_area > 0.0 else np.nan,
            "negative_percent": np.nan,
            "minimum": np.nan,
            "p01": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "p99": np.nan,
            "maximum": np.nan,
        }
    finite_values = array[selected]
    finite_weights = area[selected]
    negative_area = float(np.sum(area[selected & (array < 0.0)]))
    return {
        "finite_count": finite_count,
        "missing_percent": (
            100.0 * (total_area - finite_area) / total_area if total_area > 0.0 else 0.0
        ),
        "negative_percent": (
            100.0 * negative_area / finite_area if finite_area > 0.0 else np.nan
        ),
        "minimum": float(np.min(finite_values)),
        "p01": _weighted_quantile(finite_values, finite_weights, 0.01),
        "median": _weighted_quantile(finite_values, finite_weights, 0.50),
        "mean": _weighted_mean(finite_values, finite_weights),
        "std": _weighted_std(finite_values, finite_weights),
        "p99": _weighted_quantile(finite_values, finite_weights, 0.99),
        "maximum": float(np.max(finite_values)),
    }


def spatial_statistics(
    values: np.ndarray,
    prefix: str,
    spatial_weights: SpatialWeights,
) -> dict[str, float]:
    """Intersection-area-weighted spatial statistics over ``(lat, lon)``."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-2:] != spatial_weights.intersects.shape:
        raise ValueError(
            f"Spatial weights shape {spatial_weights.intersects.shape} does not "
            f"match data {array.shape[-2:]}"
        )
    keep = spatial_weights.intersects & np.isfinite(array)
    if not keep.any():
        return {
            f"{prefix}_spatial_median": np.nan,
            f"{prefix}_spatial_mean": np.nan,
            f"{prefix}_spatial_p10": np.nan,
            f"{prefix}_spatial_p90": np.nan,
            f"{prefix}_spatial_max": np.nan,
        }
    finite_values = array[keep]
    finite_weights = spatial_weights.area_km2[keep]
    return {
        f"{prefix}_spatial_median": _weighted_quantile(finite_values, finite_weights, 0.50),
        f"{prefix}_spatial_mean": _weighted_mean(finite_values, finite_weights),
        f"{prefix}_spatial_p10": _weighted_quantile(finite_values, finite_weights, 0.10),
        f"{prefix}_spatial_p90": _weighted_quantile(finite_values, finite_weights, 0.90),
        f"{prefix}_spatial_max": float(np.max(finite_values)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def change_field(
    future_mean: np.ndarray,
    baseline_mean: np.ndarray,
    variable: str,
) -> np.ndarray:
    future_mean = np.asarray(future_mean, dtype=np.float64)
    baseline_mean = np.asarray(baseline_mean, dtype=np.float64)
    if variable == "tas":
        return future_mean - baseline_mean
    floor = core.RATIO_DENOMINATOR_FLOOR[variable]
    valid = np.isfinite(baseline_mean) & (np.abs(baseline_mean) > floor)
    result = np.full(baseline_mean.shape, np.nan, dtype=np.float64)
    np.divide(future_mean, baseline_mean, out=result, where=valid)
    return (result - 1.0) * 100.0


def moving_window_qdm(
    model_fit: np.ndarray,
    reference_fit: np.ndarray,
    model_future: np.ndarray,
    variable: str,
    lower: float,
    upper: float,
    n_quantiles: int,
    window_years: int,
    correction_mode: str,
) -> MovingQDMResult:
    """Correct one calendar month using a centred future CDF window."""
    original_shape = model_future.shape
    fit = model_fit.reshape(model_fit.shape[0], -1).astype(np.float64)
    obs = reference_fit.reshape(reference_fit.shape[0], -1).astype(np.float64)
    target = model_future.reshape(model_future.shape[0], -1).astype(np.float64)
    if window_years > target.shape[0]:
        raise ValueError(
            f"QDM window ({window_years}) exceeds future samples ({target.shape[0]})"
        )

    complete_history = np.all(np.isfinite(fit), axis=0) & np.all(
        np.isfinite(obs), axis=0
    )
    fit[:, ~complete_history] = np.nan
    obs[:, ~complete_history] = np.nan
    probabilities = np.linspace(lower, upper, n_quantiles)
    model_quantiles = core.nanquantile(fit, probabilities)
    reference_quantiles = core.nanquantile(obs, probabilities)
    corrected = np.full(target.shape, np.nan, dtype=np.float64)
    all_tail = np.zeros(target.shape, dtype=bool)
    all_fallback = np.zeros(target.shape, dtype=bool)
    tail_count = 0
    fallback_count = 0
    finite_count = 0

    for index in range(target.shape[0]):
        start = min(
            max(index - window_years // 2, 0),
            target.shape[0] - window_years,
        )
        window = target[start : start + window_years]
        value = target[index : index + 1].copy()
        valid_cell = complete_history & np.all(np.isfinite(window), axis=0)
        value[:, ~valid_cell] = np.nan
        window = window.copy()
        window[:, ~valid_cell] = np.nan
        local_probability = core.empirical_probabilities_against_sample(value, window)
        tail = np.isfinite(local_probability) & (
            (local_probability < lower) | (local_probability > upper)
        )
        historical_model_q = core.interpolate_uniform_quantiles(
            model_quantiles, local_probability, lower, upper
        )
        historical_reference_q = core.interpolate_uniform_quantiles(
            reference_quantiles, local_probability, lower, upper
        )

        if correction_mode == "additive":
            corrected_value = value + historical_reference_q - historical_model_q
            fallback = np.zeros(value.shape, dtype=bool)
        elif correction_mode == "ratio":
            floor = core.RATIO_DENOMINATOR_FLOOR[variable]
            safe = np.isfinite(historical_model_q) & (
                np.abs(historical_model_q) > floor
            )
            corrected_value = np.full(value.shape, np.nan, dtype=np.float64)
            np.multiply(
                value,
                historical_reference_q / np.where(safe, historical_model_q, 1.0),
                out=corrected_value,
                where=safe,
            )
            fallback = np.isfinite(value) & ~safe
            additive_value = value + historical_reference_q - historical_model_q
            corrected_value[fallback] = additive_value[fallback]
        else:
            raise ValueError(f"Unsupported QDM correction mode: {correction_mode}")

        minimum = core.PHYSICAL_MINIMUM[variable]
        if minimum is not None:
            corrected_value = np.maximum(corrected_value, minimum)
        corrected[index] = corrected_value[0]
        all_tail[index] = tail[0]
        all_fallback[index] = fallback[0]
        tail_count += int(tail.sum())
        fallback_count += int(fallback.sum())
        finite_count += int(np.isfinite(value).sum())

    percentage = lambda count: 100.0 * count / finite_count if finite_count else np.nan
    return MovingQDMResult(
        corrected=corrected.reshape(original_shape),
        tail_percent=percentage(tail_count),
        fallback_percent=percentage(fallback_count),
        tail_mask=all_tail.reshape(original_shape),
        fallback_mask=all_fallback.reshape(original_shape),
    )


def diagnostic_percentage(
    event_mask: np.ndarray,
    values: np.ndarray,
    spatial_weights: SpatialWeights,
) -> float:
    """Area-weighted event proportion (trigger / fallback) over finite cells."""
    events = np.asarray(event_mask, dtype=bool)
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-2:] != spatial_weights.intersects.shape:
        raise ValueError(
            f"Spatial weights shape {spatial_weights.intersects.shape} does not "
            f"match data {array.shape[-2:]}"
        )
    keep = np.broadcast_to(spatial_weights.intersects, array.shape)
    area = np.broadcast_to(spatial_weights.area_km2, array.shape)
    finite = np.isfinite(array)
    denominator = float(np.sum(area[finite & keep]))
    if denominator <= 0.0:
        return np.nan
    numerator = float(np.sum(area[events & finite & keep]))
    return 100.0 * numerator / denominator


def qm_ratio_diagnostic_masks(
    model_fit: np.ndarray,
    reference_fit: np.ndarray,
    model_future: np.ndarray,
    variable: str,
    lower: float,
    upper: float,
    n_quantiles: int,
) -> tuple[np.ndarray, np.ndarray]:
    original_shape = model_future.shape
    fit = model_fit.reshape(model_fit.shape[0], -1).astype(np.float64)
    obs = reference_fit.reshape(reference_fit.shape[0], -1).astype(np.float64)
    target = model_future.reshape(model_future.shape[0], -1).astype(np.float64)
    complete = (
        np.all(np.isfinite(fit), axis=0)
        & np.all(np.isfinite(obs), axis=0)
        & np.all(np.isfinite(target), axis=0)
    )
    fit[:, ~complete] = np.nan
    target[:, ~complete] = np.nan
    probabilities = core.empirical_probabilities_against_sample(target, fit)
    tail = np.isfinite(probabilities) & (
        (probabilities < lower) | (probabilities > upper)
    )
    quantile_nodes = np.linspace(lower, upper, n_quantiles)
    model_quantiles = core.nanquantile(fit, quantile_nodes)
    model_q = core.interpolate_uniform_quantiles(
        model_quantiles, probabilities, lower, upper
    )
    floor = core.RATIO_DENOMINATOR_FLOOR[variable]
    safe = np.isfinite(model_q) & (np.abs(model_q) > floor)
    fallback = np.isfinite(target) & ~safe
    return tail.reshape(original_shape), fallback.reshape(original_shape)


def build_spatial_mask(
    shapefile_path: Path | None,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> SpatialWeights:
    """Return QC spatial weights; correction output itself is never spatially clipped.

    With a China boundary shapefile, the selection mask and per-cell weights are
    the geodesic intersection between each native grid cell and the boundary
    polygon (``china_intersection_weights`` from ``scripts/common/area_weights``).
    The corrected NetCDF keeps the full rectangular native grid in every case.
    """
    if shapefile_path is None:
        areas = geodesic_gridcell_areas(latitudes, longitudes)
        return SpatialWeights(
            intersects=np.ones(areas.shape, dtype=bool),
            area_km2=areas,
            region="rectangular_domain",
            weighting="geodesic_gridcell_area",
        )
    if not shapefile_path.is_file():
        raise FileNotFoundError(f"China boundary shapefile does not exist: {shapefile_path}")
    weights = china_intersection_weights(latitudes, longitudes, shapefile_path)
    intersects = weights["china_intersects"]
    if not intersects.any():
        raise ValueError("China boundary polygon intersects no native grid cells")
    return SpatialWeights(
        intersects=intersects,
        area_km2=weights["china_intersection_area_km2"],
        region="china_boundary",
        weighting="geodesic_china_intersection_area",
    )


def make_qc_rows(
    model: str,
    scenario: str,
    variable: str,
    future: xr.DataArray,
    qm_output: np.ndarray,
    qdm_output: np.ndarray,
    monthly_diagnostics: dict[int, dict[str, float]],
    spatial_weights: SpatialWeights,
    qdm_window_years: int,
) -> list[dict]:
    months = np.asarray(future.time.dt.month.values, dtype=np.int64)
    rows: list[dict] = []
    modes = {
        "RAW": "none",
        "QM": "ratio",
        "QDM": "additive" if variable == "tas" else "ratio",
    }
    for month in MONTHS:
        indices = np.flatnonzero(months == month)
        arrays = {
            "RAW": np.asarray(future.isel(time=indices).values),
            "QM": qm_output[indices],
            "QDM": qdm_output[indices],
        }
        for method, values in arrays.items():
            diagnostics = monthly_diagnostics[month]
            row = {
                "model": model,
                "scenario": scenario,
                "variable": variable,
                "month": month,
                "method": method,
                "correction_mode": modes[method],
                "qc_region": spatial_weights.region,
                "spatial_weighting": spatial_weights.weighting,
                "qdm_future_cdf_window_years": (
                    qdm_window_years if method == "QDM" else ""
                ),
                "samples": len(indices),
                "tail_rule_trigger_percent": 0.0 if method == "RAW" else diagnostics[f"{method.lower()}_tail_percent"],
                "ratio_fallback_percent": 0.0 if method == "RAW" else diagnostics[f"{method.lower()}_fallback_percent"],
            }
            row.update(finite_statistics(values, spatial_weights))
            rows.append(row)
    return rows


def make_signal_rows(
    model: str,
    scenario: str,
    variable: str,
    historical_model: xr.DataArray,
    historical_reference: xr.DataArray,
    future: xr.DataArray,
    qm_output: np.ndarray,
    qdm_output: np.ndarray,
    spatial_weights: SpatialWeights,
) -> list[dict]:
    hist_months = np.asarray(historical_model.time.dt.month.values, dtype=np.int64)
    reference_months = np.asarray(historical_reference.time.dt.month.values, dtype=np.int64)
    future_months = np.asarray(future.time.dt.month.values, dtype=np.int64)
    rows: list[dict] = []
    signal_unit = "K" if variable == "tas" else "%"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for month in MONTHS:
            hi = np.flatnonzero(hist_months == month)
            ri = np.flatnonzero(reference_months == month)
            fi = np.flatnonzero(future_months == month)
            historical_model_mean = np.nanmean(
                np.asarray(historical_model.isel(time=hi).values), axis=0
            )
            historical_reference_mean = np.nanmean(
                np.asarray(historical_reference.isel(time=ri).values), axis=0
            )
            raw_future_mean = np.nanmean(np.asarray(future.isel(time=fi).values), axis=0)
            raw_signal = change_field(raw_future_mean, historical_model_mean, variable)
            method_values = {
                "RAW": raw_future_mean,
                "QM": np.nanmean(qm_output[fi], axis=0),
                "QDM": np.nanmean(qdm_output[fi], axis=0),
            }
            for method, future_mean in method_values.items():
                baseline = historical_model_mean if method == "RAW" else historical_reference_mean
                signal = change_field(future_mean, baseline, variable)
                difference = np.abs(signal - raw_signal)
                row = {
                    "model": model,
                    "scenario": scenario,
                    "variable": variable,
                    "month": month,
                    "method": method,
                    "signal_unit": signal_unit,
                    "qc_region": spatial_weights.region,
                    "spatial_weighting": spatial_weights.weighting,
                }
                row.update(spatial_statistics(signal, "change_signal", spatial_weights))
                row.update(
                    spatial_statistics(
                        difference,
                        "absolute_difference_from_raw_signal",
                        spatial_weights,
                    )
                )
                rows.append(row)
    return rows


def make_period_signal_rows(
    model: str,
    scenario: str,
    variable: str,
    historical_model: xr.DataArray,
    historical_reference: xr.DataArray,
    future: xr.DataArray,
    qm_output: np.ndarray,
    qdm_output: np.ndarray,
    spatial_weights: SpatialWeights,
) -> list[dict]:
    hist_months = np.asarray(historical_model.time.dt.month.values, dtype=np.int64)
    reference_months = np.asarray(historical_reference.time.dt.month.values, dtype=np.int64)
    future_months = np.asarray(future.time.dt.month.values, dtype=np.int64)
    future_years = np.asarray(future.time.dt.year.values, dtype=np.int64)
    rows: list[dict] = []
    signal_unit = "K" if variable == "tas" else "%"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for period_name, start_year, end_year in SIGNAL_PERIODS:
            for month in MONTHS:
                hi = np.flatnonzero(hist_months == month)
                ri = np.flatnonzero(reference_months == month)
                fi = np.flatnonzero(
                    (future_months == month)
                    & (future_years >= start_year)
                    & (future_years <= end_year)
                )
                historical_model_mean = np.nanmean(
                    np.asarray(historical_model.isel(time=hi).values), axis=0
                )
                historical_reference_mean = np.nanmean(
                    np.asarray(historical_reference.isel(time=ri).values), axis=0
                )
                raw_future_mean = np.nanmean(
                    np.asarray(future.isel(time=fi).values), axis=0
                )
                raw_signal = change_field(
                    raw_future_mean, historical_model_mean, variable
                )
                method_values = {
                    "RAW": raw_future_mean,
                    "QM": np.nanmean(qm_output[fi], axis=0),
                    "QDM": np.nanmean(qdm_output[fi], axis=0),
                }
                for method, future_mean in method_values.items():
                    baseline = (
                        historical_model_mean
                        if method == "RAW"
                        else historical_reference_mean
                    )
                    signal = change_field(future_mean, baseline, variable)
                    difference = np.abs(signal - raw_signal)
                    row = {
                        "model": model,
                        "scenario": scenario,
                        "variable": variable,
                        "period": period_name,
                        "period_start_year": start_year,
                        "period_end_year": end_year,
                        "month": month,
                        "method": method,
                        "signal_unit": signal_unit,
                        "qc_region": spatial_weights.region,
                        "spatial_weighting": spatial_weights.weighting,
                    }
                    row.update(
                        spatial_statistics(signal, "change_signal", spatial_weights)
                    )
                    row.update(
                        spatial_statistics(
                            difference,
                            "absolute_difference_from_raw_signal",
                            spatial_weights,
                        )
                    )
                    rows.append(row)
    return rows


def write_netcdf(
    path: Path,
    model: str,
    scenario: str,
    variable: str,
    future: xr.DataArray,
    qm_output: np.ndarray,
    qdm_output: np.ndarray,
    bounds_label: str,
    n_quantiles: int,
    qdm_window_years: int,
) -> None:
    raw = future.astype(np.float32).rename("raw")
    qm = xr.DataArray(
        qm_output.astype(np.float32),
        coords=future.coords,
        dims=future.dims,
        name="qm_paper",
    )
    qdm = xr.DataArray(
        qdm_output.astype(np.float32),
        coords=future.coords,
        dims=future.dims,
        name="qdm_improved",
    )
    unit = str(future.attrs.get("units", ""))
    raw.attrs.update({"units": unit, "correction_method": "none"})
    qm.attrs.update(
        {
            "units": unit,
            "correction_method": "monthly multiplicative quantile mapping",
            "role": "paper-style reproduction baseline",
        }
    )
    qdm_mode = "additive" if variable == "tas" else "multiplicative"
    qdm.attrs.update(
        {
            "units": unit,
            "correction_method": (
                f"monthly {qdm_mode} quantile delta mapping with "
                f"{qdm_window_years}-year moving future CDF"
            ),
            "role": "change-preserving improved branch",
        }
    )
    output = xr.Dataset({"raw": raw, "qm_paper": qm, "qdm_improved": qdm})
    output.attrs.update(
        {
            "model": model,
            "scenario": scenario,
            "variable": variable,
            "historical_fit_period": f"{HISTORICAL_START_YEAR}-{HISTORICAL_END_YEAR}",
            "future_period": f"{FUTURE_START_YEAR}-{FUTURE_END_YEAR}",
            "quantile_bounds": bounds_label,
            "quantile_nodes": n_quantiles,
            "qdm_future_cdf_window_years": qdm_window_years,
            "note": "Production pilot. Inspect QC CSV files before batch processing.",
        }
    )
    for coordinate in ("time", "lat", "lon"):
        if coordinate in output.coords and output[coordinate].attrs.get("bounds") not in output:
            output[coordinate].attrs.pop("bounds", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunksizes = (
        min(12, future.sizes["time"]),
        future.sizes["lat"],
        future.sizes["lon"],
    )
    encoding = {
        name: {
            "zlib": True,
            "complevel": 3,
            "dtype": "float32",
            "chunksizes": chunksizes,
        }
        for name in output.data_vars
    }
    try:
        output.to_netcdf(path, encoding=encoding)
    except (ValueError, TypeError):
        output.to_netcdf(path)


def run_variable(
    arguments: argparse.Namespace,
    variable: str,
    historical_file: Path,
    era5_file: Path,
    future_files: list[Path],
    output_root: Path,
    spatial_weights: SpatialWeights,
) -> dict[str, object]:
    historical = load_series([historical_file], variable, f"CMIP6 historical/{arguments.model}")
    reference = load_series([era5_file], variable, f"ERA5/{arguments.model}")
    future = load_series(future_files, variable, f"CMIP6 {arguments.scenario}/{arguments.model}")
    validate_month_sequence(core.month_keys(historical.time), HISTORICAL_START_YEAR, HISTORICAL_END_YEAR, "historical CMIP6")
    validate_month_sequence(core.month_keys(reference.time), HISTORICAL_START_YEAR, HISTORICAL_END_YEAR, "ERA5")
    validate_month_sequence(core.month_keys(future.time), FUTURE_START_YEAR, FUTURE_END_YEAR, "future CMIP6")

    historical_months = np.asarray(historical.time.dt.month.values, dtype=np.int64)
    reference_months = np.asarray(reference.time.dt.month.values, dtype=np.int64)
    future_months = np.asarray(future.time.dt.month.values, dtype=np.int64)
    qm_output = np.full(future.shape, np.nan, dtype=np.float32)
    qdm_output = np.full(future.shape, np.nan, dtype=np.float32)
    monthly_diagnostics: dict[int, dict[str, float]] = {}

    for month in MONTHS:
        hi = np.flatnonzero(historical_months == month)
        ri = np.flatnonzero(reference_months == month)
        fi = np.flatnonzero(future_months == month)
        model_fit = np.asarray(historical.isel(time=hi).values)
        reference_fit = np.asarray(reference.isel(time=ri).values)
        model_future = np.asarray(future.isel(time=fi).values)

        qm_result = core.correct_month(
            model_fit,
            reference_fit,
            model_future,
            variable,
            arguments.lower,
            arguments.upper,
            arguments.n_quantiles,
            "ratio",
        )
        qdm_mode = "additive" if variable == "tas" else "ratio"
        qdm_result = moving_window_qdm(
            model_fit,
            reference_fit,
            model_future,
            variable,
            arguments.lower,
            arguments.upper,
            arguments.n_quantiles,
            arguments.qdm_window_years,
            qdm_mode,
        )
        qm_tail_mask, qm_fallback_mask = qm_ratio_diagnostic_masks(
            model_fit,
            reference_fit,
            model_future,
            variable,
            arguments.lower,
            arguments.upper,
            arguments.n_quantiles,
        )
        qm_output[fi] = qm_result.qm.astype(np.float32)
        qdm_output[fi] = qdm_result.corrected.astype(np.float32)
        monthly_diagnostics[month] = {
            "qm_tail_percent": diagnostic_percentage(
                qm_tail_mask, qm_result.qm, spatial_weights
            ),
            "qdm_tail_percent": diagnostic_percentage(
                qdm_result.tail_mask, qdm_result.corrected, spatial_weights
            ),
            "qm_fallback_percent": diagnostic_percentage(
                qm_fallback_mask, qm_result.qm, spatial_weights
            ),
            "qdm_fallback_percent": diagnostic_percentage(
                qdm_result.fallback_mask, qdm_result.corrected, spatial_weights
            ),
        }
        print(f"  {variable}: month {month:02d} completed")

    qc_rows = make_qc_rows(
        arguments.model,
        arguments.scenario,
        variable,
        future,
        qm_output,
        qdm_output,
        monthly_diagnostics,
        spatial_weights,
        arguments.qdm_window_years,
    )
    signal_rows = make_signal_rows(
        arguments.model,
        arguments.scenario,
        variable,
        historical,
        reference,
        future,
        qm_output,
        qdm_output,
        spatial_weights,
    )
    period_signal_rows = make_period_signal_rows(
        arguments.model,
        arguments.scenario,
        variable,
        historical,
        reference,
        future,
        qm_output,
        qdm_output,
        spatial_weights,
    )
    qc_path = output_root / f"future_qc_{variable}.csv"
    signal_path = output_root / f"future_change_signal_{variable}.csv"
    period_signal_path = output_root / f"future_period_change_signal_{variable}.csv"
    netcdf_path = output_root / (
        f"corrected_{arguments.model}_{arguments.scenario}_{variable}_"
        f"{arguments.bounds_label}_MW{arguments.qdm_window_years}_201501-210012.nc"
    )
    write_csv(qc_path, qc_rows)
    write_csv(signal_path, signal_rows)
    write_csv(period_signal_path, period_signal_rows)
    if not arguments.skip_netcdf:
        write_netcdf(
            netcdf_path,
            arguments.model,
            arguments.scenario,
            variable,
            future,
            qm_output,
            qdm_output,
            arguments.bounds_label,
            arguments.n_quantiles,
            arguments.qdm_window_years,
        )
    return {
        "variable": variable,
        "historical_file": str(historical_file),
        "era5_file": str(era5_file),
        "future_files": [str(path) for path in future_files],
        "qm_mode": "ratio",
        "qdm_mode": "additive" if variable == "tas" else "ratio",
        "qc_csv": str(qc_path),
        "signal_csv": str(signal_path),
        "period_signal_csv": str(period_signal_path),
        "netcdf": None if arguments.skip_netcdf else str(netcdf_path),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit 1959-2014 monthly QM/QDM and correct one 2015-2100 CMIP6 scenario."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=tuple(core.EXPECTED_UNITS),
        default=list(DEFAULT_VARIABLES),
    )
    parser.add_argument("--cmip6-root", type=Path, default=core.CMIP6_ROOT)
    parser.add_argument("--era5-root", type=Path, default=core.ERA5_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lower", type=float, default=0.02)
    parser.add_argument("--upper", type=float, default=0.98)
    parser.add_argument("--n-quantiles", type=int, default=99)
    parser.add_argument(
        "--qdm-window-years",
        type=int,
        default=DEFAULT_QDM_WINDOW_YEARS,
        help="Future CDF moving-window length for QDM (default: 30 years).",
    )
    parser.add_argument(
        "--china-shapefile",
        type=Path,
        help=(
            "Optional China boundary .shp used only for QC statistics; corrected "
            "NetCDF data retain the full rectangular model grid."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-netcdf",
        action="store_true",
        help="Compute and write CSV diagnostics without corrected NetCDF files.",
    )
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if not (0.0 < arguments.lower < arguments.upper < 1.0):
        parser.error("Require 0 < --lower < --upper < 1")
    if arguments.n_quantiles < 3:
        parser.error("--n-quantiles must be at least 3")
    if arguments.qdm_window_years < 5:
        parser.error("--qdm-window-years must be at least 5")
    if arguments.qdm_window_years > FUTURE_END_YEAR - FUTURE_START_YEAR + 1:
        parser.error("--qdm-window-years cannot exceed the future period length")
    arguments.bounds_label = (
        f"Q{round(arguments.lower * 100):02d}-Q{round(arguments.upper * 100):02d}"
    )
    return arguments


def run_self_test() -> int:
    core.run_self_test()
    rng = np.random.default_rng(20260919)
    reference = rng.uniform(270.0, 300.0, size=(56, 2, 3))
    model = reference + 2.0
    future = rng.uniform(272.0, 305.0, size=(86, 2, 3))
    result = core.correct_month(
        model,
        reference,
        future,
        "tas",
        0.02,
        0.98,
        99,
        "additive",
    )
    if not np.allclose(result.qdm, future - 2.0, rtol=1.0e-10, atol=1.0e-10):
        raise AssertionError("Additive temperature QDM failed the known +2 K case")
    if result.qdm_fallback_percent != 0.0:
        raise AssertionError("Additive temperature QDM unexpectedly used ratio fallback")
    moving_result = moving_window_qdm(
        model,
        reference,
        future,
        "tas",
        0.02,
        0.98,
        99,
        30,
        "additive",
    )
    if not np.allclose(
        moving_result.corrected, future - 2.0, rtol=1.0e-10, atol=1.0e-10
    ):
        raise AssertionError("Moving-window additive QDM failed the known +2 K case")
    print(
        "SELF-TEST PASSED: whole-period and 30-year moving-window additive QDM "
        "recovered the known 2 K bias"
    )
    return 0


def main() -> int:
    arguments = parse_arguments()
    if arguments.self_test:
        return run_self_test()
    if xr is None:
        raise RuntimeError("xarray is required; activate the pvwind environment")

    output_root = arguments.output_root / arguments.model / arguments.scenario
    print(f"Model       : {arguments.model}")
    print(f"Scenario    : {arguments.scenario}")
    print(f"Variables   : {', '.join(arguments.variables)}")
    print(f"Fit period  : {HISTORICAL_START_YEAR}-{HISTORICAL_END_YEAR}")
    print(f"Future      : {FUTURE_START_YEAR}-{FUTURE_END_YEAR}")
    print(f"Bounds      : {arguments.bounds_label}")
    print(f"QDM window  : {arguments.qdm_window_years} years")
    print(f"Output root : {output_root}")

    resolved: dict[str, tuple[Path, Path, list[Path], SeriesInspection, SeriesInspection, SeriesInspection]] = {}
    for variable in arguments.variables:
        historical_file = core.find_cmip6_file(arguments.cmip6_root, arguments.model, variable)
        era5_file = core.find_era5_file(arguments.era5_root, arguments.model)
        future_files = find_future_files(
            arguments.cmip6_root, arguments.model, arguments.scenario, variable
        )
        historical_info = inspect_series(
            [historical_file], variable, f"CMIP6 historical/{arguments.model}"
        )
        era5_info = inspect_series([era5_file], variable, f"ERA5/{arguments.model}")
        future_info = inspect_series(
            future_files, variable, f"CMIP6 {arguments.scenario}/{arguments.model}"
        )
        validate_month_sequence(
            historical_info.month_keys,
            HISTORICAL_START_YEAR,
            HISTORICAL_END_YEAR,
            f"historical CMIP6/{variable}",
        )
        validate_month_sequence(
            era5_info.month_keys,
            HISTORICAL_START_YEAR,
            HISTORICAL_END_YEAR,
            f"ERA5/{variable}",
        )
        validate_month_sequence(
            future_info.month_keys,
            FUTURE_START_YEAR,
            FUTURE_END_YEAR,
            f"future CMIP6/{variable}",
        )
        if not same_grid(historical_info, era5_info):
            raise ValueError(f"Historical CMIP6 and ERA5 grids differ for {variable}")
        if not same_grid(historical_info, future_info):
            raise ValueError(f"Historical and future CMIP6 grids differ for {variable}")
        resolved[variable] = (
            historical_file,
            era5_file,
            future_files,
            historical_info,
            era5_info,
            future_info,
        )
        print(f"\n[{variable}]")
        print(f"  Historical : {historical_file}")
        print(f"  ERA5       : {era5_file}")
        print(f"  Future     : {len(future_files)} file(s)")
        for path in future_files:
            print(f"               {path}")
        print(f"  Grid       : {future_info.latitudes.size} lat x {future_info.longitudes.size} lon")
        print(f"  Unit       : {future_info.unit}")
        print(f"  Calendar   : {future_info.calendar}")
        print(f"  Time steps : {len(future_info.month_keys)}")

    first_variable = arguments.variables[0]
    mask_grid = resolved[first_variable][5]
    spatial_weights = build_spatial_mask(
        arguments.china_shapefile,
        mask_grid.latitudes,
        mask_grid.longitudes,
    )
    if arguments.china_shapefile is None:
        print(
            "\nWARNING: no --china-shapefile supplied; QC statistics use the full "
            "rectangular domain, including sea and non-China grid cells."
        )
    else:
        print(
            f"\nQC region   : China boundary ({int(spatial_weights.intersects.sum())} "
            f"intersecting grid cells; {float(spatial_weights.area_km2.sum()):.1f} km^2 "
            "intersection area)"
        )

    if arguments.dry_run:
        print("\nDRY RUN COMPLETED: inputs, calendars, grids, units, and monthly coverage are valid.")
        print("No output files were written.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "purpose": "single-model future correction production pilot",
        "model": arguments.model,
        "scenario": arguments.scenario,
        "historical_fit_period": [HISTORICAL_START_YEAR, HISTORICAL_END_YEAR],
        "future_period": [FUTURE_START_YEAR, FUTURE_END_YEAR],
        "bounds": [arguments.lower, arguments.upper],
        "n_quantiles": arguments.n_quantiles,
        "qdm_future_cdf_window_years": arguments.qdm_window_years,
        "qc_region": spatial_weights.region,
        "qc_spatial_weighting": spatial_weights.weighting,
        "china_shapefile": (
            None if arguments.china_shapefile is None else str(arguments.china_shapefile)
        ),
        "qc_grid_cell_intersects": int(spatial_weights.intersects.sum()),
        "qc_intersection_area_km2": float(spatial_weights.area_km2.sum()),
        "variables": [],
    }
    for variable in arguments.variables:
        historical_file, era5_file, future_files, *_ = resolved[variable]
        print(f"\nCorrecting {variable} ...")
        result = run_variable(
            arguments,
            variable,
            historical_file,
            era5_file,
            future_files,
            output_root,
            spatial_weights,
        )
        manifest["variables"].append(result)

    manifest_path = output_root / "future_run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 100)
    print("FUTURE PRODUCTION PILOT COMPLETED")
    print("=" * 100)
    print(f"Output root : {output_root}")
    print(f"Manifest    : {manifest_path}")
    print(
        "Inspect future_qc_*.csv and future_period_change_signal_*.csv "
        "before batch processing."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
