#!/usr/bin/env python
r"""Build equal-weight 17-model ensemble products on a common 1-degree grid.

Place this script in::

    scripts/05_figure_data

Default input and output::

    input : data/processed/energy_metrics
    output: results/figure_data/multimodel_ensemble

Methodological rules
--------------------
* First calculate period means and changes independently for every model.
* Bilinearly interpolate those model-level diagnostics to the common grid.
* Then form an equal-weight ensemble across models.
* Never concatenate model time series before calculating diagnostics.
* Primary periods are 1994-2014, 2040-2060, and 2080-2100.
"""

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
from area_weights import china_intersection_weights  # noqa: E402
# -------------------------------------------------------------------

import argparse
import csv
import json
import math
import sys
import traceback
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import xarray as xr
except ImportError as error:  # pragma: no cover
    raise RuntimeError(
        "xarray is required. Activate the pvwind conda environment."
    ) from error


SCENARIOS = ("ssp126", "ssp245", "ssp585")
METHODS = ("paper_qm", "optimized")
PERIODS = {
    "historical": (1994, 2014),
    "mid_century": (2040, 2060),
    "late_century": (2080, 2100),
}
FUTURE_PERIODS = ("mid_century", "late_century")
ENERGY_VARIABLES = ("pvpot", "wpd")
DEFINITIONS = (
    "monthly_full",
    "monthly_climatology",
    "seasonal_full",
    "seasonal_climatology",
)
ENSEMBLE_STATISTICS = ("mean", "median", "std", "p10", "p90", "model_count")
CHANGE_METRICS = ("absolute_change", "relative_change_percent")
METHOD_DIFFERENCE_METRICS = (
    "period_absolute_difference",
    "period_relative_difference_percent",
    "absolute_change_difference",
    "relative_change_difference_percentage_points",
)
EXPECTED_MODELS = 17

# Aggregate summary tables and the transition-warning CSV were relocated during
# the path reorganization: the combined CSVs now live under results/tables/ and
# the transition warnings under results/audits/ (not under energy_metrics/).
TABLES_ROOT = get_path("results_tables")
AUDITS_ROOT = get_path("results_audits")
COMBINED_PERIOD_SUMMARY = TABLES_ROOT / "combined_energy_period_summary.csv"
COMBINED_HUB_HEIGHT_SENSITIVITY = (
    TABLES_ROOT / "combined_hub_height_sensitivity.csv"
)
TRANSITION_WARNINGS = (
    AUDITS_ROOT / "energy_transition_z_gt_3_warnings.csv"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build common-grid equal-weight ensemble energy, change, "
            "complementarity, uncertainty, and sensitivity products."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Default: <project-root>/data/processed/energy_metrics",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Default: <project-root>/results/figure_data/multimodel_ensemble",
    )
    parser.add_argument(
        "--china-shapefile",
        type=Path,
        help=(
            "Default: <project-root>/data/boundaries/"
            "china_qc_boundary.shp"
        ),
    )
    parser.add_argument("--models", nargs="+", help="Optional model subset.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
    )
    parser.add_argument("--expected-models", type=int, default=EXPECTED_MODELS)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--lon-min", type=float, default=73.0)
    parser.add_argument("--lon-max", type=float, default=135.0)
    parser.add_argument("--lat-min", type=float, default=15.0)
    parser.add_argument("--lat-max", type=float, default=54.0)
    parser.add_argument(
        "--min-models",
        type=int,
        help="Minimum valid models per target cell. Default: ceil(80%% of models).",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.resolution <= 0.0:
        parser.error("--resolution must be positive")
    if arguments.lon_min >= arguments.lon_max:
        parser.error("Require --lon-min < --lon-max")
    if arguments.lat_min >= arguments.lat_max:
        parser.error("Require --lat-min < --lat-max")
    return arguments


def decoder():
    try:
        return xr.coders.CFDatetimeCoder(use_cftime=True)
    except AttributeError:  # pragma: no cover
        return True


def open_netcdf(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=decoder())


def discover_models(
    input_root: Path,
    requested: list[str] | None,
    scenarios: list[str],
    expected: int,
) -> list[str]:
    if requested:
        models = sorted(dict.fromkeys(requested))
    else:
        models = sorted(
            path.name
            for path in input_root.iterdir()
            if path.is_dir()
            and path.name != "production_audit"
            and all((path / scenario).is_dir() for scenario in scenarios)
        )
    if not models:
        raise FileNotFoundError(f"No model directories found below {input_root}")
    if requested is None and len(models) != expected:
        raise ValueError(
            f"Expected {expected} models; found {len(models)}: "
            + ", ".join(models)
        )
    return models


def coordinate(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    values = start + step * np.arange(count + 1, dtype=np.float64)
    if values[-1] < stop - step * 1.0e-6:
        values = np.append(values, values[-1] + step)
    return values


def build_china_weights(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    shapefile: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Intersection-area weights on the common 1-degree grid.

    Returns ``(gridcell_area_km2, china_intersection_area_km2, china_intersects,
    china_area_fraction, metadata)``.  A cell is selected when its geodesic area
    overlaps the China boundary polygon; partial coastal cells are retained (no
    centre-point pre-NaN of border cells).
    """
    result = china_intersection_weights(latitudes, longitudes, shapefile)
    gridcell_area = result["gridcell_area_km2"]
    intersection = result["china_intersection_area_km2"]
    intersects = result["china_intersects"]
    fraction = result["china_area_fraction"]
    counts = result["counts"]
    if not intersects.any():
        raise ValueError("China boundary selected zero common-grid cells")
    metadata: dict[str, object] = {
        "shapefile": str(shapefile),
        "selection_rule": "grid-cell area intersecting the China polygon",
        "weighting": "geodesic intersection area with the China boundary (WGS84)",
        "selected_cells": int(counts["intersects"]),
        "fully_inside_cells": int(counts["fully_inside"]),
        "partial_cells": int(counts["partial"]),
        "outside_cells": int(counts["outside"]),
        "total_cells": int(counts["total"]),
        "selected_fraction": float(counts["intersects"] / counts["total"]),
        "china_boundary_area_km2": float(result["china_boundary_area_km2"]),
        "china_intersection_total_km2": float(intersection.sum()),
    }
    return gridcell_area, intersection, intersects, fraction, metadata


def expected_paths(
    input_root: Path,
    models: list[str],
    scenarios: list[str],
) -> list[Path]:
    paths = []
    for model in models:
        for scenario in scenarios:
            for method in METHODS:
                paths.append(
                    input_root
                    / model
                    / scenario
                    / f"energy_monthly_{model}_{scenario}_{method}_199401-210012.nc"
                )
                paths.append(
                    input_root
                    / model
                    / scenario
                    / f"complementarity_{model}_{scenario}_{method}.nc"
                )
    paths.extend(
        [
            COMBINED_PERIOD_SUMMARY,
            COMBINED_HUB_HEIGHT_SENSITIVITY,
            TRANSITION_WARNINGS,
        ]
    )
    return paths


def regrid_map(
    data: xr.DataArray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    china_intersects: np.ndarray,
) -> np.ndarray:
    array = data.squeeze(drop=True)
    if set(array.dims) != {"lat", "lon"}:
        raise ValueError(f"Expected a lat/lon map; got dims={array.dims}")
    array = array.transpose("lat", "lon").sortby("lat").sortby("lon")
    target = array.interp(
        lat=xr.DataArray(latitudes, dims="lat"),
        lon=xr.DataArray(longitudes, dims="lon"),
        method="linear",
    )
    values = np.asarray(target.values, dtype=np.float64)
    values[~china_intersects] = np.nan
    return values


def period_mean(dataset: xr.Dataset, variable: str, start: int, end: int) -> xr.DataArray:
    years = np.asarray(dataset.time.dt.year.values, dtype=np.int64)
    indices = np.flatnonzero((years >= start) & (years <= end))
    expected = (end - start + 1) * 12
    if indices.size != expected:
        raise ValueError(
            f"{variable}/{start}-{end}: expected {expected} months; found {indices.size}"
        )
    return dataset[variable].isel(time=indices).mean("time", skipna=True)


def safe_relative_percent(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        100.0 * numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=np.isfinite(denominator) & (np.abs(denominator) > 1.0e-12),
    )


def safe_relative_percent_dataarray(
    numerator: xr.DataArray, denominator: xr.DataArray
) -> xr.DataArray:
    return xr.where(
        np.isfinite(denominator) & (np.abs(denominator) > 1.0e-12),
        100.0 * numerator / denominator,
        np.nan,
    )


def ensemble_statistics(
    arrays: list[np.ndarray], min_models: int
) -> dict[str, np.ndarray]:
    if not arrays:
        raise ValueError("Cannot summarize an empty model stack")
    stack = np.stack(arrays, axis=0).astype(np.float64)
    count = np.sum(np.isfinite(stack), axis=0).astype(np.float32)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        outputs = {
            "mean": np.nanmean(stack, axis=0),
            "median": np.nanmedian(stack, axis=0),
            "std": np.nanstd(stack, axis=0),
            "p10": np.nanquantile(stack, 0.10, axis=0),
            "p90": np.nanquantile(stack, 0.90, axis=0),
            "model_count": count,
        }
    valid = count >= min_models
    for name in ENSEMBLE_STATISTICS[:-1]:
        outputs[name] = np.where(valid, outputs[name], np.nan).astype(np.float32)
    return outputs


def sign_agreement_percent(arrays: list[np.ndarray], min_models: int) -> np.ndarray:
    stack = np.stack(arrays, axis=0).astype(np.float64)
    finite = np.isfinite(stack)
    count = finite.sum(axis=0)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
        positive = ((stack > 0.0) & finite).sum(axis=0)
        negative = ((stack < 0.0) & finite).sum(axis=0)
        matching = np.where(mean > 0.0, positive, np.where(mean < 0.0, negative, 0))
        agreement = 100.0 * matching / count
    agreement[(count < min_models) | (mean == 0.0)] = np.nan
    return agreement.astype(np.float32)


def stats_array(
    storage: dict[tuple[str, ...], list[np.ndarray]],
    keys: Iterable[tuple[str, ...]],
    min_models: int,
) -> np.ndarray:
    pieces = []
    for key in keys:
        summary = ensemble_statistics(storage[key], min_models)
        pieces.append(np.stack([summary[name] for name in ENSEMBLE_STATISTICS]))
    return np.stack(pieces)


def write_dataset(path: Path, dataset: xr.Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {}
    for name, variable in dataset.data_vars.items():
        options: dict[str, object] = {"zlib": True, "complevel": 3}
        if np.issubdtype(variable.dtype, np.floating):
            options["dtype"] = "float32"
        encoding[name] = options
    try:
        dataset.to_netcdf(path, encoding=encoding)
    except (ValueError, TypeError):
        dataset.to_netcdf(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def numeric_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "model_count": 0,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "p10": math.nan,
            "p90": math.nan,
            "minimum": math.nan,
            "maximum": math.nan,
        }
    return {
        "model_count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def build_scalar_summaries(
    input_root: Path, output_root: Path
) -> tuple[Path, Path, Path]:
    period_rows = read_csv(COMBINED_PERIOD_SUMMARY)
    grouped: dict[tuple[str, ...], list[tuple[str, float]]] = defaultdict(list)
    model_values: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for row in period_rows:
        if row.get("aggregation") != "period_mean_map":
            continue
        if row.get("variable") not in ENERGY_VARIABLES:
            continue
        key = (
            row["scenario"],
            row["method"],
            row["period"],
            row["variable"],
        )
        value = float(row["area_weighted_spatial_mean"])
        grouped[key].append((row["model"], value))
        model_values[key][row["model"]] = value
    scalar_rows = []
    for key, values in sorted(grouped.items()):
        row: dict[str, object] = {
            "scenario": key[0],
            "method": key[1],
            "period": key[2],
            "variable": key[3],
            "ensemble_rule": "equal weight across model-level China means",
        }
        row.update(numeric_summary(value for _, value in values))
        scalar_rows.append(row)
    scalar_path = output_root / "multimodel_scalar_summary.csv"
    write_csv(scalar_path, scalar_rows)

    difference_rows = []
    for scenario in SCENARIOS:
        for period in PERIODS:
            for variable in ENERGY_VARIABLES:
                paper = model_values.get((scenario, "paper_qm", period, variable), {})
                optimized = model_values.get(
                    (scenario, "optimized", period, variable), {}
                )
                common = sorted(set(paper) & set(optimized))
                absolute = [optimized[m] - paper[m] for m in common]
                relative = [
                    100.0 * (optimized[m] / paper[m] - 1.0)
                    for m in common
                    if paper[m] != 0.0
                ]
                for metric, values in (
                    ("optimized_minus_paper", absolute),
                    ("optimized_minus_paper_percent", relative),
                ):
                    row = {
                        "scenario": scenario,
                        "period": period,
                        "variable": variable,
                        "metric": metric,
                    }
                    row.update(numeric_summary(values))
                    difference_rows.append(row)
    difference_path = output_root / "multimodel_method_difference_scalar_summary.csv"
    write_csv(difference_path, difference_rows)

    height_rows = read_csv(COMBINED_HUB_HEIGHT_SENSITIVITY)
    height_groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in height_rows:
        key = (
            row["scenario"],
            row["method"],
            row["period"],
            row["hub_height_m"],
        )
        height_groups[key].append(float(row["area_weighted_mean_wpd"]))
    height_summary = []
    for key, values in sorted(height_groups.items()):
        row = {
            "scenario": key[0],
            "method": key[1],
            "period": key[2],
            "hub_height_m": float(key[3]),
            "variable": "wpd",
        }
        row.update(numeric_summary(values))
        height_summary.append(row)
    height_path = output_root / "multimodel_hub_height_sensitivity.csv"
    write_csv(height_path, height_summary)
    return scalar_path, difference_path, height_path


def warning_family(variable: str) -> str:
    if variable in {"tas_c", "tcell"}:
        return "temperature"
    if variable in {"rsds", "pvpot", "performance_ratio"}:
        return "solar_pv"
    if variable in {"sfcWind_10m", "wind_speed_hub", "wpd"}:
        return "wind"
    return variable


def summarize_transition_warnings(
    input_root: Path, output_root: Path
) -> tuple[Path, Path]:
    source = TRANSITION_WARNINGS
    rows = read_csv(source)
    exact: dict[tuple[str, ...], dict[str, object]] = {}
    families: dict[tuple[str, ...], dict[str, object]] = {}
    for row in rows:
        exact_key = (row["model"], row["year_month"], row["variable"])
        family_key = (
            row["model"],
            row["year_month"],
            warning_family(row["variable"]),
        )
        z = float(row["standardized_anomaly_z"])
        for key, storage, label in (
            (exact_key, exact, row["variable"]),
            (family_key, families, warning_family(row["variable"])),
        ):
            record = storage.setdefault(
                key,
                {
                    "model": row["model"],
                    "year_month": row["year_month"],
                    "warning_group": label,
                    "scenarios": set(),
                    "methods": set(),
                    "variables": set(),
                    "raw_rows": 0,
                    "maximum_absolute_z": 0.0,
                },
            )
            record["scenarios"].add(row["scenario"])
            record["methods"].add(row["method"])
            record["variables"].add(row["variable"])
            record["raw_rows"] += 1
            record["maximum_absolute_z"] = max(
                float(record["maximum_absolute_z"]), abs(z)
            )

    def finalize(records: dict[tuple[str, ...], dict[str, object]]):
        output = []
        for _, record in sorted(records.items()):
            output.append(
                {
                    "model": record["model"],
                    "year_month": record["year_month"],
                    "warning_group": record["warning_group"],
                    "scenario_count": len(record["scenarios"]),
                    "scenarios": ";".join(sorted(record["scenarios"])),
                    "method_count": len(record["methods"]),
                    "methods": ";".join(sorted(record["methods"])),
                    "variables": ";".join(sorted(record["variables"])),
                    "raw_rows": record["raw_rows"],
                    "maximum_absolute_z": record["maximum_absolute_z"],
                }
            )
        return output

    exact_path = output_root / "transition_warning_unique_variable_events.csv"
    family_path = output_root / "transition_warning_unique_family_events.csv"
    write_csv(exact_path, finalize(exact))
    write_csv(family_path, finalize(families))
    return exact_path, family_path


def main() -> int:
    arguments = parse_arguments()
    script_dir = Path(__file__).resolve().parent
    project_root = PROJECT_ROOT
    input_root = (
        arguments.input_root.expanduser().resolve()
        if arguments.input_root
        else get_path("data_processed_energy_metrics")
    )
    output_root = (
        arguments.output_root.expanduser().resolve()
        if arguments.output_root
        else get_path("results_figure_data_multimodel_ensemble")
    )
    shapefile = (
        arguments.china_shapefile.expanduser().resolve()
        if arguments.china_shapefile
        else get_path("china_shapefile")
    )
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    models = discover_models(
        input_root,
        arguments.models,
        list(arguments.scenarios),
        arguments.expected_models,
    )
    min_models = (
        arguments.min_models
        if arguments.min_models is not None
        else int(math.ceil(0.80 * len(models)))
    )
    if not (1 <= min_models <= len(models)):
        raise ValueError(
            f"--min-models must be between 1 and {len(models)}; got {min_models}"
        )
    latitudes = coordinate(arguments.lat_min, arguments.lat_max, arguments.resolution)
    longitudes = coordinate(arguments.lon_min, arguments.lon_max, arguments.resolution)
    china_gridcell, china_area, china_intersects, china_fraction, china_metadata = build_china_weights(
        latitudes, longitudes, shapefile
    )
    required = expected_paths(input_root, models, list(arguments.scenarios))
    missing = [path for path in required if not path.is_file()]

    print("=" * 100)
    print("MULTI-MODEL ENSEMBLE PRODUCTION")
    print("=" * 100)
    print(f"Models          : {len(models)}")
    print(f"Scenarios       : {', '.join(arguments.scenarios)}")
    print(f"Combinations    : {len(models) * len(arguments.scenarios)}")
    print(f"Common grid     : {latitudes.size} lat x {longitudes.size} lon")
    print(f"Resolution      : {arguments.resolution:g} degree")
    print(
        f"China cells     : {int(china_intersects.sum())}/{int(china_intersects.size)} "
        f"intersecting (fully-inside {china_metadata['fully_inside_cells']}, "
        f"partial {china_metadata['partial_cells']})"
    )
    print(f"Minimum models  : {min_models}/{len(models)}")
    print(f"Input root      : {input_root}")
    print(f"Output root     : {output_root}")
    if missing:
        display = "\n  ".join(str(path) for path in missing[:20])
        raise FileNotFoundError(
            f"Missing {len(missing)} required input files:\n  {display}"
        )
    if arguments.dry_run:
        print("DRY RUN COMPLETED: all required inputs and the common China grid are valid.")
        print("No ensemble files were written.")
        return 0

    period_storage: dict[tuple[str, ...], list[np.ndarray]] = defaultdict(list)
    change_storage: dict[tuple[str, ...], list[np.ndarray]] = defaultdict(list)
    method_difference_storage: dict[tuple[str, ...], list[np.ndarray]] = defaultdict(list)
    complementarity_storage: dict[tuple[str, ...], list[np.ndarray]] = defaultdict(list)

    try:
        for model_index, model in enumerate(models, start=1):
            print(f"[{model_index:02d}/{len(models):02d}] {model}")
            for scenario in arguments.scenarios:
                model_maps: dict[str, dict[str, dict[str, np.ndarray]]] = {}
                native_maps: dict[str, dict[str, dict[str, xr.DataArray]]] = {}
                native_changes: dict[
                    str, dict[str, dict[str, dict[str, xr.DataArray]]]
                ] = {}
                for method in METHODS:
                    energy_path = input_root / model / scenario / (
                        f"energy_monthly_{model}_{scenario}_{method}_199401-210012.nc"
                    )
                    method_maps: dict[str, dict[str, np.ndarray]] = {}
                    method_native_maps: dict[str, dict[str, xr.DataArray]] = {}
                    method_native_changes: dict[
                        str, dict[str, dict[str, xr.DataArray]]
                    ] = {}
                    with open_netcdf(energy_path) as dataset:
                        for period, (start, end) in PERIODS.items():
                            method_maps[period] = {}
                            method_native_maps[period] = {}
                            for variable in ENERGY_VARIABLES:
                                native = period_mean(
                                    dataset, variable, start, end
                                ).load()
                                method_native_maps[period][variable] = native
                                common = regrid_map(
                                    native, latitudes, longitudes, china_intersects
                                )
                                method_maps[period][variable] = common
                                period_storage[
                                    (scenario, method, period, variable)
                                ].append(common)
                        for future_period in FUTURE_PERIODS:
                            method_native_changes[future_period] = {}
                            for variable in ENERGY_VARIABLES:
                                historical_native = method_native_maps[
                                    "historical"
                                ][variable]
                                future_native = method_native_maps[future_period][
                                    variable
                                ]
                                absolute_native = (
                                    future_native - historical_native
                                )
                                relative_native = safe_relative_percent_dataarray(
                                    absolute_native, historical_native
                                )
                                method_native_changes[future_period][variable] = {
                                    "absolute_change": absolute_native,
                                    "relative_change_percent": relative_native,
                                }
                                absolute = regrid_map(
                                    absolute_native,
                                    latitudes,
                                    longitudes,
                                    china_intersects,
                                )
                                relative = regrid_map(
                                    relative_native,
                                    latitudes,
                                    longitudes,
                                    china_intersects,
                                )
                                change_storage[
                                    (
                                        scenario,
                                        method,
                                        future_period,
                                        variable,
                                        "absolute_change",
                                    )
                                ].append(absolute)
                                change_storage[
                                    (
                                        scenario,
                                        method,
                                        future_period,
                                        variable,
                                        "relative_change_percent",
                                    )
                                ].append(relative)
                    model_maps[method] = method_maps
                    native_maps[method] = method_native_maps
                    native_changes[method] = method_native_changes

                    comp_path = input_root / model / scenario / (
                        f"complementarity_{model}_{scenario}_{method}.nc"
                    )
                    with open_netcdf(comp_path) as complementarity:
                        for period in PERIODS:
                            for definition in DEFINITIONS:
                                native = complementarity["spearman_rho"].sel(
                                    period=period, definition=definition
                                )
                                common = regrid_map(
                                    native, latitudes, longitudes, china_intersects
                                )
                                complementarity_storage[
                                    (scenario, method, period, definition)
                                ].append(common)

                for period in PERIODS:
                    for variable in ENERGY_VARIABLES:
                        paper_native = native_maps["paper_qm"][period][variable]
                        optimized_native = native_maps["optimized"][period][
                            variable
                        ]
                        absolute_native = optimized_native - paper_native
                        relative_native = safe_relative_percent_dataarray(
                            absolute_native, paper_native
                        )
                        absolute = regrid_map(
                            absolute_native, latitudes, longitudes, china_intersects
                        )
                        relative = regrid_map(
                            relative_native, latitudes, longitudes, china_intersects
                        )
                        method_difference_storage[
                            (
                                scenario,
                                period,
                                variable,
                                "period_absolute_difference",
                            )
                        ].append(absolute)
                        method_difference_storage[
                            (
                                scenario,
                                period,
                                variable,
                                "period_relative_difference_percent",
                            )
                        ].append(relative)
                for future_period in FUTURE_PERIODS:
                    for variable in ENERGY_VARIABLES:
                        paper = native_changes["paper_qm"][future_period][variable]
                        optimized = native_changes["optimized"][future_period][
                            variable
                        ]
                        absolute_change_difference = regrid_map(
                            optimized["absolute_change"] - paper["absolute_change"],
                            latitudes,
                            longitudes,
                            china_intersects,
                        )
                        relative_change_difference = regrid_map(
                            optimized["relative_change_percent"]
                            - paper["relative_change_percent"],
                            latitudes,
                            longitudes,
                            china_intersects,
                        )
                        method_difference_storage[
                            (
                                scenario,
                                future_period,
                                variable,
                                "absolute_change_difference",
                            )
                        ].append(absolute_change_difference)
                        method_difference_storage[
                            (
                                scenario,
                                future_period,
                                variable,
                                "relative_change_difference_percentage_points",
                            )
                        ].append(relative_change_difference)
    except Exception as error:
        print(f"ERROR while building model diagnostics: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1

    coords = {
        "scenario": list(arguments.scenarios),
        "method": list(METHODS),
        "period": list(PERIODS),
        "future_period": list(FUTURE_PERIODS),
        "energy_variable": list(ENERGY_VARIABLES),
        "definition": list(DEFINITIONS),
        "statistic": list(ENSEMBLE_STATISTICS),
        "change_metric": list(CHANGE_METRICS),
        "method_difference_metric": list(METHOD_DIFFERENCE_METRICS),
        "lat": latitudes,
        "lon": longitudes,
    }

    period_values = np.empty(
        (
            len(arguments.scenarios),
            len(METHODS),
            len(PERIODS),
            len(ENERGY_VARIABLES),
            len(ENSEMBLE_STATISTICS),
            latitudes.size,
            longitudes.size,
        ),
        dtype=np.float32,
    )
    for si, scenario in enumerate(arguments.scenarios):
        for mi, method in enumerate(METHODS):
            for pi, period in enumerate(PERIODS):
                for vi, variable in enumerate(ENERGY_VARIABLES):
                    summary = ensemble_statistics(
                        period_storage[(scenario, method, period, variable)],
                        min_models,
                    )
                    period_values[si, mi, pi, vi] = np.stack(
                        [summary[name] for name in ENSEMBLE_STATISTICS]
                    )
    period_dataset = xr.Dataset(
        {
            "period_value": (
                (
                    "scenario",
                    "method",
                    "period",
                    "energy_variable",
                    "statistic",
                    "lat",
                    "lon",
                ),
                period_values,
            ),
            "gridcell_area_km2": (("lat", "lon"), china_gridcell),
            "china_intersects": (("lat", "lon"), china_intersects.astype(np.int8)),
            "china_intersection_area_km2": (("lat", "lon"), china_area),
            "china_area_fraction": (("lat", "lon"), china_fraction),
        },
        coords={
            key: coords[key]
            for key in (
                "scenario",
                "method",
                "period",
                "energy_variable",
                "statistic",
                "lat",
                "lon",
            )
        },
    )
    period_dataset.attrs.update(
        ensemble_rule="equal weight after model-level period calculation and regridding",
        regridding="xarray linear interpolation of intensive model-level diagnostics",
        common_grid_resolution_degree=float(arguments.resolution),
        minimum_valid_models=int(min_models),
        periods="1994-2014, 2040-2060, 2080-2100",
    )

    change_values = np.empty(
        (
            len(arguments.scenarios),
            len(METHODS),
            len(FUTURE_PERIODS),
            len(ENERGY_VARIABLES),
            len(CHANGE_METRICS),
            len(ENSEMBLE_STATISTICS),
            latitudes.size,
            longitudes.size,
        ),
        dtype=np.float32,
    )
    agreement_values = np.empty(
        (
            len(arguments.scenarios),
            len(METHODS),
            len(FUTURE_PERIODS),
            len(ENERGY_VARIABLES),
            len(CHANGE_METRICS),
            latitudes.size,
            longitudes.size,
        ),
        dtype=np.float32,
    )
    for si, scenario in enumerate(arguments.scenarios):
        for mi, method in enumerate(METHODS):
            for pi, period in enumerate(FUTURE_PERIODS):
                for vi, variable in enumerate(ENERGY_VARIABLES):
                    for ci, metric in enumerate(CHANGE_METRICS):
                        arrays = change_storage[
                            (scenario, method, period, variable, metric)
                        ]
                        summary = ensemble_statistics(arrays, min_models)
                        change_values[si, mi, pi, vi, ci] = np.stack(
                            [summary[name] for name in ENSEMBLE_STATISTICS]
                        )
                        agreement_values[si, mi, pi, vi, ci] = (
                            sign_agreement_percent(arrays, min_models)
                        )
    change_dataset = xr.Dataset(
        {
            "change_value": (
                (
                    "scenario",
                    "method",
                    "future_period",
                    "energy_variable",
                    "change_metric",
                    "statistic",
                    "lat",
                    "lon",
                ),
                change_values,
            ),
            "sign_agreement_percent": (
                (
                    "scenario",
                    "method",
                    "future_period",
                    "energy_variable",
                    "change_metric",
                    "lat",
                    "lon",
                ),
                agreement_values,
            ),
            "gridcell_area_km2": (("lat", "lon"), china_gridcell),
            "china_intersects": (("lat", "lon"), china_intersects.astype(np.int8)),
            "china_intersection_area_km2": (("lat", "lon"), china_area),
            "china_area_fraction": (("lat", "lon"), china_fraction),
        },
        coords={
            key: coords[key]
            for key in (
                "scenario",
                "method",
                "future_period",
                "energy_variable",
                "change_metric",
                "statistic",
                "lat",
                "lon",
            )
        },
    )
    change_dataset.attrs.update(
        ensemble_rule="calculate each model change before equal-weight ensemble",
        reference_period="1994-2014",
        sign_agreement_definition=(
            "percentage of valid models whose change sign matches the "
            "equal-weight ensemble-mean sign"
        ),
        minimum_valid_models=int(min_models),
    )

    difference_values = np.full(
        (
            len(arguments.scenarios),
            len(PERIODS),
            len(ENERGY_VARIABLES),
            len(METHOD_DIFFERENCE_METRICS),
            len(ENSEMBLE_STATISTICS),
            latitudes.size,
            longitudes.size,
        ),
        np.nan,
        dtype=np.float32,
    )
    difference_agreement = np.full(
        (
            len(arguments.scenarios),
            len(PERIODS),
            len(ENERGY_VARIABLES),
            len(METHOD_DIFFERENCE_METRICS),
            latitudes.size,
            longitudes.size,
        ),
        np.nan,
        dtype=np.float32,
    )
    for si, scenario in enumerate(arguments.scenarios):
        for pi, period in enumerate(PERIODS):
            for vi, variable in enumerate(ENERGY_VARIABLES):
                for di, metric in enumerate(METHOD_DIFFERENCE_METRICS):
                    key = (scenario, period, variable, metric)
                    if key not in method_difference_storage:
                        continue
                    arrays = method_difference_storage[key]
                    summary = ensemble_statistics(arrays, min_models)
                    difference_values[si, pi, vi, di] = np.stack(
                        [summary[name] for name in ENSEMBLE_STATISTICS]
                    )
                    difference_agreement[si, pi, vi, di] = (
                        sign_agreement_percent(arrays, min_models)
                    )
    difference_dataset = xr.Dataset(
        {
            "optimized_minus_paper": (
                (
                    "scenario",
                    "period",
                    "energy_variable",
                    "method_difference_metric",
                    "statistic",
                    "lat",
                    "lon",
                ),
                difference_values,
            ),
            "sign_agreement_percent": (
                (
                    "scenario",
                    "period",
                    "energy_variable",
                    "method_difference_metric",
                    "lat",
                    "lon",
                ),
                difference_agreement,
            ),
            "gridcell_area_km2": (("lat", "lon"), china_gridcell),
            "china_intersects": (("lat", "lon"), china_intersects.astype(np.int8)),
            "china_intersection_area_km2": (("lat", "lon"), china_area),
            "china_area_fraction": (("lat", "lon"), china_fraction),
        },
        coords={
            key: coords[key]
            for key in (
                "scenario",
                "period",
                "energy_variable",
                "method_difference_metric",
                "statistic",
                "lat",
                "lon",
            )
        },
    )
    difference_dataset.attrs.update(
        difference_direction="optimized minus paper_qm",
        ensemble_rule="model-level method difference followed by equal-weight ensemble",
        minimum_valid_models=int(min_models),
    )

    complementarity_values = np.empty(
        (
            len(arguments.scenarios),
            len(METHODS),
            len(PERIODS),
            len(DEFINITIONS),
            len(ENSEMBLE_STATISTICS),
            latitudes.size,
            longitudes.size,
        ),
        dtype=np.float32,
    )
    for si, scenario in enumerate(arguments.scenarios):
        for mi, method in enumerate(METHODS):
            for pi, period in enumerate(PERIODS):
                for di, definition in enumerate(DEFINITIONS):
                    summary = ensemble_statistics(
                        complementarity_storage[
                            (scenario, method, period, definition)
                        ],
                        min_models,
                    )
                    complementarity_values[si, mi, pi, di] = np.stack(
                        [summary[name] for name in ENSEMBLE_STATISTICS]
                    )
    complementarity_dataset = xr.Dataset(
        {
            "spearman_rho": (
                (
                    "scenario",
                    "method",
                    "period",
                    "definition",
                    "statistic",
                    "lat",
                    "lon",
                ),
                complementarity_values,
            ),
            "gridcell_area_km2": (("lat", "lon"), china_gridcell),
            "china_intersects": (("lat", "lon"), china_intersects.astype(np.int8)),
            "china_intersection_area_km2": (("lat", "lon"), china_area),
            "china_area_fraction": (("lat", "lon"), china_fraction),
        },
        coords={
            key: coords[key]
            for key in (
                "scenario",
                "method",
                "period",
                "definition",
                "statistic",
                "lat",
                "lon",
            )
        },
    )
    complementarity_dataset.attrs.update(
        ensemble_rule="equal weight across regridded model-level Spearman-rho maps",
        primary_definitions="monthly_full and seasonal_full",
        sensitivity_definitions="monthly_climatology and seasonal_climatology",
        minimum_valid_models=int(min_models),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    period_path = output_root / "multimodel_energy_period_maps_1deg.nc"
    change_path = output_root / "multimodel_energy_change_maps_1deg.nc"
    difference_path = output_root / "multimodel_method_difference_maps_1deg.nc"
    complementarity_path = output_root / "multimodel_complementarity_maps_1deg.nc"
    write_dataset(period_path, period_dataset)
    write_dataset(change_path, change_dataset)
    write_dataset(difference_path, difference_dataset)
    write_dataset(complementarity_path, complementarity_dataset)

    scalar_path, scalar_difference_path, height_path = build_scalar_summaries(
        input_root, output_root
    )
    warning_exact_path, warning_family_path = summarize_transition_warnings(
        input_root, output_root
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "models": models,
        "model_count": len(models),
        "scenarios": list(arguments.scenarios),
        "methods": list(METHODS),
        "periods": {
            name: {"start_year": years[0], "end_year": years[1]}
            for name, years in PERIODS.items()
        },
        "common_grid": {
            "resolution_degree": float(arguments.resolution),
            "lat_min": float(latitudes.min()),
            "lat_max": float(latitudes.max()),
            "lon_min": float(longitudes.min()),
            "lon_max": float(longitudes.max()),
            "lat_count": int(latitudes.size),
            "lon_count": int(longitudes.size),
        },
        "china_intersection": china_metadata,
        "minimum_valid_models": int(min_models),
        "ensemble_order": (
            "model-level diagnostic -> linear interpolation to common grid -> "
            "equal-weight ensemble"
        ),
        "outputs": {
            "period_maps": str(period_path),
            "change_maps": str(change_path),
            "method_difference_maps": str(difference_path),
            "complementarity_maps": str(complementarity_path),
            "scalar_summary": str(scalar_path),
            "method_difference_scalar_summary": str(scalar_difference_path),
            "hub_height_sensitivity": str(height_path),
            "transition_unique_variable_events": str(warning_exact_path),
            "transition_unique_family_events": str(warning_family_path),
        },
    }
    manifest_path = output_root / "multimodel_ensemble_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 100)
    print("MULTI-MODEL ENSEMBLE COMPLETED")
    print("=" * 100)
    print(f"Models                    : {len(models)}")
    print(f"Common grid               : {latitudes.size} x {longitudes.size}")
    print(f"Minimum valid models      : {min_models}")
    print(f"Period maps               : {period_path}")
    print(f"Change maps               : {change_path}")
    print(f"Method-difference maps    : {difference_path}")
    print(f"Complementarity maps      : {complementarity_path}")
    print(f"Scalar summary            : {scalar_path}")
    print(f"Hub-height sensitivity    : {height_path}")
    print(f"Transition event summary  : {warning_family_path}")
    print(f"Manifest                  : {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
