#!/usr/bin/env python
r"""Build ensemble-first PV/WPD and complementarity products (routes A/C).

Place this script beside ``compute_energy_metrics_pilot_v2.py`` in::

    scripts/05_figure_data

This script complements, rather than replaces, ``build_multimodel_ensemble.py``.

Routes produced
---------------
A. paper_qm: monthly corrected meteorology ensemble -> PV/WPD -> diagnostics
C. optimized: monthly corrected meteorology ensemble -> PV/WPD -> diagnostics

The existing ``multimodel_ensemble`` archive contains routes B/D, in which
energy diagnostics are calculated per model before the model ensemble.
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
# -------------------------------------------------------------------

import argparse
import csv
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import cftime
    import xarray as xr
except ImportError as error:  # pragma: no cover
    raise RuntimeError(
        "cftime and xarray are required. Activate the pvwind conda environment."
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import compute_energy_metrics_pilot_v2 as engine
except ImportError as error:  # pragma: no cover
    raise RuntimeError(
        "Place compute_energy_metrics_pilot_v2.py beside this script."
    ) from error


SCENARIOS = ("ssp126", "ssp245", "ssp585")
METHODS = ("paper_qm", "optimized")
VARIABLES = ("tas", "rsds", "sfcWind")
ENERGY_VARIABLES = ("pvpot", "wpd")
PERIODS = {
    "historical": (1994, 2014),
    "mid_century": (2040, 2060),
    "late_century": (2080, 2100),
}
FUTURE_PERIODS = ("mid_century", "late_century")
DEFINITIONS = (
    "monthly_full",
    "monthly_climatology",
    "seasonal_full",
    "seasonal_climatology",
)
CHANGE_METRICS = ("absolute_change", "relative_change_percent")
EXPECTED_MODELS = 17
HISTORICAL_COUNT_WITH_SUPPORT = 253  # 1993-12 through 2014-12
FUTURE_COUNT = 1032  # 2015-01 through 2100-12
ENERGY_OUTPUT_COUNT = 1284  # 1994-01 through 2100-12


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Equal-weight corrected monthly meteorology first, then calculate "
            "PV/WPD and wind-solar complementarity."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Default: the centralized project root.",
    )
    parser.add_argument(
        "--cmip6-root",
        type=Path,
        help="Default: <project-root>/data/interim/cmip6_china_clipped",
    )
    parser.add_argument(
        "--era5-root",
        type=Path,
        help="Default: <project-root>/data/interim/era5_on_gcm_grid",
    )
    parser.add_argument(
        "--corrected-root",
        type=Path,
        help="Default: <project-root>/data/processed/bias_correction",
    )
    parser.add_argument(
        "--model-first-root",
        type=Path,
        help="Default: <project-root>/results/figure_data/multimodel_ensemble",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Default: <project-root>/data/processed/validation/ensemble_first_energy",
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
        help="Default: ceil(80%% of selected models), normally 14/17.",
    )
    parser.add_argument(
        "--hub-height",
        type=float,
        default=engine.MAIN_HUB_HEIGHT_M,
    )
    parser.add_argument("--lower", type=float, default=engine.DEFAULT_LOWER)
    parser.add_argument("--upper", type=float, default=engine.DEFAULT_UPPER)
    parser.add_argument(
        "--n-quantiles", type=int, default=engine.DEFAULT_N_QUANTILES
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of an existing completed A/C output archive.",
    )
    arguments = parser.parse_args()
    if arguments.resolution <= 0.0:
        parser.error("--resolution must be positive")
    if arguments.hub_height <= engine.REFERENCE_HEIGHT_M:
        parser.error("--hub-height must be greater than 10 m")
    if not (0.0 < arguments.lower < arguments.upper < 1.0):
        parser.error("Require 0 < --lower < --upper < 1")
    return arguments


def resolve_paths(arguments: argparse.Namespace) -> dict[str, Path]:
    project_root = (
        arguments.project_root.expanduser().resolve()
        if arguments.project_root
        else PROJECT_ROOT
    )
    paths = {
        "project_root": project_root,
        "cmip6_root": (
            arguments.cmip6_root.expanduser().resolve()
            if arguments.cmip6_root
            else get_path("data_interim_cmip6_china_clipped")
        ),
        "era5_root": (
            arguments.era5_root.expanduser().resolve()
            if arguments.era5_root
            else get_path("data_interim_era5_on_gcm_grid")
        ),
        "corrected_root": (
            arguments.corrected_root.expanduser().resolve()
            if arguments.corrected_root
            else get_path("data_processed_bias_correction")
        ),
        "model_first_root": (
            arguments.model_first_root.expanduser().resolve()
            if arguments.model_first_root
            else get_path("results_figure_data_multimodel_ensemble")
        ),
        "output_root": (
            arguments.output_root.expanduser().resolve()
            if arguments.output_root
            else get_path("data_processed_validation") / "ensemble_first_energy"
        ),
        "china_shapefile": (
            arguments.china_shapefile.expanduser().resolve()
            if arguments.china_shapefile
            else get_path("china_shapefile")
        ),
    }
    for name in (
        "cmip6_root",
        "era5_root",
        "corrected_root",
        "model_first_root",
    ):
        if not paths[name].is_dir():
            raise FileNotFoundError(f"{name} does not exist: {paths[name]}")
    if not paths["china_shapefile"].is_file():
        raise FileNotFoundError(
            f"China shapefile does not exist: {paths['china_shapefile']}"
        )
    return paths


def discover_models(
    corrected_root: Path,
    requested: list[str] | None,
    scenarios: list[str],
    expected: int,
) -> list[str]:
    if requested:
        models = sorted(dict.fromkeys(requested))
    else:
        models = sorted(
            path.name
            for path in corrected_root.iterdir()
            if path.is_dir()
            and all((path / scenario).is_dir() for scenario in scenarios)
        )
    if not models:
        raise FileNotFoundError(f"No models found below {corrected_root}")
    if requested is None and len(models) != expected:
        raise ValueError(
            f"Expected {expected} models; found {len(models)}: "
            + ", ".join(models)
        )
    return models


def coordinate(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    return start + step * np.arange(count + 1, dtype=np.float64)


def build_mask(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    shapefile: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    missing = [
        str(shapefile.with_suffix(suffix))
        for suffix in (".shx", ".dbf", ".prj")
        if not shapefile.with_suffix(suffix).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing shapefile sidecars: " + ", ".join(missing))
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import Point
        from shapely.ops import unary_union
    except ImportError as error:
        raise RuntimeError(
            "China mask requires cartopy and shapely in the pvwind environment."
        ) from error
    reader = shpreader.Reader(str(shapefile))
    geometry = unary_union(list(reader.geometries()))
    close = getattr(reader, "close", None)
    if close is not None:
        close()
    mask = np.zeros((latitudes.size, longitudes.size), dtype=bool)
    for i, latitude in enumerate(latitudes):
        for j, longitude in enumerate(longitudes):
            mask[i, j] = geometry.covers(
                Point(float(longitude), float(latitude))
            )
    if not mask.any():
        raise ValueError("China boundary selected zero common-grid cells")
    return mask, {
        "shapefile": str(shapefile),
        "selected_cells": int(mask.sum()),
        "total_cells": int(mask.size),
        "geometry_bounds": [float(value) for value in geometry.bounds],
        "selection_rule": "target-grid centre covered by China polygon",
    }


def monthly_dates(start_year: int, start_month: int, count: int) -> np.ndarray:
    dates = []
    year, month = start_year, start_month
    for _ in range(count):
        dates.append(cftime.DatetimeNoLeap(year, month, 1))
        month += 1
        if month == 13:
            month = 1
            year += 1
    return np.asarray(dates, dtype=object)


def validate_months(
    data: xr.DataArray | xr.Dataset,
    start_year: int,
    start_month: int,
    expected_count: int,
    label: str,
) -> None:
    years = np.asarray(data.time.dt.year.values, dtype=np.int64)
    months = np.asarray(data.time.dt.month.values, dtype=np.int64)
    expected = monthly_dates(start_year, start_month, expected_count)
    expected_keys = np.asarray(
        [value.year * 100 + value.month for value in expected], dtype=np.int64
    )
    keys = years * 100 + months
    if not np.array_equal(keys, expected_keys):
        raise ValueError(
            f"{label}: expected {expected_count} consecutive months from "
            f"{start_year:04d}-{start_month:02d}; found count={keys.size}, "
            f"first={keys[0] if keys.size else None}, "
            f"last={keys[-1] if keys.size else None}"
        )


def regrid_monthly(
    data: xr.DataArray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    china_mask: np.ndarray,
) -> np.ndarray:
    array = data.transpose("time", "lat", "lon").sortby("lat").sortby("lon")
    target = array.interp(
        lat=xr.DataArray(latitudes, dims="lat"),
        lon=xr.DataArray(longitudes, dims="lon"),
        method="linear",
    )
    values = np.asarray(target.values, dtype=np.float64)
    values[:, ~china_mask] = np.nan
    return values


def new_accumulator(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    return {
        "sum": np.zeros(shape, dtype=np.float64),
        "count": np.zeros(shape, dtype=np.uint16),
    }


def accumulate(accumulator: dict[str, np.ndarray], values: np.ndarray) -> None:
    if values.shape != accumulator["sum"].shape:
        raise ValueError(
            f"Accumulator shape {accumulator['sum'].shape} != values {values.shape}"
        )
    finite = np.isfinite(values)
    accumulator["sum"] += np.where(finite, values, 0.0)
    accumulator["count"] += finite.astype(np.uint16)


def finalize_accumulator(
    accumulator: dict[str, np.ndarray], min_models: int
) -> np.ndarray:
    return np.divide(
        accumulator["sum"],
        accumulator["count"],
        out=np.full(accumulator["sum"].shape, np.nan, dtype=np.float64),
        where=accumulator["count"] >= min_models,
    ).astype(np.float32)


def method_branch(method: str, variable: str) -> str:
    if method == "paper_qm":
        return "qm_paper"
    if method == "optimized":
        return "qdm_improved" if variable in ("tas", "sfcWind") else "qm_paper"
    raise ValueError(f"Unknown method: {method}")


def meteorology_dataset(
    arrays: dict[str, np.ndarray],
    times: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> xr.Dataset:
    dataset = xr.Dataset(
        {
            variable: (("time", "lat", "lon"), arrays[variable])
            for variable in VARIABLES
        },
        coords={"time": times, "lat": latitudes, "lon": longitudes},
    )
    dataset["tas"].attrs["units"] = "K"
    dataset["rsds"].attrs["units"] = "W m-2"
    dataset["sfcWind"].attrs["units"] = "m s-1"
    return dataset


def select_years(dataset: xr.Dataset, start: int, end: int) -> xr.Dataset:
    years = np.asarray(dataset.time.dt.year.values, dtype=np.int64)
    return dataset.isel(time=np.flatnonzero((years >= start) & (years <= end)))


def safe_relative(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
    return xr.where(
        np.isfinite(denominator) & (np.abs(denominator) > 1.0e-12),
        100.0 * numerator / denominator,
        np.nan,
    )


def area_weighted_mean(data: xr.DataArray, mask: np.ndarray) -> float:
    values = np.asarray(data.values, dtype=np.float64)
    weights = np.cos(np.deg2rad(np.asarray(data.lat.values)))[:, None] * mask
    valid = np.isfinite(values) & mask
    denominator = np.where(valid, weights, 0.0).sum()
    return (
        float(np.where(valid, values * weights, 0.0).sum() / denominator)
        if denominator > 0.0
        else math.nan
    )


def write_dataset(path: Path, dataset: xr.Dataset, time_chunks: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {}
    for name, variable in dataset.data_vars.items():
        options: dict[str, object] = {"zlib": True, "complevel": 3}
        if np.issubdtype(variable.dtype, np.floating):
            options["dtype"] = "float32"
        if time_chunks and variable.dims == ("time", "lat", "lon"):
            options["chunksizes"] = (
                min(12, dataset.sizes["time"]),
                dataset.sizes["lat"],
                dataset.sizes["lon"],
            )
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


def build_historical_ensemble(
    paths: dict[str, Path],
    models: list[str],
    first_scenario: str,
    arguments: argparse.Namespace,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    mask: np.ndarray,
    min_models: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, str]]]:
    shape = (HISTORICAL_COUNT_WITH_SUPPORT, latitudes.size, longitudes.size)
    accumulators = {variable: new_accumulator(shape) for variable in VARIABLES}
    input_files: dict[str, dict[str, str]] = {}
    print("\nBuilding the shared 17-model historical-QM meteorology ensemble ...")
    for index, model in enumerate(models, start=1):
        print(f"  [{index:02d}/{len(models):02d}] {model}")
        discovered = engine.discover_inputs(
            paths["cmip6_root"],
            paths["era5_root"],
            paths["corrected_root"],
            model,
            first_scenario,
        )
        baselines, _ = engine.load_inputs(
            discovered,
            arguments.lower,
            arguments.upper,
            arguments.n_quantiles,
        )
        input_files[model] = {
            f"historical_{variable}": str(discovered["historical"][variable])
            for variable in VARIABLES
        }
        input_files[model]["era5"] = str(discovered["era5"]["tas"])
        for variable in VARIABLES:
            data = baselines["qm_historical"][variable]
            validate_months(
                data, 1993, 12, HISTORICAL_COUNT_WITH_SUPPORT,
                f"{model} historical QM {variable}",
            )
            accumulate(
                accumulators[variable],
                regrid_monthly(data, latitudes, longitudes, mask),
            )
    return (
        {
            variable: finalize_accumulator(accumulators[variable], min_models)
            for variable in VARIABLES
        },
        input_files,
    )


def build_future_ensembles(
    paths: dict[str, Path],
    models: list[str],
    scenario: str,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    mask: np.ndarray,
    min_models: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, str]]]:
    shape = (FUTURE_COUNT, latitudes.size, longitudes.size)
    accumulators = {
        method: {variable: new_accumulator(shape) for variable in VARIABLES}
        for method in METHODS
    }
    files: dict[str, dict[str, str]] = {}
    print(f"\nBuilding monthly meteorology ensembles for {scenario} ...")
    for index, model in enumerate(models, start=1):
        print(f"  [{index:02d}/{len(models):02d}] {model}")
        discovered = engine.discover_inputs(
            paths["cmip6_root"],
            paths["era5_root"],
            paths["corrected_root"],
            model,
            scenario,
        )
        files[model] = {
            variable: str(discovered["corrected"][variable])
            for variable in VARIABLES
        }
        for variable in VARIABLES:
            corrected = engine.extract_corrected(
                discovered["corrected"][variable], variable
            )
            validate_months(
                corrected, 2015, 1, FUTURE_COUNT,
                f"{model}/{scenario} corrected {variable}",
            )
            for method in METHODS:
                branch = method_branch(method, variable)
                accumulate(
                    accumulators[method][variable],
                    regrid_monthly(
                        corrected[branch], latitudes, longitudes, mask
                    ),
                )
    return (
        {
            method: {
                variable: finalize_accumulator(
                    accumulators[method][variable], min_models
                )
                for variable in VARIABLES
            }
            for method in METHODS
        },
        files,
    )


def compare_with_model_first(
    paths: dict[str, Path],
    period_dataset: xr.Dataset,
    change_dataset: xr.Dataset,
    complementarity_dataset: xr.Dataset,
    output_root: Path,
) -> tuple[Path, Path]:
    model_first_period_path = (
        paths["model_first_root"] / "multimodel_energy_period_maps_1deg.nc"
    )
    model_first_change_path = (
        paths["model_first_root"] / "multimodel_energy_change_maps_1deg.nc"
    )
    model_first_comp_path = (
        paths["model_first_root"] / "multimodel_complementarity_maps_1deg.nc"
    )
    with xr.open_dataset(model_first_period_path) as source:
        model_first_period = source["period_value"].sel(statistic="mean").load()
    with xr.open_dataset(model_first_change_path) as source:
        model_first_change = source["change_value"].sel(statistic="mean").load()
    with xr.open_dataset(model_first_comp_path) as source:
        model_first_comp = source["spearman_rho"].sel(statistic="mean").load()

    period_difference = (
        period_dataset["period_value"] - model_first_period
    ).rename("ensemble_first_minus_model_first_period")
    change_difference = (
        change_dataset["change_value"] - model_first_change
    ).rename("ensemble_first_minus_model_first_change")
    comp_difference = (
        complementarity_dataset["spearman_rho"] - model_first_comp
    ).rename("ensemble_first_minus_model_first_rho")
    comparison = xr.Dataset(
        {
            period_difference.name: period_difference,
            change_difference.name: change_difference,
            comp_difference.name: comp_difference,
            "china_mask": period_dataset["china_mask"],
        }
    )
    comparison.attrs.update(
        difference_direction="ensemble-first minus model-first",
        ensemble_first=(
            "equal-weight monthly meteorology ensemble before PV/WPD/Spearman"
        ),
        model_first=(
            "PV/WPD/Spearman per model before equal-weight diagnostic ensemble"
        ),
    )
    comparison_path = output_root / "ensemble_order_difference_maps_1deg.nc"
    write_dataset(comparison_path, comparison)

    rows = []
    for scenario in period_dataset.scenario.values.tolist():
        for method in period_dataset.method.values.tolist():
            for period in period_dataset.period.values.tolist():
                for variable in period_dataset.energy_variable.values.tolist():
                    data = period_difference.sel(
                        {
                            "scenario": scenario,
                            "method": method,
                            "period": period,
                            "energy_variable": variable,
                        }
                    )
                    rows.append(
                        {
                            "diagnostic": "period_value",
                            "scenario": str(scenario),
                            "method": str(method),
                            "period": str(period),
                            "energy_variable": str(variable),
                            "definition": "",
                            "change_metric": "",
                            "area_weighted_mean_difference": area_weighted_mean(
                                data, np.asarray(period_dataset["china_mask"].values, dtype=bool)
                            ),
                        }
                    )
            for period in change_dataset.future_period.values.tolist():
                for variable in change_dataset.energy_variable.values.tolist():
                    for metric in change_dataset.change_metric.values.tolist():
                        data = change_difference.sel(
                            {
                                "scenario": scenario,
                                "method": method,
                                "future_period": period,
                                "energy_variable": variable,
                                "change_metric": metric,
                            }
                        )
                        rows.append(
                            {
                                "diagnostic": "change_value",
                                "scenario": str(scenario),
                                "method": str(method),
                                "period": str(period),
                                "energy_variable": str(variable),
                                "definition": "",
                                "change_metric": str(metric),
                                "area_weighted_mean_difference": area_weighted_mean(
                                    data,
                                    np.asarray(
                                        period_dataset["china_mask"].values,
                                        dtype=bool,
                                    ),
                                ),
                            }
                        )
            for period in complementarity_dataset.period.values.tolist():
                for definition in complementarity_dataset.definition.values.tolist():
                    data = comp_difference.sel(
                        {
                            "scenario": scenario,
                            "method": method,
                            "period": period,
                            "definition": definition,
                        }
                    )
                    rows.append(
                        {
                            "diagnostic": "spearman_rho",
                            "scenario": str(scenario),
                            "method": str(method),
                            "period": str(period),
                            "energy_variable": "pvpot_wpd",
                            "definition": str(definition),
                            "change_metric": "",
                            "area_weighted_mean_difference": area_weighted_mean(
                                data,
                                np.asarray(
                                    period_dataset["china_mask"].values,
                                    dtype=bool,
                                ),
                            ),
                        }
                    )
    summary_path = output_root / "ensemble_order_difference_summary.csv"
    write_csv(summary_path, rows)
    return comparison_path, summary_path


def resume_completed_core_outputs(
    paths: dict[str, Path],
    models: list[str],
    arguments: argparse.Namespace,
    min_models: int,
    mask_metadata: dict[str, object],
) -> bool:
    """Finish only the order comparison after the original final-step error."""
    output_root = paths["output_root"]
    period_path = output_root / "ensemble_first_energy_period_maps_1deg.nc"
    change_path = output_root / "ensemble_first_energy_change_maps_1deg.nc"
    comp_path = output_root / "ensemble_first_complementarity_maps_1deg.nc"
    scalar_path = output_root / "ensemble_first_scalar_summary.csv"
    monthly_paths = {
        scenario: {
            method: output_root
            / (
                f"ensemble_first_energy_monthly_{scenario}_{method}_"
                "199401-210012.nc"
            )
            for method in METHODS
        }
        for scenario in arguments.scenarios
    }
    required = [period_path, change_path, comp_path, scalar_path]
    required.extend(
        monthly_paths[scenario][method]
        for scenario in arguments.scenarios
        for method in METHODS
    )
    if not all(path.is_file() for path in required):
        return False

    print("\nDetected complete A/C core outputs from the interrupted run.")
    print("Resuming only the A/B and C/D ensemble-order comparison; no climate data will be recomputed.")
    with xr.open_dataset(period_path) as source:
        period_dataset = source.load()
    with xr.open_dataset(change_path) as source:
        change_dataset = source.load()
    with xr.open_dataset(comp_path) as source:
        complementarity_dataset = source.load()

    expected_scenarios = set(arguments.scenarios)
    if set(str(value) for value in period_dataset.scenario.values.tolist()) != expected_scenarios:
        raise ValueError("Existing A/C period file scenarios do not match this run")
    if set(str(value) for value in period_dataset.method.values.tolist()) != set(METHODS):
        raise ValueError("Existing A/C period file methods do not match this run")

    order_difference_path, order_summary_path = compare_with_model_first(
        paths,
        period_dataset,
        change_dataset,
        complementarity_dataset,
        output_root,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "resumed_from_completed_core_outputs": True,
        "resumption_reason": (
            "The original run completed A/C climate and energy products but "
            "stopped during the final xarray coordinate selection."
        ),
        "routes": {
            "A": "paper_qm meteorology ensemble before energy calculation",
            "C": "optimized meteorology ensemble before energy calculation",
        },
        "models": models,
        "model_count": len(models),
        "scenarios": list(arguments.scenarios),
        "minimum_valid_models": int(min_models),
        "china_mask": mask_metadata,
        "common_calendar": "noleap",
        "outputs": {
            "monthly_energy": {
                scenario: {
                    method: str(monthly_paths[scenario][method])
                    for method in METHODS
                }
                for scenario in arguments.scenarios
            },
            "period_maps": str(period_path),
            "change_maps": str(change_path),
            "complementarity_maps": str(comp_path),
            "scalar_summary": str(scalar_path),
            "ensemble_order_difference_maps": str(order_difference_path),
            "ensemble_order_difference_summary": str(order_summary_path),
        },
    }
    manifest_path = output_root / "ensemble_first_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("=" * 100)
    print("ENSEMBLE-FIRST ENERGY PRODUCTION RESUMED AND COMPLETED")
    print("=" * 100)
    print("Climate/energy recomputation : skipped")
    print(f"Order-difference maps        : {order_difference_path}")
    print(f"Order-difference summary     : {order_summary_path}")
    print(f"Manifest                     : {manifest_path}")
    return True


def main() -> int:
    arguments = parse_arguments()
    paths = resolve_paths(arguments)
    models = discover_models(
        paths["corrected_root"],
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
        raise ValueError(f"--min-models must be between 1 and {len(models)}")
    latitudes = coordinate(
        arguments.lat_min, arguments.lat_max, arguments.resolution
    )
    longitudes = coordinate(
        arguments.lon_min, arguments.lon_max, arguments.resolution
    )
    china_mask, mask_metadata = build_mask(
        latitudes, longitudes, paths["china_shapefile"]
    )
    comparison_files = (
        paths["model_first_root"] / "multimodel_energy_period_maps_1deg.nc",
        paths["model_first_root"] / "multimodel_energy_change_maps_1deg.nc",
        paths["model_first_root"] / "multimodel_complementarity_maps_1deg.nc",
    )
    missing_comparison = [path for path in comparison_files if not path.is_file()]
    if missing_comparison:
        raise FileNotFoundError(
            "Missing model-first ensemble files: "
            + ", ".join(str(path) for path in missing_comparison)
        )
    manifest_path = paths["output_root"] / "ensemble_first_manifest.json"
    if manifest_path.is_file() and not arguments.force and not arguments.dry_run:
        raise FileExistsError(
            f"Completed output appears to exist: {manifest_path}. "
            "Use --force only if you intentionally want to replace it."
        )

    print("=" * 100)
    print("ENSEMBLE-FIRST ENERGY PRODUCTION (ROUTES A/C)")
    print("=" * 100)
    print(f"Models          : {len(models)}")
    print(f"Scenarios       : {', '.join(arguments.scenarios)}")
    print(f"Common grid     : {latitudes.size} lat x {longitudes.size} lon")
    print(f"Resolution      : {arguments.resolution:g} degree")
    print(f"China cells     : {int(china_mask.sum())}/{china_mask.size}")
    print(f"Minimum models  : {min_models}/{len(models)}")
    print("Route A         : paper QM meteorology ensemble -> energy")
    print("Route C         : optimized meteorology ensemble -> energy")
    print(f"Output root     : {paths['output_root']}")
    if arguments.dry_run:
        validated = 0
        for model in models:
            for scenario in arguments.scenarios:
                engine.discover_inputs(
                    paths["cmip6_root"],
                    paths["era5_root"],
                    paths["corrected_root"],
                    model,
                    scenario,
                )
                validated += 1
        print(
            "DRY RUN COMPLETED: roots, all model/scenario input paths, common "
            "grid, and the B/D comparison archive are valid."
        )
        print(f"Validated combinations: {validated}/{len(models) * len(arguments.scenarios)}")
        print("No output files were written.")
        return 0

    if not arguments.force and resume_completed_core_outputs(
        paths,
        models,
        arguments,
        min_models,
        mask_metadata,
    ):
        return 0

    output_root = paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        historical_arrays, historical_input_files = build_historical_ensemble(
            paths,
            models,
            arguments.scenarios[0],
            arguments,
            latitudes,
            longitudes,
            china_mask,
            min_models,
        )
        historical_time = monthly_dates(
            1993, 12, HISTORICAL_COUNT_WITH_SUPPORT
        )
        future_time = monthly_dates(2015, 1, FUTURE_COUNT)

        period_maps: dict[tuple[str, ...], np.ndarray] = {}
        change_maps: dict[tuple[str, ...], np.ndarray] = {}
        complementarity_maps: dict[tuple[str, ...], np.ndarray] = {}
        scalar_rows: list[dict[str, object]] = []
        monthly_outputs: dict[str, dict[str, str]] = {}
        future_input_files: dict[str, object] = {}

        for scenario in arguments.scenarios:
            future_arrays, scenario_files = build_future_ensembles(
                paths,
                models,
                scenario,
                latitudes,
                longitudes,
                china_mask,
                min_models,
            )
            future_input_files[scenario] = scenario_files
            monthly_outputs[scenario] = {}
            for method in METHODS:
                historical = meteorology_dataset(
                    historical_arrays,
                    historical_time,
                    latitudes,
                    longitudes,
                )
                future = meteorology_dataset(
                    future_arrays[method],
                    future_time,
                    latitudes,
                    longitudes,
                )
                meteorology = xr.concat([historical, future], dim="time")
                validate_months(
                    meteorology,
                    1993,
                    12,
                    HISTORICAL_COUNT_WITH_SUPPORT + FUTURE_COUNT,
                    f"{scenario}/{method} ensemble meteorology",
                )
                energy_support = engine.compute_energy(
                    meteorology, arguments.hub_height
                )
                energy = select_years(energy_support, 1994, 2100)
                if energy.sizes["time"] != ENERGY_OUTPUT_COUNT:
                    raise ValueError(
                        f"{scenario}/{method}: expected {ENERGY_OUTPUT_COUNT} "
                        f"energy months; found {energy.sizes['time']}"
                    )
                energy.attrs.update(
                    scenario=scenario,
                    method=method,
                    ensemble_order="monthly meteorology ensemble before energy calculation",
                    model_count=len(models),
                    minimum_valid_models=min_models,
                    output_period="1994-2100",
                    common_grid_resolution_degree=float(arguments.resolution),
                )
                energy_path = output_root / (
                    f"ensemble_first_energy_monthly_{scenario}_{method}_"
                    "199401-210012.nc"
                )
                write_dataset(energy_path, energy, time_chunks=True)
                monthly_outputs[scenario][method] = str(energy_path)

                method_period_maps: dict[str, dict[str, xr.DataArray]] = {}
                for period, (start, end) in PERIODS.items():
                    subset = select_years(energy, start, end)
                    if subset.sizes["time"] != 252:
                        raise ValueError(
                            f"{scenario}/{method}/{period}: expected 252 months"
                        )
                    method_period_maps[period] = {}
                    for variable in ENERGY_VARIABLES:
                        mean_map = subset[variable].mean("time", skipna=True)
                        method_period_maps[period][variable] = mean_map
                        period_maps[(scenario, method, period, variable)] = (
                            np.asarray(mean_map.values, dtype=np.float32)
                        )
                        scalar_rows.append(
                            {
                                "scenario": scenario,
                                "method": method,
                                "period": period,
                                "variable": variable,
                                "metric": "period_value",
                                "unit": str(mean_map.attrs.get("units", "")),
                                "area_weighted_china_mean": area_weighted_mean(
                                    mean_map, china_mask
                                ),
                            }
                        )
                for future_period in FUTURE_PERIODS:
                    for variable in ENERGY_VARIABLES:
                        baseline = method_period_maps["historical"][variable]
                        future_map = method_period_maps[future_period][variable]
                        absolute = future_map - baseline
                        relative = safe_relative(absolute, baseline)
                        for metric, data in (
                            ("absolute_change", absolute),
                            ("relative_change_percent", relative),
                        ):
                            change_maps[
                                (scenario, method, future_period, variable, metric)
                            ] = np.asarray(data.values, dtype=np.float32)
                            scalar_rows.append(
                                {
                                    "scenario": scenario,
                                    "method": method,
                                    "period": future_period,
                                    "variable": variable,
                                    "metric": metric,
                                    "unit": (
                                        str(future_map.attrs.get("units", ""))
                                        if metric == "absolute_change"
                                        else "%"
                                    ),
                                    "area_weighted_china_mean": area_weighted_mean(
                                        data, china_mask
                                    ),
                                }
                            )

                complementarity, _ = engine.compute_complementarity(
                    method,
                    energy_support,
                    china_mask,
                    "china_mask_1deg",
                )
                for period in PERIODS:
                    for definition in DEFINITIONS:
                        complementarity_maps[
                            (scenario, method, period, definition)
                        ] = np.asarray(
                            complementarity["spearman_rho"]
                            .sel(period=period, definition=definition)
                            .values,
                            dtype=np.float32,
                        )

        period_values = np.stack(
            [
                period_maps[(scenario, method, period, variable)]
                for scenario in arguments.scenarios
                for method in METHODS
                for period in PERIODS
                for variable in ENERGY_VARIABLES
            ]
        ).reshape(
            len(arguments.scenarios),
            len(METHODS),
            len(PERIODS),
            len(ENERGY_VARIABLES),
            latitudes.size,
            longitudes.size,
        )
        period_dataset = xr.Dataset(
            {
                "period_value": (
                    (
                        "scenario",
                        "method",
                        "period",
                        "energy_variable",
                        "lat",
                        "lon",
                    ),
                    period_values,
                ),
                "china_mask": (("lat", "lon"), china_mask.astype(np.int8)),
            },
            coords={
                "scenario": list(arguments.scenarios),
                "method": list(METHODS),
                "period": list(PERIODS),
                "energy_variable": list(ENERGY_VARIABLES),
                "lat": latitudes,
                "lon": longitudes,
            },
        )
        period_dataset.attrs["ensemble_order"] = (
            "monthly meteorology ensemble before energy and period calculation"
        )

        change_values = np.stack(
            [
                change_maps[(scenario, method, period, variable, metric)]
                for scenario in arguments.scenarios
                for method in METHODS
                for period in FUTURE_PERIODS
                for variable in ENERGY_VARIABLES
                for metric in CHANGE_METRICS
            ]
        ).reshape(
            len(arguments.scenarios),
            len(METHODS),
            len(FUTURE_PERIODS),
            len(ENERGY_VARIABLES),
            len(CHANGE_METRICS),
            latitudes.size,
            longitudes.size,
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
                        "lat",
                        "lon",
                    ),
                    change_values,
                ),
                "china_mask": (("lat", "lon"), china_mask.astype(np.int8)),
            },
            coords={
                "scenario": list(arguments.scenarios),
                "method": list(METHODS),
                "future_period": list(FUTURE_PERIODS),
                "energy_variable": list(ENERGY_VARIABLES),
                "change_metric": list(CHANGE_METRICS),
                "lat": latitudes,
                "lon": longitudes,
            },
        )
        change_dataset.attrs.update(
            ensemble_order="monthly meteorology ensemble before energy calculation",
            reference_period="1994-2014",
        )

        comp_values = np.stack(
            [
                complementarity_maps[(scenario, method, period, definition)]
                for scenario in arguments.scenarios
                for method in METHODS
                for period in PERIODS
                for definition in DEFINITIONS
            ]
        ).reshape(
            len(arguments.scenarios),
            len(METHODS),
            len(PERIODS),
            len(DEFINITIONS),
            latitudes.size,
            longitudes.size,
        )
        complementarity_dataset = xr.Dataset(
            {
                "spearman_rho": (
                    (
                        "scenario",
                        "method",
                        "period",
                        "definition",
                        "lat",
                        "lon",
                    ),
                    comp_values,
                ),
                "china_mask": (("lat", "lon"), china_mask.astype(np.int8)),
            },
            coords={
                "scenario": list(arguments.scenarios),
                "method": list(METHODS),
                "period": list(PERIODS),
                "definition": list(DEFINITIONS),
                "lat": latitudes,
                "lon": longitudes,
            },
        )
        complementarity_dataset.attrs.update(
            ensemble_order=(
                "monthly meteorology ensemble before PV/WPD and Spearman"
            ),
            djf_rule="December belongs to the following season-year",
            common_calendar="noleap",
            seasonal_weighting="days in month",
        )

        period_path = output_root / "ensemble_first_energy_period_maps_1deg.nc"
        change_path = output_root / "ensemble_first_energy_change_maps_1deg.nc"
        comp_path = output_root / "ensemble_first_complementarity_maps_1deg.nc"
        scalar_path = output_root / "ensemble_first_scalar_summary.csv"
        write_dataset(period_path, period_dataset)
        write_dataset(change_path, change_dataset)
        write_dataset(comp_path, complementarity_dataset)
        write_csv(scalar_path, scalar_rows)

        order_difference_path, order_summary_path = compare_with_model_first(
            paths,
            period_dataset,
            change_dataset,
            complementarity_dataset,
            output_root,
        )

        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "routes": {
                "A": "paper_qm meteorology ensemble before energy calculation",
                "C": "optimized meteorology ensemble before energy calculation",
            },
            "models": models,
            "model_count": len(models),
            "scenarios": list(arguments.scenarios),
            "minimum_valid_models": min_models,
            "common_grid": {
                "resolution_degree": float(arguments.resolution),
                "lat_count": int(latitudes.size),
                "lon_count": int(longitudes.size),
                "lat_min": float(latitudes.min()),
                "lat_max": float(latitudes.max()),
                "lon_min": float(longitudes.min()),
                "lon_max": float(longitudes.max()),
            },
            "china_mask": mask_metadata,
            "historical": {
                "source": "monthly multiplicative-QM history per model",
                "fit_period": "1959-2014",
                "ensemble_support_period": "1993-12 through 2014-12",
                "common_to_methods_and_scenarios": True,
                "input_files": historical_input_files,
            },
            "future_input_files": future_input_files,
            "common_calendar": "noleap",
            "outputs": {
                "monthly_energy": monthly_outputs,
                "period_maps": str(period_path),
                "change_maps": str(change_path),
                "complementarity_maps": str(comp_path),
                "scalar_summary": str(scalar_path),
                "ensemble_order_difference_maps": str(order_difference_path),
                "ensemble_order_difference_summary": str(order_summary_path),
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print("=" * 100)
    print("ENSEMBLE-FIRST ENERGY PRODUCTION COMPLETED")
    print("=" * 100)
    print(f"Models                    : {len(models)}")
    print(f"Scenarios                 : {len(arguments.scenarios)}")
    print(f"Routes                    : A paper_qm / C optimized")
    print(f"Period maps               : {period_path}")
    print(f"Change maps               : {change_path}")
    print(f"Complementarity maps      : {comp_path}")
    print(f"Scalar summary            : {scalar_path}")
    print(f"Order-difference maps     : {order_difference_path}")
    print(f"Order-difference summary  : {order_summary_path}")
    print(f"Manifest                  : {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
