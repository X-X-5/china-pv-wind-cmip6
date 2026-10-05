#!/usr/bin/env python
r"""Build the final, figure-ready B/D analysis products.

Place this script in::

    scripts/05_figure_data

The script does not rerun bias correction or energy calculations.  It reads
the completed per-model energy archive and the model-first ensemble maps.

Final routes
------------
* B / paper_qm: paper-compatible multiplicative QM, model-level energy first.
* D / optimized: optimized tas/sfcWind correction, model-level energy first.

The ensemble-first A/C products are deliberately not used.
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
from area_weights import area_weighted_mean, china_intersection_weights  # noqa: E402
# -------------------------------------------------------------------

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError as error:  # pragma: no cover
    raise RuntimeError("xarray is required. Activate the pvwind environment.") from error


SCENARIOS = ("ssp126", "ssp245", "ssp585")
METHODS = ("paper_qm", "optimized")
ROUTES = {"paper_qm": "B", "optimized": "D"}
PERIODS = {
    "historical": (1994, 2014),
    "mid_century": (2040, 2060),
    "late_century": (2080, 2100),
}
FUTURE_PERIODS = ("mid_century", "late_century")
TIME_SERIES_VARIABLES = ("tas_c", "rsds", "sfcWind_10m", "pvpot", "wpd")
ENERGY_VARIABLES = ("pvpot", "wpd")
PRIMARY_COMPLEMENTARITY = ("seasonal_full", "monthly_full")
ENSEMBLE_STATS = ("mean", "minimum", "maximum", "p10", "p90", "std")
EXPECTED_MODELS = 17


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final paper-baseline (B) and optimized (D) products."
    )
    parser.add_argument(
        "--energy-root",
        type=Path,
        help="Default: <project-root>/data/processed/energy_metrics",
    )
    parser.add_argument(
        "--ensemble-root",
        type=Path,
        help="Default: <project-root>/results/figure_data/multimodel_ensemble",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Default: <project-root>/results/figure_data/final_analysis",
    )
    parser.add_argument(
        "--china-shapefile",
        type=Path,
        help="Default: <project-root>/data/boundaries/china_qc_boundary.shp",
    )
    parser.add_argument("--models", nargs="+", help="Optional model subset.")
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS)
    )
    parser.add_argument("--expected-models", type=int, default=EXPECTED_MODELS)
    parser.add_argument(
        "--reuse-annual",
        action="store_true",
        help=(
            "Reuse the two existing annual CSV outputs from an interrupted run "
            "and continue with maps/summaries."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def decoder():
    try:
        return xr.coders.CFDatetimeCoder(use_cftime=True)
    except AttributeError:  # pragma: no cover
        return True


def open_netcdf(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=decoder())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def discover_models(
    energy_root: Path,
    requested: list[str] | None,
    scenarios: list[str],
    expected: int,
) -> list[str]:
    if requested:
        models = sorted(dict.fromkeys(requested))
    else:
        models = sorted(
            path.name
            for path in energy_root.iterdir()
            if path.is_dir()
            and path.name != "production_audit"
            and all((path / scenario).is_dir() for scenario in scenarios)
        )
    if not models:
        raise FileNotFoundError(f"No model directories found below {energy_root}")
    if requested is None and len(models) != expected:
        raise ValueError(
            f"Expected {expected} models; found {len(models)}: " + ", ".join(models)
        )
    return models


def energy_path(root: Path, model: str, scenario: str, method: str) -> Path:
    return root / model / scenario / (
        f"energy_monthly_{model}_{scenario}_{method}_199401-210012.nc"
    )


def required_paths(
    energy_root: Path,
    ensemble_root: Path,
    models: list[str],
    scenarios: list[str],
) -> list[Path]:
    paths = [
        energy_path(energy_root, model, scenario, method)
        for model in models
        for scenario in scenarios
        for method in METHODS
    ]
    paths.extend(
        [
            ensemble_root / "multimodel_energy_period_maps_1deg.nc",
            ensemble_root / "multimodel_energy_change_maps_1deg.nc",
            ensemble_root / "multimodel_method_difference_maps_1deg.nc",
            ensemble_root / "multimodel_complementarity_maps_1deg.nc",
        ]
    )
    return paths


def build_china_weights(lat: np.ndarray, lon: np.ndarray, shapefile: Path) -> np.ndarray:
    """Per-cell China-overlap areas (km²) on a native model grid."""
    result = china_intersection_weights(lat, lon, shapefile)
    weights = result["china_intersection_area_km2"]
    if not (weights > 0.0).any():
        raise ValueError("China shapefile selected zero grid cells")
    return weights


def spatial_weighted_mean(data: xr.DataArray, weights: np.ndarray) -> xr.DataArray:
    """China-intersection-area-weighted national mean per time step."""
    array = data.transpose(..., "lat", "lon")
    values = np.asarray(array.values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected (time, lat, lon), got {values.shape}")
    means = area_weighted_mean(values, weights)
    return xr.DataArray(
        means,
        coords={"time": array.time},
        dims="time",
        name=data.name,
    )


def native_period_means(
    path: Path,
    weights_cache: dict[tuple[bytes, bytes], np.ndarray],
    shapefile: Path,
) -> dict[str, dict[str, float]]:
    """Native-grid intersection-weighted national mean per period for one model.

    Reads a single scenario energy file (1994-2100, 1284 months) and returns
    ``{period_name: {variable: national_mean}}`` for every period in ``PERIODS``.
    The 1994-2014 slice is the shared QM historical part, so the historical mean
    is identical for both methods; the future periods come from 2040-2060 and
    2080-2100.
    """
    with open_netcdf(path) as dataset:
        missing = [name for name in TIME_SERIES_VARIABLES if name not in dataset]
        if missing:
            raise KeyError(f"{path}: missing variables {missing}")
        years = np.asarray(dataset.time.dt.year.values, dtype=np.int64)
        lat = np.asarray(dataset.lat.values, dtype=np.float64)
        lon = np.asarray(dataset.lon.values, dtype=np.float64)
        key = (lat.tobytes(), lon.tobytes())
        if key not in weights_cache:
            weights_cache[key] = build_china_weights(lat, lon, shapefile)
        weights = weights_cache[key]
        result: dict[str, dict[str, float]] = {}
        for period_name, (start, end) in PERIODS.items():
            indices = np.flatnonzero((years >= start) & (years <= end))
            if indices.size == 0:
                raise ValueError(f"{path}: no months in {period_name} {start}-{end}")
            subset = dataset[list(TIME_SERIES_VARIABLES)].isel(time=indices)
            result[period_name] = {}
            for variable in TIME_SERIES_VARIABLES:
                monthly = spatial_weighted_mean(subset[variable], weights)
                result[period_name][variable] = float(
                    np.nanmean(np.asarray(monthly.values, dtype=np.float64))
                )
        return result


def annual_native_series(path: Path, weights_cache: dict[tuple[bytes, bytes], np.ndarray], shapefile: Path) -> dict[str, np.ndarray]:
    with open_netcdf(path) as dataset:
        missing = [name for name in TIME_SERIES_VARIABLES if name not in dataset]
        if missing:
            raise KeyError(f"{path}: missing variables {missing}")
        years = np.asarray(dataset.time.dt.year.values, dtype=np.int64)
        indices = np.flatnonzero((years >= 2015) & (years <= 2100))
        if indices.size != 1032:
            raise ValueError(f"{path}: expected 1032 future months; found {indices.size}")
        subset = dataset[list(TIME_SERIES_VARIABLES)].isel(time=indices)
        lat = np.asarray(subset.lat.values, dtype=np.float64)
        lon = np.asarray(subset.lon.values, dtype=np.float64)
        key = (lat.tobytes(), lon.tobytes())
        if key not in weights_cache:
            weights_cache[key] = build_china_weights(lat, lon, shapefile)
        weights = weights_cache[key]
        output: dict[str, np.ndarray] = {}
        expected_years = np.arange(2015, 2101, dtype=np.int64)
        for variable in TIME_SERIES_VARIABLES:
            monthly = spatial_weighted_mean(subset[variable], weights)
            annual = monthly.groupby("time.year").mean("time", skipna=True)
            found_years = np.asarray(annual.year.values, dtype=np.int64)
            if not np.array_equal(found_years, expected_years):
                raise ValueError(f"{path}/{variable}: incomplete annual coverage")
            output[variable] = np.asarray(annual.values, dtype=np.float64)
        output["year"] = expected_years
        return output


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"model_count": 0, **{name: math.nan for name in ENSEMBLE_STATS}}
    return {
        "model_count": int(array.size),
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "std": float(np.std(array)),
    }


def unit_for(variable: str) -> str:
    return {
        "tas_c": "degree_Celsius",
        "rsds": "W m-2",
        "sfcWind_10m": "m s-1",
        "pvpot": "W m-2",
        "wpd": "W m-2",
    }[variable]


def build_annual_tables(
    energy_root: Path,
    output_root: Path,
    shapefile: Path,
    models: list[str],
    scenarios: list[str],
) -> tuple[Path, Path]:
    model_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    weights_cache: dict[tuple[bytes, bytes], np.ndarray] = {}
    total = len(models) * len(scenarios) * len(METHODS)
    counter = 0
    for model in models:
        for scenario in scenarios:
            for method in METHODS:
                counter += 1
                print(f"  Annual series [{counter:03d}/{total:03d}] {model}/{scenario}/{method}")
                series = annual_native_series(
                    energy_path(energy_root, model, scenario, method),
                    weights_cache,
                    shapefile,
                )
                for variable in TIME_SERIES_VARIABLES:
                    for year, value in zip(series["year"], series[variable]):
                        row = {
                            "model": model,
                            "scenario": scenario,
                            "method": method,
                            "route": ROUTES[method],
                            "year": int(year),
                            "variable": variable,
                            "unit": unit_for(variable),
                            "china_area_weighted_annual_mean": float(value),
                        }
                        model_rows.append(row)
                        grouped[(scenario, method, str(int(year)), variable)].append(float(value))

    ensemble_rows: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        row: dict[str, object] = {
            "scenario": key[0],
            "method": key[1],
            "route": ROUTES[key[1]],
            "year": int(key[2]),
            "variable": key[3],
            "unit": unit_for(key[3]),
            "ensemble_rule": "equal weight across model-level China annual means",
        }
        row.update(numeric_summary(values))
        ensemble_rows.append(row)

    model_path = output_root / "final_annual_model_timeseries.csv"
    ensemble_path = output_root / "final_annual_ensemble_timeseries.csv"
    write_csv(model_path, model_rows)
    write_csv(ensemble_path, ensemble_rows)
    return model_path, ensemble_path


def coord_values(dataset: xr.Dataset, name: str) -> list[str]:
    return [str(value) for value in dataset.coords[name].values.tolist()]


def area_mean_map(data: xr.DataArray, area_weights: xr.DataArray) -> float:
    """Intersection-area-weighted national mean of a 1-degree map."""
    values = np.asarray(data.transpose("lat", "lon").values, dtype=np.float64)
    weights = np.asarray(area_weights.transpose("lat", "lon").values, dtype=np.float64)
    return float(area_weighted_mean(values, weights))


def paper_category(values: xr.DataArray) -> xr.DataArray:
    """Return paper Table-3 categories: 1=very strong complementarity, 8=very strong similarity."""
    data = values.astype(np.float64)
    category = xr.full_like(data, np.nan, dtype=np.float32)
    category = xr.where((data >= -1.0) & (data <= -0.9), 1.0, category)
    category = xr.where((data > -0.9) & (data <= -0.6), 2.0, category)
    category = xr.where((data > -0.6) & (data <= -0.3), 3.0, category)
    category = xr.where((data > -0.3) & (data <= 0.0), 4.0, category)
    category = xr.where((data > 0.0) & (data < 0.3), 5.0, category)
    category = xr.where((data >= 0.3) & (data < 0.6), 6.0, category)
    category = xr.where((data >= 0.6) & (data < 0.9), 7.0, category)
    category = xr.where((data >= 0.9) & (data <= 1.0), 8.0, category)
    return category


def drop_selection_coords(values: xr.DataArray) -> xr.DataArray:
    """Drop scalar coordinates left behind by ``sel``.

    xarray keeps selectors such as ``statistic='mean'`` as scalar coordinates.
    Arrays selected with different values (for example ``mean`` and
    ``model_count``) cannot then be combined into one Dataset even though the
    selector is no longer a dimension.  Dimension coordinates are retained.
    """
    return values.reset_coords(drop=True)


def build_final_maps(ensemble_root: Path, output_root: Path) -> Path:
    period_path = ensemble_root / "multimodel_energy_period_maps_1deg.nc"
    change_path = ensemble_root / "multimodel_energy_change_maps_1deg.nc"
    difference_path = ensemble_root / "multimodel_method_difference_maps_1deg.nc"
    complementarity_path = ensemble_root / "multimodel_complementarity_maps_1deg.nc"

    with open_netcdf(period_path) as period_source, open_netcdf(change_path) as change_source, open_netcdf(difference_path) as difference_source, open_netcdf(complementarity_path) as comp_source:
        for required, dataset in (
            ("period_value", period_source),
            ("change_value", change_source),
            ("optimized_minus_paper", difference_source),
            ("spearman_rho", comp_source),
        ):
            if required not in dataset:
                raise KeyError(f"{required} is missing from an ensemble input")
        for dataset in (period_source, change_source, difference_source, comp_source):
            if "china_intersection_area_km2" not in dataset:
                raise KeyError("china_intersection_area_km2 is missing from an ensemble input")
        if "gridcell_area_km2" not in period_source:
            raise KeyError("gridcell_area_km2 is missing from the period maps input")

        period_mean = drop_selection_coords(
            period_source["period_value"].sel(statistic="mean").load()
        )
        period_count = drop_selection_coords(
            period_source["period_value"].sel(statistic="model_count").load()
        )
        absolute_change = drop_selection_coords(
            change_source["change_value"].sel(
                change_metric="absolute_change", statistic="mean"
            ).load()
        )
        gridwise_percent = drop_selection_coords(
            change_source["change_value"].sel(
                change_metric="relative_change_percent", statistic="mean"
            ).load()
        )
        change_count = drop_selection_coords(
            change_source["change_value"].sel(
                change_metric="relative_change_percent", statistic="model_count"
            ).load()
        )
        sign_agreement = drop_selection_coords(
            change_source["sign_agreement_percent"].sel(
                change_metric="relative_change_percent"
            ).load()
        )
        method_difference = drop_selection_coords(
            difference_source["optimized_minus_paper"].sel(statistic="mean").load()
        )
        rho = drop_selection_coords(
            comp_source["spearman_rho"].sel(
                definition=list(PRIMARY_COMPLEMENTARITY), statistic="mean"
            ).load()
        )
        rho_count = drop_selection_coords(
            comp_source["spearman_rho"].sel(
                definition=list(PRIMARY_COMPLEMENTARITY), statistic="model_count"
            ).load()
        )
        gridcell_area = period_source["gridcell_area_km2"].load()
        china_intersects = period_source["china_intersects"].load()
        china_intersection_area = period_source["china_intersection_area_km2"].load()
        china_area_fraction = period_source["china_area_fraction"].load()

    wpd = drop_selection_coords(period_mean.sel(energy_variable="wpd"))
    log10_wpd = xr.where(wpd > 0.0, np.log10(wpd), np.nan).rename("wpd_log10")
    categories = paper_category(rho).rename("spearman_category")
    categories.attrs.update(
        category_codes=(
            "1 very_strong_complementarity; 2 strong_complementarity; "
            "3 moderate_complementarity; 4 weak_complementarity; "
            "5 weak_similarity; 6 moderate_similarity; "
            "7 strong_similarity; 8 very_strong_similarity"
        ),
        zero_rule="rho == 0 belongs to weak complementarity, following paper Table 3",
    )

    final_maps = xr.Dataset(
        {
            "period_mean": period_mean,
            "period_model_count": period_count,
            "wpd_log10": log10_wpd,
            "absolute_change": absolute_change,
            "gridwise_relative_change_percent": gridwise_percent,
            "change_model_count": change_count,
            "change_sign_agreement_percent": sign_agreement,
            "optimized_minus_paper": method_difference,
            "spearman_rho": rho,
            "spearman_category": categories,
            "spearman_model_count": rho_count,
            "gridcell_area_km2": gridcell_area,
            "china_intersects": china_intersects,
            "china_intersection_area_km2": china_intersection_area,
            "china_area_fraction": china_area_fraction,
        }
    )
    final_maps.attrs.update(
        routes="B=paper_qm model-first; D=optimized model-first",
        excluded_routes="A and C ensemble-first products are not used",
        periods="1994-2014, 2040-2060, 2080-2100 inclusive",
        wpd_log_rule="log10 is display-only for the paper Fig. 7 analogue",
        change_rule="each model change first, then equal-weight model ensemble",
        national_percent_rule=(
            "official national statistics are native-grid model-first "
            "intersection-area-weighted period means (definition 2); this NetCDF "
            "only holds the common 1-degree maps and their map summary"
        ),
        primary_complementarity="seasonal_full (84 samples) and monthly_full (252 samples)",
    )
    maps_path = output_root / "final_B_D_figure_maps_1deg.nc"
    write_dataset(maps_path, final_maps)
    return maps_path


def build_national_change_tables(
    energy_root: Path,
    output_root: Path,
    shapefile: Path,
    models: list[str],
    scenarios: list[str],
    maps_path: Path,
) -> tuple[Path, Path]:
    """National change + D-B summaries from native-grid model-first means.

    ``historical_china_area_mean`` / ``future_china_area_mean`` and the derived
    ``percent_change_of_area_mean`` are the equal-weight 17-model ensemble of each
    model's native-grid intersection-area-weighted period mean (definition 2) —
    NOT the 1-degree common-grid map mean.  The 1-degree common-grid map mean is
    still reported, explicitly labelled, as ``common_grid_map_*_area_mean``.
    """
    # 1. Native-grid per-model period means -> equal-weight ensemble.
    weights_cache: dict[tuple[bytes, bytes], np.ndarray] = {}
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    total = len(models) * len(scenarios) * len(METHODS)
    counter = 0
    for model in models:
        for scenario in scenarios:
            for method in METHODS:
                counter += 1
                print(
                    f"  Native period means [{counter:03d}/{total:03d}] "
                    f"{model}/{scenario}/{method}"
                )
                means = native_period_means(
                    energy_path(energy_root, model, scenario, method),
                    weights_cache,
                    shapefile,
                )
                for period_name in PERIODS:
                    for variable in ENERGY_VARIABLES:
                        grouped[(scenario, method, period_name, variable)].append(
                            means[period_name][variable]
                        )
    ensemble = {key: float(np.mean(values)) for key, values in grouped.items()}

    # 2. Common 1-degree grid, for the map summary + gridwise-vs-national gap.
    with open_netcdf(maps_path) as maps:
        period_mean = maps["period_mean"].load()
        gridwise_percent = maps["gridwise_relative_change_percent"].load()
        china_intersection_area = maps["china_intersection_area_km2"].load()

    summary_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for method in METHODS:
            for variable in ENERGY_VARIABLES:
                historical = ensemble[(scenario, method, "historical", variable)]
                map_historical = area_mean_map(
                    period_mean.sel(
                        {
                            "scenario": scenario,
                            "method": method,
                            "period": "historical",
                            "energy_variable": variable,
                        }
                    ),
                    china_intersection_area,
                )
                for future_period in FUTURE_PERIODS:
                    future = ensemble[(scenario, method, future_period, variable)]
                    absolute = future - historical
                    national_percent = (
                        100.0 * absolute / historical
                        if abs(historical) > 1.0e-12
                        else math.nan
                    )
                    map_future = area_mean_map(
                        period_mean.sel(
                            {
                                "scenario": scenario,
                                "method": method,
                                "period": future_period,
                                "energy_variable": variable,
                            }
                        ),
                        china_intersection_area,
                    )
                    area_mean_grid_percent = area_mean_map(
                        gridwise_percent.sel(
                            {
                                "scenario": scenario,
                                "method": method,
                                "future_period": future_period,
                                "energy_variable": variable,
                            }
                        ),
                        china_intersection_area,
                    )
                    summary_rows.append(
                        {
                            "scenario": scenario,
                            "method": method,
                            "route": ROUTES[method],
                            "historical_period": "1994-2014",
                            "future_period": future_period,
                            "future_years": (
                                f"{PERIODS[future_period][0]}-{PERIODS[future_period][1]}"
                            ),
                            "variable": variable,
                            "unit": "W m-2",
                            "national_mean_source": (
                                "equal-weight 17-model ensemble of native-grid "
                                "intersection-area-weighted period means"
                            ),
                            "historical_china_area_mean": historical,
                            "future_china_area_mean": future,
                            "absolute_change_of_area_mean": absolute,
                            "percent_change_of_area_mean": national_percent,
                            "area_mean_of_gridwise_percent_change": area_mean_grid_percent,
                            "gridwise_percent_mean_minus_national_percent_change_pp": (
                                area_mean_grid_percent - national_percent
                            ),
                            "common_grid_map_historical_area_mean": map_historical,
                            "common_grid_map_future_area_mean": map_future,
                        }
                    )
    summary_path = output_root / "final_national_change_summary.csv"
    write_csv(summary_path, summary_rows)

    method_rows: list[dict[str, object]] = []
    lookup = {
        (row["scenario"], row["future_period"], row["variable"], row["method"]): row
        for row in summary_rows
    }
    for scenario in scenarios:
        for future_period in FUTURE_PERIODS:
            for variable in ENERGY_VARIABLES:
                paper = lookup[(scenario, future_period, variable, "paper_qm")]
                optimized = lookup[(scenario, future_period, variable, "optimized")]
                method_rows.append(
                    {
                        "scenario": scenario,
                        "future_period": future_period,
                        "variable": variable,
                        "comparison": "D_minus_B",
                        "national_mean_source": paper["national_mean_source"],
                        "optimized_minus_paper_period_mean": (
                            optimized["future_china_area_mean"]
                            - paper["future_china_area_mean"]
                        ),
                        "optimized_minus_paper_period_mean_percent": (
                            100.0
                            * (
                                optimized["future_china_area_mean"]
                                / paper["future_china_area_mean"]
                                - 1.0
                            )
                        ),
                        "difference_in_national_change_percentage_points": (
                            optimized["percent_change_of_area_mean"]
                            - paper["percent_change_of_area_mean"]
                        ),
                    }
                )
    method_path = output_root / "final_D_minus_B_summary.csv"
    write_csv(method_path, method_rows)
    return summary_path, method_path


def main() -> int:
    arguments = parse_arguments()
    script_dir = Path(__file__).resolve().parent
    project_root = PROJECT_ROOT
    energy_root = (
        arguments.energy_root.expanduser().resolve()
        if arguments.energy_root
        else get_path("data_processed_energy_metrics")
    )
    ensemble_root = (
        arguments.ensemble_root.expanduser().resolve()
        if arguments.ensemble_root
        else get_path("results_figure_data_multimodel_ensemble")
    )
    output_root = (
        arguments.output_root.expanduser().resolve()
        if arguments.output_root
        else get_path("results_figure_data_final_analysis")
    )
    shapefile = (
        arguments.china_shapefile.expanduser().resolve()
        if arguments.china_shapefile
        else get_path("china_shapefile")
    )
    if not energy_root.is_dir():
        raise FileNotFoundError(f"Energy root does not exist: {energy_root}")
    if not ensemble_root.is_dir():
        raise FileNotFoundError(f"Ensemble root does not exist: {ensemble_root}")
    models = discover_models(
        energy_root,
        arguments.models,
        list(arguments.scenarios),
        arguments.expected_models,
    )
    required = required_paths(
        energy_root, ensemble_root, models, list(arguments.scenarios)
    )
    missing = [path for path in required if not path.is_file()]

    print("=" * 100)
    print("FINAL B/D ANALYSIS PRODUCT BUILDER")
    print("=" * 100)
    print(f"Models          : {len(models)}")
    print(f"Scenarios       : {', '.join(arguments.scenarios)}")
    print("Routes          : B=paper_qm model-first; D=optimized model-first")
    print("Excluded        : A/C ensemble-first products")
    print("Periods         : 1994-2014, 2040-2060, 2080-2100")
    print("Complementarity : seasonal_full and monthly_full")
    print(f"Energy root     : {energy_root}")
    print(f"Ensemble root   : {ensemble_root}")
    print(f"China boundary  : {shapefile}")
    print(f"Output root     : {output_root}")
    if missing:
        preview = "\n  ".join(str(path) for path in missing[:20])
        raise FileNotFoundError(
            f"Missing {len(missing)} required files:\n  {preview}"
        )
    if not shapefile.is_file():
        raise FileNotFoundError(f"China shapefile does not exist: {shapefile}")
    if arguments.dry_run:
        print("DRY RUN COMPLETED: all B/D inputs and the China boundary are present.")
        print("No output files were written.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    model_series_path = output_root / "final_annual_model_timeseries.csv"
    ensemble_series_path = output_root / "final_annual_ensemble_timeseries.csv"
    if arguments.reuse_annual:
        invalid_annual = [
            path
            for path in (model_series_path, ensemble_series_path)
            if not path.is_file() or path.stat().st_size == 0
        ]
        if invalid_annual:
            raise FileNotFoundError(
                "--reuse-annual was requested, but an annual CSV is missing or "
                "empty: " + ", ".join(str(path) for path in invalid_annual)
            )
        print("Reusing annual CSV outputs from the completed part of the prior run.")
    else:
        model_series_path, ensemble_series_path = build_annual_tables(
            energy_root,
            output_root,
            shapefile,
            models,
            list(arguments.scenarios),
        )
    maps_path = build_final_maps(ensemble_root, output_root)
    national_path, method_path = build_national_change_tables(
        energy_root,
        output_root,
        shapefile,
        models,
        list(arguments.scenarios),
        maps_path,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "models": models,
        "model_count": len(models),
        "scenarios": list(arguments.scenarios),
        "routes": {
            "B": "paper_qm, model-level energy/complementarity before ensemble",
            "D": "optimized, model-level energy/complementarity before ensemble",
        },
        "excluded_routes": ["A", "C"],
        "periods": PERIODS,
        "annual_period": "2015-2100",
        "annual_spatial_weighting": "geodesic intersection area with the China boundary (WGS84)",
        "annual_temporal_weighting": "simple mean of 12 monthly means",
        "primary_complementarity": list(PRIMARY_COMPLEMENTARITY),
        "national_percentage_definition": (
            "100 * (future China area mean - historical China area mean) / "
            "historical China area mean"
        ),
        "national_mean_source": (
            "equal-weight 17-model ensemble of native-grid "
            "intersection-area-weighted period means (definition 2); the "
            "common 1-degree map mean is reported separately in the CSV as "
            "common_grid_map_*_area_mean"
        ),
        "map_percentage_definition": (
            "each model grid-cell percentage change first, then equal-weight ensemble"
        ),
        "outputs": {
            "annual_model_timeseries": str(model_series_path),
            "annual_ensemble_timeseries": str(ensemble_series_path),
            "figure_maps": str(maps_path),
            "national_change_summary": str(national_path),
            "D_minus_B_summary": str(method_path),
        },
    }
    manifest_path = output_root / "final_analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print("FINAL B/D ANALYSIS PRODUCTS COMPLETED")
    print("=" * 100)
    print(f"Annual model series    : {model_series_path}")
    print(f"Annual ensemble series : {ensemble_series_path}")
    print(f"Figure maps            : {maps_path}")
    print(f"National summary       : {national_path}")
    print(f"D-B summary            : {method_path}")
    print(f"Manifest               : {manifest_path}")
    print("Next: create the paper-reproduction and optimization figures.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # pragma: no cover
        print(f"ERROR: {error}", file=sys.stderr)
        raise
