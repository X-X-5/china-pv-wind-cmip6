#!/usr/bin/env python
r"""Re-summarize Stage 4 combined CSVs with intersection-area weighting.

This is a *migration-only* tool.  It reads the existing 272 energy-metrics
NetCDF files (``energy_monthly_*.nc`` and ``complementarity_*.nc``) and rewrites
the spatial-mean columns of the combined summary tables using the shared
intersection-area weights from ``scripts/common/area_weights.py``.  It never
recomputes or rewrites a NetCDF file.

Regenerated tables (``results/tables/``)
----------------------------------------
* ``combined_energy_period_summary.csv``
* ``combined_hub_height_sensitivity.csv``
* ``combined_complementarity_summary.csv``
* ``combined_historical_baseline_comparison.csv``
* ``combined_historical_baseline_complementarity_summary.csv``
* ``combined_energy_qc_summary.csv`` (``qc_region`` relabelled ``china_intersection``
  and the per-cell statistics recomputed over the intersection mask)
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
import math
import sys
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError as error:  # pragma: no cover
    raise RuntimeError("xarray is required. Activate the pvwind environment.") from error

import compute_energy_metrics_pilot_v2 as engine
from area_weights import china_intersection_weights  # noqa: E402


SCENARIOS = ("ssp126", "ssp245", "ssp585")
DEFINITIONS = (
    "monthly_full",
    "monthly_climatology",
    "seasonal_full",
    "seasonal_climatology",
)
_DEFINITION_TOTALS = {
    "monthly_full": 252,
    "monthly_climatology": 12,
    "seasonal_full": 84,
    "seasonal_climatology": 4,
}
EXPECTED_MODELS = 17
QC_REGION = "china_intersection"


def decoder():
    try:
        return xr.coders.CFDatetimeCoder(use_cftime=True)
    except AttributeError:  # pragma: no cover
        return True


def open_netcdf(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=decoder())


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-summarize Stage 4 tables with intersection-area weights.")
    parser.add_argument("--energy-root", type=Path, help="Default: data/processed/energy_metrics")
    parser.add_argument("--tables-root", type=Path, help="Default: results/tables")
    parser.add_argument("--china-shapefile", type=Path, help="Default: data/boundaries/china_qc_boundary.shp")
    parser.add_argument("--models", nargs="+", help="Optional model subset.")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--expected-models", type=int, default=EXPECTED_MODELS)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing CSVs.")
    return parser.parse_args()


def discover_models(energy_root: Path, requested, scenarios, expected: int) -> list[str]:
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
        raise ValueError(f"Expected {expected} models; found {len(models)}")
    return models


def add_context(rows, model: str, scenario: str) -> list[dict[str, object]]:
    return [{"model": model, "scenario": scenario, **row} for row in rows]


def definition_sample_info(definition: str) -> tuple[int, int]:
    total = _DEFINITION_TOTALS[definition]
    minimum = (
        total
        if "climatology" in definition
        else int(math.ceil(total * engine.MIN_FULL_SEQUENCE_FRACTION))
    )
    return total, minimum


def complementarity_rows_from_netcdf(
    path: Path,
    method: str,
    spatial_weights: np.ndarray,
    periods: tuple[tuple[str, int, int], ...],
) -> list[dict[str, object]]:
    """Re-aggregate an existing per-cell spearman_rho NetCDF into summary rows."""
    rows: list[dict[str, object]] = []
    with open_netcdf(path) as ds:
        rho_field = ds["spearman_rho"]
        for period_name, start, end in periods:
            for definition in DEFINITIONS:
                rho = np.asarray(
                    rho_field.sel(period=period_name, definition=definition).values,
                    dtype=np.float64,
                )
                total, minimum = definition_sample_info(definition)
                rows.append(
                    engine.complementarity_summary_row(
                        method,
                        period_name,
                        start,
                        end,
                        definition,
                        total,
                        minimum,
                        rho,
                        spatial_weights,
                        QC_REGION,
                    )
                )
    return rows


def historical_energy_path(energy_root: Path, model: str, baseline: str) -> Path:
    return energy_root / model / "historical" / (
        f"historical_energy_{model}_{baseline}_199401-201412.nc"
    )


def historical_complementarity_path(energy_root: Path, model: str, baseline: str) -> Path:
    return energy_root / model / "historical" / (
        f"historical_complementarity_{model}_{baseline}.nc"
    )


def scenario_energy_path(energy_root: Path, model: str, scenario: str, method: str) -> Path:
    return energy_root / model / scenario / (
        f"energy_monthly_{model}_{scenario}_{method}_199401-210012.nc"
    )


def scenario_complementarity_path(energy_root: Path, model: str, scenario: str, method: str) -> Path:
    return energy_root / model / scenario / (
        f"complementarity_{model}_{scenario}_{method}.nc"
    )


def grid_weights(lat: np.ndarray, lon: np.ndarray, shapefile: Path) -> np.ndarray:
    return china_intersection_weights(lat, lon, shapefile)["china_intersection_area_km2"]


def main() -> int:
    arguments = parse_arguments()
    energy_root = (
        arguments.energy_root.expanduser().resolve()
        if arguments.energy_root
        else get_path("data_processed_energy_metrics")
    )
    tables_root = (
        arguments.tables_root.expanduser().resolve()
        if arguments.tables_root
        else get_path("results_tables")
    )
    shapefile = (
        arguments.china_shapefile.expanduser().resolve()
        if arguments.china_shapefile
        else get_path("china_shapefile")
    )
    if not energy_root.is_dir():
        raise FileNotFoundError(f"Energy root does not exist: {energy_root}")
    if not shapefile.is_file():
        raise FileNotFoundError(f"China shapefile does not exist: {shapefile}")

    models = discover_models(energy_root, arguments.models, list(arguments.scenarios), arguments.expected_models)
    scenarios = list(arguments.scenarios)

    required: list[Path] = []
    for model in models:
        for baseline in engine.HISTORICAL_BASELINES:
            required.append(historical_energy_path(energy_root, model, baseline))
            required.append(historical_complementarity_path(energy_root, model, baseline))
        for scenario in scenarios:
            for method in engine.METHODS:
                required.append(scenario_energy_path(energy_root, model, scenario, method))
                required.append(scenario_complementarity_path(energy_root, model, scenario, method))
    missing = [p for p in required if not p.is_file()]
    if missing:
        preview = "\n  ".join(str(p) for p in missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} input NetCDF files:\n  {preview}")

    print("=" * 100)
    print("STAGE 4 RE-SUMMARIZATION (intersection-area weighting)")
    print("=" * 100)
    print(f"Models        : {len(models)}")
    print(f"Scenarios     : {', '.join(scenarios)}")
    print(f"Energy root   : {energy_root}")
    print(f"Tables root   : {tables_root}")
    print(f"China boundary: {shapefile}")
    print(f"QC region     : {QC_REGION}")

    if arguments.dry_run:
        print("DRY RUN COMPLETED: all inputs present; no CSVs written.")
        return 0

    period_rows_all: list[dict[str, object]] = []
    height_rows_all: list[dict[str, object]] = []
    complementarity_rows_all: list[dict[str, object]] = []
    historical_baseline_rows_all: list[dict[str, object]] = []
    historical_complementarity_rows_all: list[dict[str, object]] = []
    qc_rows_all: list[dict[str, object]] = []

    for model_index, model in enumerate(models, start=1):
        # One native grid per model; derive intersection weights once.
        with open_netcdf(historical_energy_path(energy_root, model, "qm_historical")) as probe:
            lat = np.asarray(probe.lat.values, dtype=np.float64)
            lon = np.asarray(probe.lon.values, dtype=np.float64)
        spatial_weights = grid_weights(lat, lon, shapefile)
        print(
            f"[{model_index:02d}/{len(models):02d}] {model}: "
            f"grid {lat.size}x{lon.size}, "
            f"{int((spatial_weights > 0.0).sum())} intersecting cells"
        )

        # --- historical baseline comparison ---------------------------------
        baseline_energies: dict[str, xr.Dataset] = {}
        for baseline in engine.HISTORICAL_BASELINES:
            baseline_energies[baseline] = open_netcdf(
                historical_energy_path(energy_root, model, baseline)
            ).load()
        try:
            baseline_rows = engine.make_historical_baseline_rows(
                baseline_energies, spatial_weights, QC_REGION
            )
            historical_baseline_rows_all.extend(add_context(baseline_rows, model, "historical"))
        finally:
            for ds in baseline_energies.values():
                ds.close()

        # --- historical complementarity -------------------------------------
        for baseline in engine.HISTORICAL_BASELINES:
            rows = complementarity_rows_from_netcdf(
                historical_complementarity_path(energy_root, model, baseline),
                baseline,
                spatial_weights,
                (("historical", 1994, 2014),),
            )
            historical_complementarity_rows_all.extend(add_context(rows, model, "historical"))

        # --- scenario period / height / complementarity summaries -----------
        for scenario in scenarios:
            for method in engine.METHODS:
                with open_netcdf(scenario_energy_path(energy_root, model, scenario, method)) as ds:
                    energy = ds.load()
                try:
                    qc_rows_all.extend(
                        add_context(
                            engine.make_qc_rows(method, energy, spatial_weights, QC_REGION),
                            model,
                            scenario,
                        )
                    )
                    period_rows_all.extend(
                        add_context(
                            engine.make_period_rows(method, energy, spatial_weights, QC_REGION),
                            model,
                            scenario,
                        )
                    )
                    height_rows_all.extend(
                        add_context(
                            engine.make_height_sensitivity_rows(
                                method,
                                energy["sfcWind_10m"],
                                list(engine.SENSITIVITY_HEIGHTS_M),
                                spatial_weights,
                                QC_REGION,
                            ),
                            model,
                            scenario,
                        )
                    )
                finally:
                    energy.close()
                rows = complementarity_rows_from_netcdf(
                    scenario_complementarity_path(energy_root, model, scenario, method),
                    method,
                    spatial_weights,
                    engine.PERIODS,
                )
                complementarity_rows_all.extend(add_context(rows, model, scenario))

    tables_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "combined_energy_period_summary.csv": period_rows_all,
        "combined_hub_height_sensitivity.csv": height_rows_all,
        "combined_complementarity_summary.csv": complementarity_rows_all,
        "combined_historical_baseline_comparison.csv": historical_baseline_rows_all,
        "combined_historical_baseline_complementarity_summary.csv": historical_complementarity_rows_all,
        "combined_energy_qc_summary.csv": qc_rows_all,
    }
    counts: dict[str, int] = {}
    for name, rows in outputs.items():
        if not rows:
            raise ValueError(f"No rows produced for {name}")
        path = tables_root / name
        engine.write_csv(path, rows)
        counts[name] = len(rows)

    print("=" * 100)
    print("STAGE 4 RE-SUMMARIZATION COMPLETED")
    print("=" * 100)
    for name, count in counts.items():
        print(f"  {name}: {count} rows")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # pragma: no cover
        print(f"ERROR: {error}", file=sys.stderr)
        raise
