#!/usr/bin/env python
r"""Formal 17-model x 3-scenario PV/WPD energy-metrics production.

Place this script beside ``compute_energy_metrics_pilot_v2.py`` in::

    scripts/04_energy_metrics

The tested v2 pilot is imported as the numerical engine.  For each model,
historical monthly QM and the ERA5 reference are computed and written once,
then reused in memory by ssp126, ssp245, and ssp585.

Examples
--------
    python compute_energy_metrics_batch.py --dry-run
    python compute_energy_metrics_batch.py
    python compute_energy_metrics_batch.py --models CAS-ESM2-0
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
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import compute_energy_metrics_pilot_v2 as engine
except ImportError as error:  # pragma: no cover - installation-specific
    raise RuntimeError(
        "Place compute_energy_metrics_pilot_v2.py in the same directory as this script."
    ) from error


SCENARIOS = ("ssp126", "ssp245", "ssp585")
EXPECTED_MODELS = 17
HISTORICAL_REQUIRED = (
    "historical_energy_{model}_qm_historical_199401-201412.nc",
    "historical_energy_{model}_era5_reference_199401-201412.nc",
    "historical_complementarity_{model}_qm_historical.nc",
    "historical_complementarity_{model}_era5_reference.nc",
    "historical_baseline_comparison.csv",
    "historical_baseline_complementarity_summary.csv",
    "historical_run_manifest.json",
)
SCENARIO_REQUIRED = (
    "energy_monthly_{model}_{scenario}_paper_qm_199401-210012.nc",
    "energy_monthly_{model}_{scenario}_optimized_199401-210012.nc",
    "complementarity_{model}_{scenario}_paper_qm.nc",
    "complementarity_{model}_{scenario}_optimized.nc",
    "energy_qc_summary.csv",
    "energy_period_summary.csv",
    "hub_height_sensitivity.csv",
    "complementarity_summary.csv",
    "historical_future_transition.csv",
    "energy_run_manifest.json",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run formal PV/WPD energy metrics for all corrected CMIP6 "
            "model/scenario combinations."
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
        "--output-root",
        type=Path,
        help="Default: <project-root>/data/processed/energy_metrics",
    )
    parser.add_argument(
        "--china-shapefile",
        type=Path,
        help=(
            "Default: <project-root>/data/boundaries/"
            "china_qc_boundary.shp"
        ),
    )
    parser.add_argument(
        "--rectangular-domain",
        action="store_true",
        help="Use the complete rectangular grid instead of the China polygon.",
    )
    parser.add_argument("--models", nargs="+", help="Optional model subset.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
    )
    parser.add_argument("--expected-models", type=int, default=EXPECTED_MODELS)
    parser.add_argument("--hub-height", type=float, default=engine.MAIN_HUB_HEIGHT_M)
    parser.add_argument(
        "--sensitivity-heights",
        nargs="+",
        type=float,
        default=list(engine.SENSITIVITY_HEIGHTS_M),
    )
    parser.add_argument("--lower", type=float, default=engine.DEFAULT_LOWER)
    parser.add_argument("--upper", type=float, default=engine.DEFAULT_UPPER)
    parser.add_argument(
        "--n-quantiles", type=int, default=engine.DEFAULT_N_QUANTILES
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all inputs and settings without writing outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute combinations whose required output files already exist.",
    )
    arguments = parser.parse_args()
    if arguments.hub_height <= engine.REFERENCE_HEIGHT_M:
        parser.error("--hub-height must be greater than 10 m")
    if any(value <= engine.REFERENCE_HEIGHT_M for value in arguments.sensitivity_heights):
        parser.error("Every sensitivity height must be greater than 10 m")
    if not (0.0 < arguments.lower < arguments.upper < 1.0):
        parser.error("Require 0 < --lower < --upper < 1")
    if arguments.n_quantiles < 3:
        parser.error("--n-quantiles must be at least 3")
    if arguments.expected_models < 1:
        parser.error("--expected-models must be positive")
    arguments.bounds_label = (
        f"Q{round(arguments.lower * 100):02d}-"
        f"Q{round(arguments.upper * 100):02d}"
    )
    return arguments


def resolve_roots(arguments: argparse.Namespace) -> dict[str, Path | None]:
    project_root = (
        arguments.project_root.expanduser().resolve()
        if arguments.project_root
        else PROJECT_ROOT
    )
    roots: dict[str, Path | None] = {
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
        "output_root": (
            arguments.output_root.expanduser().resolve()
            if arguments.output_root
            else get_path("data_processed_energy_metrics")
        ),
    }
    if arguments.rectangular_domain:
        roots["china_shapefile"] = None
    elif arguments.china_shapefile:
        roots["china_shapefile"] = arguments.china_shapefile.expanduser().resolve()
    else:
        roots["china_shapefile"] = (
            get_path("china_shapefile")
        )

    for name in ("cmip6_root", "era5_root", "corrected_root"):
        path = roots[name]
        if path is None or not path.is_dir():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    shapefile = roots["china_shapefile"]
    if shapefile is not None and not shapefile.is_file():
        raise FileNotFoundError(
            f"China shapefile does not exist: {shapefile}. "
            "Supply --china-shapefile or use --rectangular-domain."
        )
    return roots


def discover_models(
    corrected_root: Path,
    requested: list[str] | None,
    scenarios: list[str],
    expected_models: int,
) -> list[str]:
    if requested:
        models = sorted(dict.fromkeys(requested))
    else:
        models = sorted(
            path.name
            for path in corrected_root.iterdir()
            if path.is_dir() and all((path / scenario).is_dir() for scenario in scenarios)
        )
    if not models:
        raise FileNotFoundError(f"No models found below {corrected_root}")
    missing = [
        f"{model}/{scenario}"
        for model in models
        for scenario in scenarios
        if not (corrected_root / model / scenario).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing corrected model/scenario directories: " + ", ".join(missing)
        )
    if requested is None and len(models) != expected_models:
        raise ValueError(
            f"Expected {expected_models} complete models; found {len(models)}: "
            + ", ".join(models)
        )
    return models


def add_context(
    rows: Iterable[dict[str, object]], model: str, scenario: str
) -> list[dict[str, object]]:
    return [
        {"model": model, "scenario": scenario, **row}
        for row in rows
    ]


def required_exist(root: Path, templates: tuple[str, ...], **values: str) -> bool:
    return all((root / template.format(**values)).is_file() for template in templates)


def load_corrected_only(
    input_paths: dict[str, dict[str, Path]],
    historical_baselines: dict[str, dict[str, engine.xr.DataArray]],
) -> dict[str, engine.xr.Dataset]:
    corrected: dict[str, engine.xr.Dataset] = {}
    target = historical_baselines["qm_historical"]
    for variable in engine.VARIABLES:
        data = engine.extract_corrected(input_paths["corrected"][variable], variable)
        engine.validate_time_range(
            data, engine.FUTURE_START, engine.FUTURE_END, f"corrected {variable}"
        )
        engine.validate_units(variable, data["raw"], f"corrected future {variable}")
        if not engine.grids_equal(target[variable], data):
            raise ValueError(f"Historical and corrected grids differ for {variable}")
        corrected[variable] = data
    return corrected


def write_historical(
    model: str,
    output_root: Path,
    baselines: dict[str, dict[str, engine.xr.DataArray]],
    spatial_mask: np.ndarray,
    qc_region: str,
    mask_metadata: dict[str, object],
    hub_height: float,
    bounds_label: str,
    n_quantiles: int,
) -> dict[str, engine.xr.Dataset]:
    model_root = output_root / model / "historical"
    model_root.mkdir(parents=True, exist_ok=True)
    baseline_energies: dict[str, engine.xr.Dataset] = {}
    complementarity_rows: list[dict[str, object]] = []
    outputs: dict[str, object] = {}
    historical_period = (("historical", 1994, 2014),)

    for baseline in engine.HISTORICAL_BASELINES:
        meteorology = engine.historical_meteorology(baselines[baseline])
        energy_support = engine.compute_energy(meteorology, hub_height)
        baseline_energies[baseline] = energy_support
        energy = engine.select_years(energy_support, 1994, 2014)
        energy.attrs.update(
            model=model,
            scenario="historical",
            historical_baseline=baseline,
            output_period="1994-2014",
            qc_region=qc_region,
        )
        energy_path = model_root / (
            f"historical_energy_{model}_{baseline}_199401-201412.nc"
        )
        engine.write_dataset(energy_path, energy)
        complementarity, rows = engine.compute_complementarity(
            baseline,
            energy_support,
            spatial_mask,
            qc_region,
            periods=historical_period,
        )
        complementarity_path = model_root / (
            f"historical_complementarity_{model}_{baseline}.nc"
        )
        engine.write_dataset(complementarity_path, complementarity, time_chunks=False)
        complementarity_rows.extend(add_context(rows, model, "historical"))
        outputs[baseline] = {
            "energy_netcdf": str(energy_path),
            "complementarity_netcdf": str(complementarity_path),
        }

    comparison_rows = add_context(
        engine.make_historical_baseline_rows(
            baseline_energies, spatial_mask, qc_region
        ),
        model,
        "historical",
    )
    comparison_path = model_root / "historical_baseline_comparison.csv"
    complementarity_summary_path = (
        model_root / "historical_baseline_complementarity_summary.csv"
    )
    engine.write_csv(comparison_path, comparison_rows)
    engine.write_csv(complementarity_summary_path, complementarity_rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "model": model,
        "scenario": "historical",
        "historical_qm": {
            "mode": "monthly multiplicative QM for tas, rsds, and sfcWind",
            "fit_period": "1959-2014",
            "reference": "ERA5 on each GCM grid",
            "quantile_bounds": bounds_label,
            "quantile_nodes": int(n_quantiles),
        },
        "output_period": "1994-2014",
        "mask": mask_metadata,
        "outputs": outputs,
        "csv_outputs": {
            "historical_baseline_comparison": str(comparison_path),
            "historical_baseline_complementarity_summary": str(
                complementarity_summary_path
            ),
        },
    }
    manifest_path = model_root / "historical_run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return baseline_energies


def compute_scenario(
    model: str,
    scenario: str,
    output_root: Path,
    baselines: dict[str, dict[str, engine.xr.DataArray]],
    corrected: dict[str, engine.xr.Dataset],
    input_paths: dict[str, dict[str, Path]],
    spatial_mask: np.ndarray,
    qc_region: str,
    mask_metadata: dict[str, object],
    arguments: argparse.Namespace,
) -> None:
    scenario_root = output_root / model / scenario
    scenario_root.mkdir(parents=True, exist_ok=True)
    qc_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    height_rows: list[dict[str, object]] = []
    complementarity_rows: list[dict[str, object]] = []
    method_energies: dict[str, engine.xr.Dataset] = {}
    outputs: dict[str, object] = {}

    for method in engine.METHODS:
        meteorology = engine.method_meteorology(
            baselines["qm_historical"], corrected, method
        )
        energy_support = engine.compute_energy(meteorology, arguments.hub_height)
        method_energies[method] = energy_support
        energy = engine.select_years(energy_support, 1994, 2100)
        if energy.sizes["time"] != 1284:
            raise ValueError(
                f"{model}/{scenario}/{method}: expected 1284 months, "
                f"found {energy.sizes['time']}"
            )
        energy.attrs.update(
            model=model,
            scenario=scenario,
            method=method,
            output_period="1994-2100",
            paper_periods="1994-2014, 2040-2060, 2080-2100",
            qc_region=qc_region,
        )
        energy_path = scenario_root / (
            f"energy_monthly_{model}_{scenario}_{method}_199401-210012.nc"
        )
        engine.write_dataset(energy_path, energy)
        qc_rows.extend(
            add_context(
                engine.make_qc_rows(method, energy, spatial_mask, qc_region),
                model,
                scenario,
            )
        )
        period_rows.extend(
            add_context(
                engine.make_period_rows(method, energy, spatial_mask, qc_region),
                model,
                scenario,
            )
        )
        height_rows.extend(
            add_context(
                engine.make_height_sensitivity_rows(
                    method,
                    meteorology,
                    list(arguments.sensitivity_heights),
                    spatial_mask,
                    qc_region,
                ),
                model,
                scenario,
            )
        )
        complementarity, rows = engine.compute_complementarity(
            method, energy_support, spatial_mask, qc_region
        )
        complementarity_path = scenario_root / (
            f"complementarity_{model}_{scenario}_{method}.nc"
        )
        engine.write_dataset(complementarity_path, complementarity, time_chunks=False)
        complementarity_rows.extend(add_context(rows, model, scenario))
        outputs[method] = {
            "energy_netcdf": str(energy_path),
            "complementarity_netcdf": str(complementarity_path),
        }

    transition_rows = add_context(
        engine.make_transition_rows(method_energies, spatial_mask, qc_region),
        model,
        scenario,
    )
    csv_outputs = {
        "qc": scenario_root / "energy_qc_summary.csv",
        "period_summary": scenario_root / "energy_period_summary.csv",
        "hub_height_sensitivity": scenario_root / "hub_height_sensitivity.csv",
        "complementarity_summary": scenario_root / "complementarity_summary.csv",
        "historical_future_transition": scenario_root / "historical_future_transition.csv",
    }
    for path, rows in (
        (csv_outputs["qc"], qc_rows),
        (csv_outputs["period_summary"], period_rows),
        (csv_outputs["hub_height_sensitivity"], height_rows),
        (csv_outputs["complementarity_summary"], complementarity_rows),
        (csv_outputs["historical_future_transition"], transition_rows),
    ):
        engine.write_csv(path, rows)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "model": model,
        "scenario": scenario,
        "methods": {
            "paper_qm": {
                "tas": "qm_paper",
                "rsds": "qm_paper",
                "sfcWind": "qm_paper",
            },
            "optimized": {
                "tas": "qdm_improved",
                "rsds": "qm_paper",
                "sfcWind": "qdm_improved",
            },
        },
        "historical_rule": (
            "Both branches reuse this model's monthly multiplicative-QM history "
            "for 1993-12 through 2014-12."
        ),
        "historical_output_directory": str(output_root / model / "historical"),
        "main_hub_height_m": float(arguments.hub_height),
        "sensitivity_heights_m": [
            float(value) for value in arguments.sensitivity_heights
        ],
        "paper_periods": [
            {"name": name, "start_year": start, "end_year": end}
            for name, start, end in engine.PERIODS
        ],
        "mask": mask_metadata,
        "input_files": {
            group: {variable: str(path) for variable, path in files.items()}
            for group, files in input_paths.items()
        },
        "outputs": outputs,
        "csv_outputs": {name: str(path) for name, path in csv_outputs.items()},
        "transition_warning_count": int(
            sum(bool(row["absolute_z_gt_3_warning"]) for row in transition_rows)
        ),
    }
    (scenario_root / "energy_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def combine_csvs(paths: list[Path], output: Path) -> int:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_csv_rows(path))
    if rows:
        engine.write_csv(output, rows)
    return len(rows)


def write_status(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        engine.write_csv(path, rows)


def print_header(
    arguments: argparse.Namespace,
    roots: dict[str, Path | None],
    models: list[str],
) -> None:
    print("=" * 100)
    print("FORMAL ENERGY METRICS BATCH")
    print("=" * 100)
    print(f"Models        : {len(models)}")
    print(f"Scenarios     : {', '.join(arguments.scenarios)}")
    print(f"Combinations  : {len(models) * len(arguments.scenarios)}")
    print(f"Bounds        : {arguments.bounds_label}")
    print(f"Hub height    : {arguments.hub_height:g} m")
    print(
        "Sensitivity   : "
        + ", ".join(f"{value:g} m" for value in arguments.sensitivity_heights)
    )
    print(f"Corrected root: {roots['corrected_root']}")
    print(f"Output root   : {roots['output_root']}")
    print(f"China boundary: {roots['china_shapefile']}")
    print("Historical    : computed once per model, reused by all scenarios")
    print("Methods       : paper=(QM,QM,QM); optimized=(QDM,QM,QDM)")


def main() -> int:
    arguments = parse_arguments()
    roots = resolve_roots(arguments)
    corrected_root = roots["corrected_root"]
    output_root = roots["output_root"]
    assert isinstance(corrected_root, Path)
    assert isinstance(output_root, Path)
    models = discover_models(
        corrected_root,
        arguments.models,
        list(arguments.scenarios),
        arguments.expected_models,
    )
    print_header(arguments, roots, models)

    status_rows: list[dict[str, object]] = []
    failures = 0
    dry_validated = 0
    for model_index, model in enumerate(models, start=1):
        historical_root = output_root / model / "historical"
        historical_complete = required_exist(
            historical_root, HISTORICAL_REQUIRED, model=model
        )
        scenario_completion = {
            scenario: required_exist(
                output_root / model / scenario,
                SCENARIO_REQUIRED,
                model=model,
                scenario=scenario,
            )
            for scenario in arguments.scenarios
        }
        if (
            not arguments.dry_run
            and not arguments.force
            and historical_complete
            and all(scenario_completion.values())
        ):
            print(f"[{model_index:02d}/{len(models):02d}] {model}: already complete; skipped")
            status_rows.append(
                {
                    "model": model,
                    "scenario": "all",
                    "task": "model",
                    "status": "skipped_complete",
                    "message": "All historical and scenario outputs already exist.",
                }
            )
            continue

        print(f"\n[{model_index:02d}/{len(models):02d}] {model}")
        baselines = None
        spatial_mask = None
        qc_region = None
        mask_metadata = None
        first_corrected = None
        first_paths = None
        try:
            first_scenario = arguments.scenarios[0]
            first_paths = engine.discover_inputs(
                roots["cmip6_root"],
                roots["era5_root"],
                corrected_root,
                model,
                first_scenario,
            )
            baselines, first_corrected = engine.load_inputs(
                first_paths,
                arguments.lower,
                arguments.upper,
                arguments.n_quantiles,
            )
            spatial_mask, qc_region, mask_metadata = engine.build_mask(
                baselines["qm_historical"]["tas"], roots["china_shapefile"]
            )
            print(
                f"  Grid/mask: {int(spatial_mask.sum())}/{int(spatial_mask.size)} "
                f"cells ({qc_region})"
            )
        except Exception as error:
            failures += 1
            message = f"Historical/input preparation failed: {error}"
            print(f"  FAILED: {message}")
            status_rows.append(
                {
                    "model": model,
                    "scenario": "historical",
                    "task": "preparation",
                    "status": "failed",
                    "message": message,
                    "traceback": traceback.format_exc(),
                }
            )
            continue

        assert baselines is not None
        assert spatial_mask is not None
        assert qc_region is not None
        assert mask_metadata is not None
        assert first_corrected is not None
        assert first_paths is not None

        if arguments.dry_run:
            status_rows.append(
                {
                    "model": model,
                    "scenario": "historical",
                    "task": "validation",
                    "status": "validated",
                    "message": "Historical CMIP6, ERA5, monthly QM, grid, and mask valid.",
                }
            )
        elif arguments.force or not historical_complete:
            try:
                write_historical(
                    model,
                    output_root,
                    baselines,
                    spatial_mask,
                    qc_region,
                    mask_metadata,
                    arguments.hub_height,
                    arguments.bounds_label,
                    arguments.n_quantiles,
                )
                print("  Historical: completed")
                status_rows.append(
                    {
                        "model": model,
                        "scenario": "historical",
                        "task": "production",
                        "status": "completed",
                        "message": "Historical QM and ERA5 outputs written once.",
                    }
                )
            except Exception as error:
                failures += 1
                print(f"  Historical FAILED: {error}")
                status_rows.append(
                    {
                        "model": model,
                        "scenario": "historical",
                        "task": "production",
                        "status": "failed",
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                continue
        else:
            print("  Historical: existing complete output reused")

        for scenario in arguments.scenarios:
            if (
                not arguments.dry_run
                and not arguments.force
                and scenario_completion[scenario]
            ):
                print(f"  {scenario}: already complete; skipped")
                status_rows.append(
                    {
                        "model": model,
                        "scenario": scenario,
                        "task": "scenario",
                        "status": "skipped_complete",
                        "message": "Required outputs already exist.",
                    }
                )
                continue
            try:
                if scenario == arguments.scenarios[0]:
                    input_paths = first_paths
                    corrected = first_corrected
                else:
                    input_paths = engine.discover_inputs(
                        roots["cmip6_root"],
                        roots["era5_root"],
                        corrected_root,
                        model,
                        scenario,
                    )
                    corrected = load_corrected_only(input_paths, baselines)
                if arguments.dry_run:
                    dry_validated += 1
                    print(f"  {scenario}: validated")
                    status = "validated"
                    message = "Corrected inputs, variables, units, dates, and grid valid."
                else:
                    compute_scenario(
                        model,
                        scenario,
                        output_root,
                        baselines,
                        corrected,
                        input_paths,
                        spatial_mask,
                        qc_region,
                        mask_metadata,
                        arguments,
                    )
                    print(f"  {scenario}: completed")
                    status = "completed"
                    message = "Paper and optimized energy/complementarity outputs written."
                status_rows.append(
                    {
                        "model": model,
                        "scenario": scenario,
                        "task": "scenario",
                        "status": status,
                        "message": message,
                    }
                )
            except Exception as error:
                failures += 1
                print(f"  {scenario} FAILED: {error}")
                status_rows.append(
                    {
                        "model": model,
                        "scenario": scenario,
                        "task": "scenario",
                        "status": "failed",
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )

    print("\n" + "=" * 100)
    if arguments.dry_run:
        print("FORMAL ENERGY BATCH DRY RUN COMPLETED")
        print("=" * 100)
        print(f"Validated combinations: {dry_validated}/{len(models) * len(arguments.scenarios)}")
        print(f"Failures              : {failures}")
        print("No output files were written.")
        return 1 if failures else 0

    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "energy_batch_status.csv"
    write_status(status_path, status_rows)

    # Cross-model aggregate CSVs are written to results/tables/ (central config),
    # while per-model NetCDF and per-model CSVs stay in output_root (energy_metrics).
    tables_root = get_path("results_tables")
    tables_root.mkdir(parents=True, exist_ok=True)

    aggregate_specs = {
        "combined_energy_qc_summary.csv": "energy_qc_summary.csv",
        "combined_energy_period_summary.csv": "energy_period_summary.csv",
        "combined_hub_height_sensitivity.csv": "hub_height_sensitivity.csv",
        "combined_complementarity_summary.csv": "complementarity_summary.csv",
        "combined_historical_future_transition.csv": "historical_future_transition.csv",
    }
    aggregate_counts: dict[str, int] = {}
    for output_name, source_name in aggregate_specs.items():
        paths = [
            output_root / model / scenario / source_name
            for model in models
            for scenario in arguments.scenarios
            if (output_root / model / scenario / source_name).is_file()
        ]
        aggregate_counts[output_name] = combine_csvs(
            paths, tables_root / output_name
        )
    for output_name, source_name in (
        (
            "combined_historical_baseline_comparison.csv",
            "historical_baseline_comparison.csv",
        ),
        (
            "combined_historical_baseline_complementarity_summary.csv",
            "historical_baseline_complementarity_summary.csv",
        ),
    ):
        paths = [
            output_root / model / "historical" / source_name
            for model in models
            if (output_root / model / "historical" / source_name).is_file()
        ]
        aggregate_counts[output_name] = combine_csvs(
            paths, tables_root / output_name
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "models": models,
        "scenarios": list(arguments.scenarios),
        "expected_combinations": len(models) * len(arguments.scenarios),
        "historical_computations": len(models),
        "status_counts": {
            name: sum(row["status"] == name for row in status_rows)
            for name in sorted({str(row["status"]) for row in status_rows})
        },
        "failures": failures,
        "aggregate_csv_rows": aggregate_counts,
        "aggregate_csv_root": str(tables_root),
        "output_root": str(output_root),
    }
    manifest_path = output_root / "energy_batch_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("FORMAL ENERGY METRICS BATCH COMPLETED")
    print("=" * 100)
    print(f"Models       : {len(models)}")
    print(f"Combinations : {len(models) * len(arguments.scenarios)}")
    print(f"Failures     : {failures}")
    print(f"Output root  : {output_root}")
    print(f"Aggregate CSV root: {tables_root}")
    print(f"Status CSV   : {status_path}")
    print(f"Manifest     : {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
