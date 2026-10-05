#!/usr/bin/env python
r"""Run the tested monthly QM/QDM production program for all models/scenarios.

Place this file in:
    scripts/03_bias_correction

The script reuses ``qm_qdm_future_production_test.py`` without changing its
scientific calculations.  Corrected data, diagnostics, logs, and batch status
files are written to the configured production and audit directories.
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
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCENARIOS = ("ssp126", "ssp245", "ssp585")
VARIABLES = ("tas", "rsds", "sfcWind")
EXPECTED_MODEL_COUNT = 17
LOWER_QUANTILE = 0.02
UPPER_QUANTILE = 0.98
N_QUANTILES = 99
QDM_WINDOW_YEARS = 30


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch monthly QM/QDM production for all available CMIP6 models "
            "and SSP scenarios."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional model subset. Default: discover all model directories.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root. Default: the centralized project root.",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        help=(
            "Path to qm_qdm_future_production_test.py. Default: search the "
            "scripts/03_bias_correction directory."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output base. Default: <project-root>/data/processed/bias_correction.",
    )
    parser.add_argument(
        "--china-shapefile",
        type=Path,
        help=(
            "China boundary .shp used only for QC statistics (corrected NetCDF "
            "data keep the full rectangular grid). Default: project_paths "
            "'china_shapefile'."
        ),
    )
    parser.add_argument(
        "--skip-netcdf",
        action="store_true",
        help="Compute and write CSV diagnostics only, without corrected NetCDF files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every input combination without writing corrected data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun combinations that already have a complete manifest and NetCDF files.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed model/scenario combination.",
    )
    parser.add_argument(
        "--expected-model-count",
        type=int,
        default=EXPECTED_MODEL_COUNT,
        help="Expected discovered model count (default: 17; use 0 to disable the check).",
    )
    return parser.parse_args()


def resolve_paths(arguments: argparse.Namespace) -> dict[str, Path]:
    script_dir = Path(__file__).resolve().parent
    project_root = (
        arguments.project_root.resolve()
        if arguments.project_root
        else PROJECT_ROOT
    )
    cmip6_root = get_path("data_interim_cmip6_china_clipped")
    era5_root = get_path("data_interim_era5_on_gcm_grid")
    output_root = (
        arguments.output_root.resolve()
        if arguments.output_root
        else get_path("data_processed_bias_correction")
    )

    if arguments.engine:
        engine = arguments.engine.resolve()
    else:
        candidates = (
            script_dir / "qm_qdm_future_production_test.py",
            project_root / "scripts" / "03_bias_correction" / "qm_qdm_future_production_test.py",
        )
        engine = next((path for path in candidates if path.is_file()), candidates[0])

    china_shapefile = (
        arguments.china_shapefile.resolve()
        if arguments.china_shapefile
        else get_path("china_shapefile")
    )

    required = {
        "project_root": project_root,
        "cmip6_root": cmip6_root,
        "era5_root": era5_root,
        "output_root": output_root,
        "engine": engine,
        "china_shapefile": china_shapefile,
        "log_root": get_path("results_audits") / "qm_qdm_batch_logs",
    }
    for label in ("cmip6_root", "era5_root"):
        if not required[label].is_dir():
            raise FileNotFoundError(f"Missing {label}: {required[label]}")
    if not engine.is_file():
        raise FileNotFoundError(
            "Cannot find qm_qdm_future_production_test.py. Put it in "
            f"scripts/03_bias_correction or supply --engine. Expected: {engine}"
        )
    if not china_shapefile.is_file():
        raise FileNotFoundError(f"Missing China shapefile: {china_shapefile}")
    return required


def discover_models(cmip6_root: Path, era5_root: Path) -> list[str]:
    cmip6_models = {path.name for path in cmip6_root.iterdir() if path.is_dir()}
    era5_models = {path.name for path in era5_root.iterdir() if path.is_dir()}
    return sorted(cmip6_models & era5_models)


def combination_complete(output_root: Path, model: str, scenario: str) -> bool:
    destination = output_root / model / scenario
    manifest = destination / "future_run_manifest.json"
    if not manifest.is_file():
        return False
    for variable in VARIABLES:
        matches = list(
            destination.glob(
                f"corrected_{model}_{scenario}_{variable}_Q02-Q98_MW30_201501-210012.nc"
            )
        )
        if len(matches) != 1:
            return False
    return True


def run_combination(
    *,
    engine: Path,
    cmip6_root: Path,
    era5_root: Path,
    output_root: Path,
    log_root: Path,
    model: str,
    scenario: str,
    dry_run: bool,
    china_shapefile: Path,
    skip_netcdf: bool,
) -> tuple[int, Path]:
    mode = "dry_run" if dry_run else "production"
    log_path = log_root / f"{model}_{scenario}_{mode}.txt"
    command = [
        sys.executable,
        str(engine),
        "--model",
        model,
        "--scenario",
        scenario,
        "--cmip6-root",
        str(cmip6_root),
        "--era5-root",
        str(era5_root),
        "--output-root",
        str(output_root),
        "--lower",
        str(LOWER_QUANTILE),
        "--upper",
        str(UPPER_QUANTILE),
        "--n-quantiles",
        str(N_QUANTILES),
        "--qdm-window-years",
        str(QDM_WINDOW_YEARS),
        "--china-shapefile",
        str(china_shapefile),
    ]
    if skip_netcdf:
        command.append("--skip-netcdf")
    if dry_run:
        command.append("--dry-run")

    print("\n" + "=" * 100)
    print(f"Running {model} / {scenario} / {mode}")
    print("=" * 100)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()

    return return_code, log_path


def write_status_files(
    *,
    records: list[dict[str, object]],
    configuration: dict[str, object],
    dry_run: bool,
) -> tuple[Path, Path, Path]:
    status_dir = get_path("results_manifests")
    status_dir.mkdir(parents=True, exist_ok=True)
    suffix = "dry_run" if dry_run else "production"
    csv_path = status_dir / f"batch_{suffix}_status.csv"
    json_path = status_dir / f"batch_{suffix}_manifest.json"
    text_path = status_dir / f"batch_{suffix}_report.txt"

    fieldnames = [
        "model",
        "scenario",
        "status",
        "return_code",
        "elapsed_seconds",
        "log_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file_object:
        writer = csv.DictWriter(file_object, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": suffix,
        "configuration": configuration,
        "records": records,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    lines = [
        "QM/QDM BATCH REPORT",
        "=" * 100,
        f"Mode                 : {suffix}",
        f"Models               : {len(configuration['models'])}",
        f"Scenarios            : {', '.join(configuration['scenarios'])}",
        f"Expected combinations: {len(records)}",
    ]
    for status in sorted(counts):
        lines.append(f"{status:21s}: {counts[status]}")
    lines.extend(
        [
            f"Output root          : {configuration['output_root']}",
            f"Status CSV           : {csv_path}",
            f"Manifest             : {json_path}",
        ]
    )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, text_path


def main() -> int:
    arguments = parse_arguments()
    paths = resolve_paths(arguments)
    paths["output_root"].mkdir(parents=True, exist_ok=True)
    paths["log_root"].mkdir(parents=True, exist_ok=True)

    discovered = discover_models(paths["cmip6_root"], paths["era5_root"])
    models = sorted(dict.fromkeys(arguments.models)) if arguments.models else discovered
    missing = [model for model in models if model not in discovered]
    if missing:
        raise FileNotFoundError(
            "Models missing from CMIP6 or ERA5 roots: " + ", ".join(missing)
        )
    if (
        not arguments.models
        and arguments.expected_model_count > 0
        and len(models) != arguments.expected_model_count
    ):
        raise RuntimeError(
            f"Discovered {len(models)} models, expected {arguments.expected_model_count}. "
            "Resolve the model inventory or use --expected-model-count 0 explicitly."
        )

    combinations = [(model, scenario) for model in models for scenario in arguments.scenarios]
    configuration = {
        "models": models,
        "scenarios": list(arguments.scenarios),
        "variables": list(VARIABLES),
        "fit_period": "1959-2014",
        "future_period": "2015-2100",
        "quantile_bounds": [LOWER_QUANTILE, UPPER_QUANTILE],
        "n_quantiles": N_QUANTILES,
        "qdm_window_years": QDM_WINDOW_YEARS,
        "cmip6_root": str(paths["cmip6_root"]),
        "era5_root": str(paths["era5_root"]),
        "output_root": str(paths["output_root"]),
        "engine": str(paths["engine"]),
    }

    print("=" * 100)
    print("QM/QDM FORMAL BATCH")
    print("=" * 100)
    print(f"Engine       : {paths['engine']}")
    print(f"Models       : {len(models)}")
    print(f"Scenarios    : {', '.join(arguments.scenarios)}")
    print(f"Combinations : {len(combinations)}")
    print(f"Output root  : {paths['output_root']}")
    print(f"Mode         : {'DRY RUN' if arguments.dry_run else 'PRODUCTION'}")
    print(f"QC weighting : geodesic China-intersection area ({paths['china_shapefile'].name})")
    print("Note         : correction grid is never clipped; QC statistics are area-weighted")

    records: list[dict[str, object]] = []
    for model, scenario in combinations:
        if (
            not arguments.dry_run
            and not arguments.force
            and combination_complete(paths["output_root"], model, scenario)
        ):
            print(f"SKIPPED COMPLETE: {model} / {scenario}")
            records.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "status": "SKIPPED_COMPLETE",
                    "return_code": 0,
                    "elapsed_seconds": 0.0,
                    "log_file": "",
                }
            )
            write_status_files(
                records=records,
                configuration=configuration,
                dry_run=arguments.dry_run,
            )
            continue

        started = datetime.now()
        return_code, log_path = run_combination(
            engine=paths["engine"],
            cmip6_root=paths["cmip6_root"],
            era5_root=paths["era5_root"],
            output_root=paths["output_root"],
            log_root=paths["log_root"],
            model=model,
            scenario=scenario,
            dry_run=arguments.dry_run,
            china_shapefile=paths["china_shapefile"],
            skip_netcdf=arguments.skip_netcdf,
        )
        elapsed = (datetime.now() - started).total_seconds()
        status = "COMPLETED" if return_code == 0 else "FAILED"
        records.append(
            {
                "model": model,
                "scenario": scenario,
                "status": status,
                "return_code": return_code,
                "elapsed_seconds": round(elapsed, 3),
                "log_file": str(log_path),
            }
        )
        # Persist progress after every combination so an interrupted batch can
        # be audited and safely resumed.
        write_status_files(
            records=records,
            configuration=configuration,
            dry_run=arguments.dry_run,
        )
        if return_code != 0 and arguments.stop_on_error:
            break

    csv_path, json_path, text_path = write_status_files(
        records=records,
        configuration=configuration,
        dry_run=arguments.dry_run,
    )
    failures = [record for record in records if record["status"] == "FAILED"]

    print("\n" + "=" * 100)
    print("BATCH FINISHED")
    print("=" * 100)
    print(f"Completed/skipped: {len(records) - len(failures)}/{len(records)}")
    print(f"Failed           : {len(failures)}")
    print(f"Status CSV       : {csv_path}")
    print(f"Manifest         : {json_path}")
    print(f"Text report      : {text_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
