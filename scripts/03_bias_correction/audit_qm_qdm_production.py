#!/usr/bin/env python
r"""Audit the complete 17-model, 3-scenario QM/QDM production archive.

Place this file in:
    scripts/03_bias_correction

The audit is read-only. Results are written to ``results/audits``.
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
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr


EXPECTED_MODELS = 17
SCENARIOS = ("ssp126", "ssp245", "ssp585")
VARIABLES = ("tas", "rsds", "sfcWind")
METHODS = ("raw", "qm_paper", "qdm_improved")
EXPECTED_MONTHS = 1032
EXPECTED_START = (2015, 1)
EXPECTED_END = (2100, 12)
EXPECTED_BOUNDS = (0.02, 0.98)
EXPECTED_N_QUANTILES = 99
EXPECTED_QDM_WINDOW = 30

EXPECTED_CSV_ROWS = {
    "future_qc": 36,
    "future_change_signal": 36,
    "future_period_change_signal": 108,
}

EXPECTED_CSV_COLUMNS = {
    "future_qc": {
        "model", "scenario", "variable", "month", "method", "qc_region",
        "samples", "tail_rule_trigger_percent", "ratio_fallback_percent",
    },
    "future_change_signal": {
        "model", "scenario", "variable", "month", "method", "signal_unit",
        "qc_region",
    },
    "future_period_change_signal": {
        "model", "scenario", "variable", "period", "period_start_year",
        "period_end_year", "month", "method", "signal_unit", "qc_region",
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit of all formal QM/QDM production outputs."
    )
    parser.add_argument(
        "--production-root",
        type=Path,
        help=(
            "Production root. Default: <project-root>/data/processed/bias_correction."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Audit output directory. Default: <project-root>/results/audits.",
    )
    parser.add_argument(
        "--expected-model-count",
        type=int,
        default=EXPECTED_MODELS,
        help="Expected model count (default: 17; use 0 to disable).",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip full-array finite/min/max checks for a faster structural audit.",
    )
    return parser.parse_args()


def add_issue(
    issues: list[dict[str, str]],
    severity: str,
    model: str = "",
    scenario: str = "",
    variable: str = "",
    check: str = "",
    detail: str = "",
) -> None:
    issues.append(
        {
            "severity": severity,
            "model": model,
            "scenario": scenario,
            "variable": variable,
            "check": check,
            "detail": detail,
        }
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file_object:
        writer = csv.DictWriter(file_object, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file_object:
        reader = csv.DictReader(file_object)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def normalize_unit(value: Any) -> str:
    return str(value or "").lower().replace("**", "^").replace(" ", "")


def unit_is_valid(variable: str, unit: Any) -> bool:
    normalized = normalize_unit(unit)
    if variable == "tas":
        return normalized in {"k", "kelvin"}
    if variable == "rsds":
        return normalized in {
            "wm-2", "w/m2", "wm^-2", "wm^(-2)", "wattm-2", "wattsm-2"
        }
    return normalized in {
        "ms-1", "m/s", "ms^-1", "ms^(-1)", "metersecond-1", "metres-1"
    }


def physical_range_issues(
    variable: str,
    method: str,
    minimum: float,
    maximum: float,
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        found.append(("FAIL", f"{method} has no finite min/max"))
        return found
    if variable == "tas":
        if minimum < 150 or maximum > 400:
            found.append(("FAIL", f"{method} tas range {minimum:.3f}..{maximum:.3f} K"))
        elif minimum < 190 or maximum > 370:
            found.append(("WARNING", f"{method} unusual tas range {minimum:.3f}..{maximum:.3f} K"))
    elif variable == "rsds":
        if minimum < -1e-4 or maximum > 1000:
            found.append(("FAIL", f"{method} rsds range {minimum:.3f}..{maximum:.3f} W m-2"))
        elif maximum > 500:
            found.append(("WARNING", f"{method} high rsds maximum {maximum:.3f} W m-2"))
    elif variable == "sfcWind":
        if minimum < -1e-4 or maximum > 100:
            found.append(("FAIL", f"{method} sfcWind range {minimum:.3f}..{maximum:.3f} m s-1"))
        elif maximum > 50:
            found.append(("WARNING", f"{method} high sfcWind maximum {maximum:.3f} m s-1"))
    return found


def inspect_netcdf(
    path: Path,
    model: str,
    scenario: str,
    variable: str,
    metadata_only: bool,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "scenario": scenario,
        "variable": variable,
        "path": str(path),
        "exists": path.is_file(),
        "file_size_mb": round(path.stat().st_size / 1024**2, 3) if path.is_file() else "",
        "time": "",
        "lat": "",
        "lon": "",
        "start": "",
        "end": "",
        "calendar": "",
        "units": "",
    }
    for method in METHODS:
        row[f"{method}_finite_percent"] = ""
        row[f"{method}_min"] = ""
        row[f"{method}_max"] = ""

    if not path.is_file():
        add_issue(issues, "FAIL", model, scenario, variable, "netcdf_missing", str(path))
        return row

    try:
        with xr.open_dataset(path, decode_times=True) as dataset:
            row["time"] = dataset.sizes.get("time", "")
            row["lat"] = dataset.sizes.get("lat", "")
            row["lon"] = dataset.sizes.get("lon", "")
            missing_methods = [name for name in METHODS if name not in dataset.data_vars]
            extra_methods = [name for name in dataset.data_vars if name not in METHODS]
            if missing_methods:
                add_issue(
                    issues, "FAIL", model, scenario, variable,
                    "missing_data_variables", ", ".join(missing_methods),
                )
            if extra_methods:
                add_issue(
                    issues, "WARNING", model, scenario, variable,
                    "unexpected_data_variables", ", ".join(extra_methods),
                )
            if dataset.sizes.get("time") != EXPECTED_MONTHS:
                add_issue(
                    issues, "FAIL", model, scenario, variable, "time_count",
                    f"{dataset.sizes.get('time')} != {EXPECTED_MONTHS}",
                )
            if dataset.sizes.get("lat", 0) <= 0 or dataset.sizes.get("lon", 0) <= 0:
                add_issue(issues, "FAIL", model, scenario, variable, "grid_size", str(dict(dataset.sizes)))

            if "time" in dataset.coords and dataset.sizes.get("time", 0):
                years = np.asarray(dataset.time.dt.year.values, dtype=int)
                months = np.asarray(dataset.time.dt.month.values, dtype=int)
                start = (int(years[0]), int(months[0]))
                end = (int(years[-1]), int(months[-1]))
                row["start"] = f"{start[0]:04d}-{start[1]:02d}"
                row["end"] = f"{end[0]:04d}-{end[1]:02d}"
                row["calendar"] = str(dataset.time.encoding.get("calendar", dataset.time.attrs.get("calendar", "")))
                month_ids = years * 12 + months - 1
                if start != EXPECTED_START or end != EXPECTED_END:
                    add_issue(
                        issues, "FAIL", model, scenario, variable, "time_bounds",
                        f"{start}..{end}",
                    )
                if len(np.unique(month_ids)) != len(month_ids) or not np.all(np.diff(month_ids) == 1):
                    add_issue(issues, "FAIL", model, scenario, variable, "time_sequence", "duplicate or missing months")

            attributes = dataset.attrs
            expected_attributes = {
                "model": model,
                "scenario": scenario,
                "variable": variable,
                "historical_fit_period": "1959-2014",
                "future_period": "2015-2100",
                "quantile_bounds": "Q02-Q98",
            }
            for key, expected in expected_attributes.items():
                actual = str(attributes.get(key, ""))
                if actual != expected:
                    add_issue(
                        issues, "FAIL", model, scenario, variable,
                        f"attribute_{key}", f"{actual!r} != {expected!r}",
                    )
            if int(attributes.get("quantile_nodes", -1)) != EXPECTED_N_QUANTILES:
                add_issue(issues, "FAIL", model, scenario, variable, "quantile_nodes", str(attributes.get("quantile_nodes")))
            if int(attributes.get("qdm_future_cdf_window_years", -1)) != EXPECTED_QDM_WINDOW:
                add_issue(issues, "FAIL", model, scenario, variable, "qdm_window", str(attributes.get("qdm_future_cdf_window_years")))

            available = [name for name in METHODS if name in dataset.data_vars]
            units = [str(dataset[name].attrs.get("units", "")) for name in available]
            row["units"] = units[0] if units else ""
            if not units or any(unit != units[0] for unit in units):
                add_issue(issues, "FAIL", model, scenario, variable, "unit_consistency", repr(units))
            elif not unit_is_valid(variable, units[0]):
                add_issue(issues, "FAIL", model, scenario, variable, "unit_value", units[0])

            if not metadata_only:
                for method in available:
                    values = np.asarray(dataset[method].values)
                    finite = np.isfinite(values)
                    finite_percent = 100.0 * float(finite.sum()) / float(values.size)
                    minimum = float(np.nanmin(values)) if finite.any() else math.nan
                    maximum = float(np.nanmax(values)) if finite.any() else math.nan
                    row[f"{method}_finite_percent"] = finite_percent
                    row[f"{method}_min"] = minimum
                    row[f"{method}_max"] = maximum
                    if finite_percent < 99.0:
                        add_issue(
                            issues, "FAIL", model, scenario, variable,
                            f"{method}_finite", f"{finite_percent:.6f}%",
                        )
                    elif finite_percent < 100.0:
                        add_issue(
                            issues, "WARNING", model, scenario, variable,
                            f"{method}_finite", f"{finite_percent:.6f}%",
                        )
                    for severity, detail in physical_range_issues(variable, method, minimum, maximum):
                        add_issue(issues, severity, model, scenario, variable, "physical_range", detail)
    except Exception as error:
        add_issue(issues, "FAIL", model, scenario, variable, "netcdf_open", repr(error))
    return row


def inspect_csv(
    path: Path,
    csv_type: str,
    model: str,
    scenario: str,
    variable: str,
    issues: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    inventory = {
        "model": model,
        "scenario": scenario,
        "variable": variable,
        "csv_type": csv_type,
        "path": str(path),
        "exists": path.is_file(),
        "rows": "",
        "columns": "",
        "max_tail_percent": "",
        "max_fallback_percent": "",
    }
    if not path.is_file():
        add_issue(issues, "FAIL", model, scenario, variable, f"{csv_type}_missing", str(path))
        return inventory, []
    try:
        rows, columns = read_csv_rows(path)
        inventory["rows"] = len(rows)
        inventory["columns"] = len(columns)
        if len(rows) != EXPECTED_CSV_ROWS[csv_type]:
            add_issue(
                issues, "FAIL", model, scenario, variable, f"{csv_type}_rows",
                f"{len(rows)} != {EXPECTED_CSV_ROWS[csv_type]}",
            )
        missing_columns = EXPECTED_CSV_COLUMNS[csv_type] - set(columns)
        if missing_columns:
            add_issue(
                issues, "FAIL", model, scenario, variable, f"{csv_type}_columns",
                ", ".join(sorted(missing_columns)),
            )
        for row in rows:
            if row.get("model") != model or row.get("scenario") != scenario or row.get("variable") != variable:
                add_issue(
                    issues, "FAIL", model, scenario, variable, f"{csv_type}_identity",
                    f"row has {row.get('model')}/{row.get('scenario')}/{row.get('variable')}",
                )
                break
        if csv_type == "future_qc" and rows:
            tails = [safe_float(row.get("tail_rule_trigger_percent")) for row in rows]
            fallbacks = [safe_float(row.get("ratio_fallback_percent")) for row in rows]
            max_tail = float(np.nanmax(tails))
            max_fallback = float(np.nanmax(fallbacks))
            inventory["max_tail_percent"] = max_tail
            inventory["max_fallback_percent"] = max_fallback
            if max_fallback > 1.0:
                add_issue(
                    issues, "WARNING", model, scenario, variable,
                    "fallback_percent", f"maximum {max_fallback:.6f}%",
                )
            if max_tail > 50.0:
                add_issue(
                    issues, "WARNING", model, scenario, variable,
                    "tail_trigger_percent", f"maximum {max_tail:.6f}%",
                )
        return inventory, rows
    except Exception as error:
        add_issue(issues, "FAIL", model, scenario, variable, f"{csv_type}_read", repr(error))
        return inventory, []


def inspect_manifest(
    path: Path,
    model: str,
    scenario: str,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    row = {
        "model": model,
        "scenario": scenario,
        "path": str(path),
        "exists": path.is_file(),
        "historical_fit_period": "",
        "future_period": "",
        "bounds": "",
        "n_quantiles": "",
        "qdm_window_years": "",
        "qc_region": "",
        "variable_count": "",
    }
    if not path.is_file():
        add_issue(issues, "FAIL", model, scenario, check="manifest_missing", detail=str(path))
        return row
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        row.update(
            {
                "historical_fit_period": data.get("historical_fit_period"),
                "future_period": data.get("future_period"),
                "bounds": data.get("bounds"),
                "n_quantiles": data.get("n_quantiles"),
                "qdm_window_years": data.get("qdm_future_cdf_window_years"),
                "qc_region": data.get("qc_region"),
                "variable_count": len(data.get("variables", [])),
            }
        )
        checks = {
            "model": model,
            "scenario": scenario,
            "historical_fit_period": [1959, 2014],
            "future_period": [2015, 2100],
            "bounds": list(EXPECTED_BOUNDS),
            "n_quantiles": EXPECTED_N_QUANTILES,
            "qdm_future_cdf_window_years": EXPECTED_QDM_WINDOW,
            "qc_region": "rectangular_domain",
        }
        for key, expected in checks.items():
            if data.get(key) != expected:
                add_issue(
                    issues, "FAIL", model, scenario, check=f"manifest_{key}",
                    detail=f"{data.get(key)!r} != {expected!r}",
                )
        variables = data.get("variables", [])
        names = [item.get("variable") for item in variables if isinstance(item, dict)]
        if sorted(names) != sorted(VARIABLES):
            add_issue(issues, "FAIL", model, scenario, check="manifest_variables", detail=repr(names))
    except Exception as error:
        add_issue(issues, "FAIL", model, scenario, check="manifest_read", detail=repr(error))
    return row


def main() -> int:
    arguments = parse_arguments()
    script_dir = Path(__file__).resolve().parent
    production_root = (
        arguments.production_root.resolve()
        if arguments.production_root
        else get_path("data_processed_bias_correction")
    )
    output_root = (
        arguments.output_root.resolve()
        if arguments.output_root
        else get_path("results_audits")
    )
    if not production_root.is_dir():
        raise FileNotFoundError(f"Production root not found: {production_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    model_names = sorted(path.name for path in production_root.iterdir() if path.is_dir())
    issues: list[dict[str, str]] = []
    if arguments.expected_model_count > 0 and len(model_names) != arguments.expected_model_count:
        add_issue(
            issues, "FAIL", check="model_count",
            detail=f"{len(model_names)} != {arguments.expected_model_count}: {model_names}",
        )

    manifest_rows: list[dict[str, Any]] = []
    netcdf_rows: list[dict[str, Any]] = []
    csv_inventory: list[dict[str, Any]] = []
    combined_qc: list[dict[str, str]] = []
    combined_signal: list[dict[str, str]] = []
    combined_period_signal: list[dict[str, str]] = []

    expected_combinations = len(model_names) * len(SCENARIOS)
    counter = 0
    for model in model_names:
        for scenario in SCENARIOS:
            counter += 1
            destination = production_root / model / scenario
            print(f"[{counter:02d}/{expected_combinations:02d}] {model} / {scenario}")
            if not destination.is_dir():
                add_issue(issues, "FAIL", model, scenario, check="combination_directory", detail=str(destination))
            manifest_rows.append(
                inspect_manifest(destination / "future_run_manifest.json", model, scenario, issues)
            )
            combination_grids: list[tuple[str, Any, Any]] = []
            for variable in VARIABLES:
                netcdf_path = destination / (
                    f"corrected_{model}_{scenario}_{variable}_"
                    "Q02-Q98_MW30_201501-210012.nc"
                )
                netcdf_row = inspect_netcdf(
                    netcdf_path, model, scenario, variable,
                    arguments.metadata_only, issues,
                )
                netcdf_rows.append(netcdf_row)
                combination_grids.append((variable, netcdf_row["lat"], netcdf_row["lon"]))

                csv_specs = (
                    ("future_qc", destination / f"future_qc_{variable}.csv", combined_qc),
                    ("future_change_signal", destination / f"future_change_signal_{variable}.csv", combined_signal),
                    (
                        "future_period_change_signal",
                        destination / f"future_period_change_signal_{variable}.csv",
                        combined_period_signal,
                    ),
                )
                for csv_type, csv_path, combined in csv_specs:
                    inventory, rows = inspect_csv(
                        csv_path, csv_type, model, scenario, variable, issues
                    )
                    csv_inventory.append(inventory)
                    combined.extend(rows)

            grid_shapes = {(lat, lon) for _, lat, lon in combination_grids if lat != "" and lon != ""}
            if len(grid_shapes) > 1:
                add_issue(
                    issues, "FAIL", model, scenario, check="variable_grid_consistency",
                    detail=repr(combination_grids),
                )

    manifest_fields = [
        "model", "scenario", "path", "exists", "historical_fit_period",
        "future_period", "bounds", "n_quantiles", "qdm_window_years",
        "qc_region", "variable_count",
    ]
    netcdf_fields = [
        "model", "scenario", "variable", "path", "exists", "file_size_mb",
        "time", "lat", "lon", "start", "end", "calendar", "units",
    ] + [field for method in METHODS for field in (
        f"{method}_finite_percent", f"{method}_min", f"{method}_max"
    )]
    csv_inventory_fields = [
        "model", "scenario", "variable", "csv_type", "path", "exists", "rows",
        "columns", "max_tail_percent", "max_fallback_percent",
    ]
    issue_fields = ["severity", "model", "scenario", "variable", "check", "detail"]

    write_csv(output_root / "manifest_inventory.csv", manifest_rows, manifest_fields)
    write_csv(output_root / "netcdf_inventory.csv", netcdf_rows, netcdf_fields)
    write_csv(output_root / "csv_inventory.csv", csv_inventory, csv_inventory_fields)
    write_csv(output_root / "audit_issues.csv", issues, issue_fields)

    if combined_qc:
        write_csv(output_root / "future_qc_all.csv", combined_qc, combined_qc[0].keys())
    if combined_signal:
        write_csv(
            output_root / "future_change_signal_all.csv",
            combined_signal,
            combined_signal[0].keys(),
        )
    if combined_period_signal:
        write_csv(
            output_root / "future_period_change_signal_all.csv",
            combined_period_signal,
            combined_period_signal[0].keys(),
        )

    severity_counts = Counter(issue["severity"] for issue in issues)
    expected_netcdf = len(model_names) * len(SCENARIOS) * len(VARIABLES)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "production_root": str(production_root),
        "audit_output_root": str(output_root),
        "metadata_only": arguments.metadata_only,
        "models": len(model_names),
        "scenarios": len(SCENARIOS),
        "combinations": len(model_names) * len(SCENARIOS),
        "expected_netcdf_files": expected_netcdf,
        "existing_netcdf_files": sum(bool(row["exists"]) for row in netcdf_rows),
        "existing_manifests": sum(bool(row["exists"]) for row in manifest_rows),
        "existing_csv_files": sum(bool(row["exists"]) for row in csv_inventory),
        "combined_qc_rows": len(combined_qc),
        "combined_change_signal_rows": len(combined_signal),
        "combined_period_signal_rows": len(combined_period_signal),
        "failures": severity_counts.get("FAIL", 0),
        "warnings": severity_counts.get("WARNING", 0),
        "status": "PASS" if severity_counts.get("FAIL", 0) == 0 else "FAIL",
    }
    (output_root / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_lines = [
        "QM/QDM FORMAL PRODUCTION AUDIT",
        "=" * 100,
        f"Status                       : {summary['status']}",
        f"Models                       : {summary['models']}",
        f"Scenarios                    : {summary['scenarios']}",
        f"Model/scenario combinations  : {summary['combinations']}",
        f"NetCDF files                 : {summary['existing_netcdf_files']}/{summary['expected_netcdf_files']}",
        f"Run manifests                : {summary['existing_manifests']}/{summary['combinations']}",
        f"Diagnostic CSV files         : {summary['existing_csv_files']}/{expected_netcdf * 3}",
        f"Combined QC rows             : {summary['combined_qc_rows']}",
        f"Combined signal rows         : {summary['combined_change_signal_rows']}",
        f"Combined period-signal rows  : {summary['combined_period_signal_rows']}",
        f"Failures                     : {summary['failures']}",
        f"Warnings                     : {summary['warnings']}",
        f"Metadata-only                : {summary['metadata_only']}",
        f"Output root                  : {output_root}",
    ]
    if severity_counts.get("WARNING", 0):
        report_lines.append("Warnings require review but do not automatically invalidate the archive.")
    (output_root / "audit_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 100)
    print("FORMAL PRODUCTION AUDIT COMPLETED")
    print("=" * 100)
    for line in report_lines[2:]:
        print(line)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
