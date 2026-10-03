#!/usr/bin/env python
r"""Audit the formal multi-model PV/WPD energy-metrics archive.

Place this script in::

    scripts/04_energy_metrics

Default input::

    data/processed/energy_metrics

The audit is read-only with respect to production results.  Reports are written
to ``results/audits``.
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
from collections import Counter
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
BASELINES = ("qm_historical", "era5_reference")
EXPECTED_MODELS = 17
ENERGY_VARIABLES = (
    "tas_c",
    "rsds",
    "sfcWind_10m",
    "tcell",
    "performance_ratio",
    "pvpot",
    "roughness_exponent",
    "wind_speed_hub",
    "wpd",
)
COMPLEMENTARITY_VARIABLES = (
    "spearman_rho",
    "spearman_p_value",
    "valid_n",
)
PERIODS = ("historical", "mid_century", "late_century")
DEFINITIONS = (
    "monthly_full",
    "monthly_climatology",
    "seasonal_full",
    "seasonal_climatology",
)
FORMULA_SAMPLE_INDICES = (0, 120, 251, 252, 552, 792, 1032, 1283)
AIR_DENSITY = 1.225
PV_GAMMA = -0.005
PV_T_STC_C = 25.0
PV_C1_C = 4.3
PV_C2 = 0.943
PV_C3_M2_W = 0.028
PV_C4_C_S_M = -1.528


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit file completeness, dates, physical values, formulae, "
            "historical consistency, complementarity, and transition warnings."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Default: <project-root>/data/processed/energy_metrics",
    )
    parser.add_argument("--models", nargs="+", help="Optional model subset.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
    )
    parser.add_argument("--expected-models", type=int, default=EXPECTED_MODELS)
    parser.add_argument(
        "--formula-atol",
        type=float,
        default=1.0e-4,
        help="Absolute tolerance for sampled formula reconstruction.",
    )
    parser.add_argument(
        "--formula-rtol",
        type=float,
        default=1.0e-5,
        help="Relative tolerance for sampled formula reconstruction.",
    )
    parser.add_argument(
        "--history-atol",
        type=float,
        default=1.0e-6,
        help="Absolute tolerance for shared historical segments.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Skip full-array pvpot/WPD/performance-ratio extrema scans. "
            "Structural, formula, history, and complementarity checks still run."
        ),
    )
    return parser.parse_args()


def decoder():
    try:
        return xr.coders.CFDatetimeCoder(use_cftime=True)
    except AttributeError:  # pragma: no cover
        return True


def open_netcdf(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=decoder())


def discover_models(
    root: Path,
    requested: list[str] | None,
    scenarios: list[str],
    expected_models: int,
) -> list[str]:
    if requested:
        models = sorted(dict.fromkeys(requested))
    else:
        models = sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir()
            and path.name != "production_audit"
            and (path / "historical").is_dir()
            and all((path / scenario).is_dir() for scenario in scenarios)
        )
    if not models:
        raise FileNotFoundError(f"No complete model directory found below {root}")
    if requested is None and len(models) != expected_models:
        raise ValueError(
            f"Expected {expected_models} models; found {len(models)}: "
            + ", ".join(models)
        )
    return models


def append_check(
    rows: list[dict[str, object]],
    severity: str,
    category: str,
    model: str = "",
    scenario: str = "",
    method: str = "",
    file: str = "",
    message: str = "",
    value: object = "",
) -> None:
    rows.append(
        {
            "severity": severity,
            "category": category,
            "model": model,
            "scenario": scenario,
            "method": method,
            "file": file,
            "message": message,
            "value": value,
        }
    )


def require_file(
    path: Path,
    rows: list[dict[str, object]],
    category: str,
    model: str,
    scenario: str,
    method: str = "",
) -> bool:
    if path.is_file():
        return True
    append_check(
        rows,
        "FAIL",
        category,
        model,
        scenario,
        method,
        str(path),
        "Required file is missing.",
    )
    return False


def time_keys(dataset: xr.Dataset) -> np.ndarray:
    years = np.asarray(dataset.time.dt.year.values, dtype=np.int64)
    months = np.asarray(dataset.time.dt.month.values, dtype=np.int64)
    return years * 100 + months


def expected_keys(start_year: int, start_month: int, count: int) -> np.ndarray:
    values = []
    year, month = start_year, start_month
    for _ in range(count):
        values.append(year * 100 + month)
        month += 1
        if month == 13:
            year += 1
            month = 1
    return np.asarray(values, dtype=np.int64)


def max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return math.inf
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return math.inf
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(a[finite] - b[finite])))


def sampled(dataset: xr.Dataset) -> xr.Dataset:
    count = int(dataset.sizes["time"])
    indices = sorted({index for index in FORMULA_SAMPLE_INDICES if index < count})
    return dataset.isel(time=indices).load()


def audit_energy_dataset(
    path: Path,
    model: str,
    scenario: str,
    method: str,
    rows: list[dict[str, object]],
    arguments: argparse.Namespace,
    historical_only: bool = False,
) -> None:
    try:
        with open_netcdf(path) as dataset:
            expected_count = 252 if historical_only else 1284
            expected = expected_keys(1994, 1, expected_count)
            missing_variables = [
                variable for variable in ENERGY_VARIABLES if variable not in dataset
            ]
            if missing_variables:
                raise ValueError(
                    "Missing energy variables: " + ", ".join(missing_variables)
                )
            required_dimensions = {"time", "lat", "lon"}
            if not required_dimensions.issubset(dataset.dims):
                raise ValueError(
                    f"Missing dimensions; found {tuple(dataset.dims)}"
                )
            keys = time_keys(dataset)
            if not np.array_equal(keys, expected):
                raise ValueError(
                    f"Time axis is not complete 1994-01 with {expected_count} months; "
                    f"found {keys.size}, first={keys[0] if keys.size else None}, "
                    f"last={keys[-1] if keys.size else None}"
                )

            subset = sampled(dataset)
            expected_tcell = (
                PV_C1_C
                + PV_C2 * subset["tas_c"]
                + PV_C3_M2_W * subset["rsds"]
                + PV_C4_C_S_M * subset["sfcWind_10m"]
            )
            expected_ratio = 1.0 + PV_GAMMA * (subset["tcell"] - PV_T_STC_C)
            expected_pv = subset["performance_ratio"] * subset["rsds"]
            density = float(
                dataset.attrs.get(
                    "air_density_kg_m3",
                    dataset["wpd"].attrs.get("air_density_kg_m3", AIR_DENSITY),
                )
            )
            expected_wpd = 0.5 * density * subset["wind_speed_hub"] ** 3
            formulae = {
                "tcell": (subset["tcell"], expected_tcell),
                "performance_ratio": (
                    subset["performance_ratio"],
                    expected_ratio,
                ),
                "pvpot": (subset["pvpot"], expected_pv),
                "wpd": (subset["wpd"], expected_wpd),
            }
            for name, (actual, reconstructed) in formulae.items():
                actual_values = np.asarray(actual.values, dtype=np.float64)
                reconstructed_values = np.asarray(
                    reconstructed.values, dtype=np.float64
                )
                difference = max_abs_difference(
                    actual_values, reconstructed_values
                )
                if not np.allclose(
                    actual_values,
                    reconstructed_values,
                    rtol=arguments.formula_rtol,
                    atol=arguments.formula_atol,
                    equal_nan=True,
                ):
                    append_check(
                        rows,
                        "FAIL",
                        "formula_consistency",
                        model,
                        scenario,
                        method,
                        str(path),
                        f"{name} does not match its reconstruction.",
                        difference,
                    )

            if not arguments.quick:
                for variable in ("performance_ratio", "pvpot", "wpd"):
                    data = dataset[variable]
                    minimum = float(data.min(skipna=True).values)
                    maximum = float(data.max(skipna=True).values)
                    if not np.isfinite(minimum) or not np.isfinite(maximum):
                        append_check(
                            rows,
                            "FAIL",
                            "physical_range",
                            model,
                            scenario,
                            method,
                            str(path),
                            f"{variable} has no finite global extrema.",
                        )
                    if variable in ("pvpot", "wpd") and minimum < -1.0e-5:
                        append_check(
                            rows,
                            "FAIL",
                            "physical_range",
                            model,
                            scenario,
                            method,
                            str(path),
                            f"{variable} contains negative values.",
                            minimum,
                        )
                    if variable == "performance_ratio" and (
                        minimum < 0.0 or maximum > 2.0
                    ):
                        append_check(
                            rows,
                            "WARN",
                            "physical_range",
                            model,
                            scenario,
                            method,
                            str(path),
                            "Performance ratio extends outside the broad [0, 2] review range.",
                            f"min={minimum:.6g}; max={maximum:.6g}",
                        )
    except Exception as error:
        append_check(
            rows,
            "FAIL",
            "energy_netcdf",
            model,
            scenario,
            method,
            str(path),
            str(error),
            traceback.format_exc(),
        )


def audit_complementarity_dataset(
    path: Path,
    model: str,
    scenario: str,
    method: str,
    rows: list[dict[str, object]],
    historical_only: bool = False,
) -> None:
    try:
        with open_netcdf(path) as dataset:
            missing = [
                variable
                for variable in COMPLEMENTARITY_VARIABLES
                if variable not in dataset
            ]
            if missing:
                raise ValueError(
                    "Missing complementarity variables: " + ", ".join(missing)
                )
            periods = [str(value) for value in dataset.period.values.tolist()]
            definitions = [
                str(value) for value in dataset.definition.values.tolist()
            ]
            expected_periods = ["historical"] if historical_only else list(PERIODS)
            if set(periods) != set(expected_periods):
                raise ValueError(
                    f"Unexpected periods: {periods}; expected {expected_periods}"
                )
            if set(definitions) != set(DEFINITIONS):
                raise ValueError(
                    f"Unexpected definitions: {definitions}; expected {list(DEFINITIONS)}"
                )
            rho = np.asarray(dataset["spearman_rho"].values, dtype=np.float64)
            finite_rho = rho[np.isfinite(rho)]
            if finite_rho.size and (
                float(finite_rho.min()) < -1.000001
                or float(finite_rho.max()) > 1.000001
            ):
                append_check(
                    rows,
                    "FAIL",
                    "complementarity_range",
                    model,
                    scenario,
                    method,
                    str(path),
                    "Spearman rho is outside [-1, 1].",
                    f"min={finite_rho.min()}; max={finite_rho.max()}",
                )
            p_value = np.asarray(
                dataset["spearman_p_value"].values, dtype=np.float64
            )
            finite_p = p_value[np.isfinite(p_value)]
            if finite_p.size and (
                float(finite_p.min()) < -1.0e-8
                or float(finite_p.max()) > 1.000001
            ):
                append_check(
                    rows,
                    "FAIL",
                    "complementarity_range",
                    model,
                    scenario,
                    method,
                    str(path),
                    "Spearman p-value is outside [0, 1].",
                    f"min={finite_p.min()}; max={finite_p.max()}",
                )
            valid_n = np.asarray(dataset["valid_n"].values, dtype=np.float64)
            if np.nanmin(valid_n) < 0 or np.nanmax(valid_n) > 252:
                append_check(
                    rows,
                    "FAIL",
                    "complementarity_samples",
                    model,
                    scenario,
                    method,
                    str(path),
                    "valid_n is outside [0, 252].",
                    f"min={np.nanmin(valid_n)}; max={np.nanmax(valid_n)}",
                )
    except Exception as error:
        append_check(
            rows,
            "FAIL",
            "complementarity_netcdf",
            model,
            scenario,
            method,
            str(path),
            str(error),
            traceback.format_exc(),
        )


def read_history(path: Path) -> dict[str, np.ndarray]:
    with open_netcdf(path) as dataset:
        keys = time_keys(dataset)
        indices = np.flatnonzero((keys >= 199401) & (keys <= 201412))
        if indices.size != 252:
            raise ValueError(f"Expected 252 historical months in {path}")
        return {
            variable: np.asarray(
                dataset[variable].isel(time=indices).values, dtype=np.float32
            )
            for variable in ENERGY_VARIABLES
        }


def compare_history(
    reference: dict[str, np.ndarray],
    candidate_path: Path,
    model: str,
    scenario: str,
    method: str,
    rows: list[dict[str, object]],
    tolerance: float,
) -> None:
    try:
        candidate = read_history(candidate_path)
        for variable in ENERGY_VARIABLES:
            difference = max_abs_difference(reference[variable], candidate[variable])
            if not np.isfinite(difference) or difference > tolerance:
                append_check(
                    rows,
                    "FAIL",
                    "shared_historical_baseline",
                    model,
                    scenario,
                    method,
                    str(candidate_path),
                    f"Historical {variable} differs from the model reference.",
                    difference,
                )
    except Exception as error:
        append_check(
            rows,
            "FAIL",
            "shared_historical_baseline",
            model,
            scenario,
            method,
            str(candidate_path),
            str(error),
            traceback.format_exc(),
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def check_csv_rows(
    path: Path,
    expected: int,
    model: str,
    scenario: str,
    rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    try:
        content = read_csv(path)
        if len(content) != expected:
            append_check(
                rows,
                "FAIL",
                "csv_row_count",
                model,
                scenario,
                file=str(path),
                message=f"Expected {expected} rows; found {len(content)}.",
                value=len(content),
            )
        return content
    except Exception as error:
        append_check(
            rows,
            "FAIL",
            "csv_read",
            model,
            scenario,
            file=str(path),
            message=str(error),
        )
        return []


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def bool_from_csv(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def audit_model(
    root: Path,
    model: str,
    scenarios: list[str],
    rows: list[dict[str, object]],
    transition_warnings: list[dict[str, object]],
    arguments: argparse.Namespace,
) -> None:
    historical_root = root / model / "historical"
    historical_manifest = historical_root / "historical_run_manifest.json"
    require_file(
        historical_manifest, rows, "manifest", model, "historical"
    )
    historical_csv_specs = (
        ("historical_baseline_comparison.csv", 16),
        ("historical_baseline_complementarity_summary.csv", 8),
    )
    for filename, expected_rows in historical_csv_specs:
        path = historical_root / filename
        if require_file(path, rows, "csv", model, "historical"):
            check_csv_rows(path, expected_rows, model, "historical", rows)

    for baseline in BASELINES:
        energy_path = historical_root / (
            f"historical_energy_{model}_{baseline}_199401-201412.nc"
        )
        comp_path = historical_root / (
            f"historical_complementarity_{model}_{baseline}.nc"
        )
        if require_file(
            energy_path, rows, "energy_file", model, "historical", baseline
        ):
            audit_energy_dataset(
                energy_path,
                model,
                "historical",
                baseline,
                rows,
                arguments,
                historical_only=True,
            )
        if require_file(
            comp_path,
            rows,
            "complementarity_file",
            model,
            "historical",
            baseline,
        ):
            audit_complementarity_dataset(
                comp_path,
                model,
                "historical",
                baseline,
                rows,
                historical_only=True,
            )

    reference_path = root / model / scenarios[0] / (
        f"energy_monthly_{model}_{scenarios[0]}_paper_qm_199401-210012.nc"
    )
    historical_reference = None
    if reference_path.is_file():
        try:
            historical_reference = read_history(reference_path)
        except Exception as error:
            append_check(
                rows,
                "FAIL",
                "shared_historical_baseline",
                model,
                scenarios[0],
                "paper_qm",
                str(reference_path),
                str(error),
            )

    scenario_csv_specs = (
        ("energy_qc_summary.csv", 18),
        ("energy_period_summary.csv", 60),
        ("hub_height_sensitivity.csv", 18),
        ("complementarity_summary.csv", 24),
        ("historical_future_transition.csv", 1848),
    )
    for scenario in scenarios:
        scenario_root = root / model / scenario
        manifest_path = scenario_root / "energy_run_manifest.json"
        require_file(manifest_path, rows, "manifest", model, scenario)
        for filename, expected_rows in scenario_csv_specs:
            path = scenario_root / filename
            if not require_file(path, rows, "csv", model, scenario):
                continue
            content = check_csv_rows(path, expected_rows, model, scenario, rows)
            if filename == "historical_future_transition.csv":
                for row in content:
                    if bool_from_csv(row.get("absolute_z_gt_3_warning", "")):
                        transition_warnings.append(dict(row))

        for method in METHODS:
            energy_path = scenario_root / (
                f"energy_monthly_{model}_{scenario}_{method}_199401-210012.nc"
            )
            comp_path = scenario_root / (
                f"complementarity_{model}_{scenario}_{method}.nc"
            )
            energy_exists = require_file(
                energy_path, rows, "energy_file", model, scenario, method
            )
            if energy_exists:
                audit_energy_dataset(
                    energy_path,
                    model,
                    scenario,
                    method,
                    rows,
                    arguments,
                )
                if historical_reference is not None:
                    compare_history(
                        historical_reference,
                        energy_path,
                        model,
                        scenario,
                        method,
                        rows,
                        arguments.history_atol,
                    )
            if require_file(
                comp_path,
                rows,
                "complementarity_file",
                model,
                scenario,
                method,
            ):
                audit_complementarity_dataset(
                    comp_path, model, scenario, method, rows
                )


def audit_aggregate_csvs(
    tables_root: Path,
    model_count: int,
    scenario_count: int,
    rows: list[dict[str, object]],
) -> None:
    combinations = model_count * scenario_count
    specifications = {
        "combined_energy_qc_summary.csv": 18 * combinations,
        "combined_energy_period_summary.csv": 60 * combinations,
        "combined_hub_height_sensitivity.csv": 18 * combinations,
        "combined_complementarity_summary.csv": 24 * combinations,
        "combined_historical_future_transition.csv": 1848 * combinations,
        "combined_historical_baseline_comparison.csv": 16 * model_count,
        "combined_historical_baseline_complementarity_summary.csv": 8 * model_count,
    }
    for filename, expected in specifications.items():
        path = tables_root / filename
        if require_file(path, rows, "aggregate_csv", "ALL", "ALL"):
            check_csv_rows(path, expected, "ALL", "ALL", rows)


def main() -> int:
    arguments = parse_arguments()
    root = (
        arguments.root.expanduser().resolve()
        if arguments.root
        else get_path("data_processed_energy_metrics")
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Production root does not exist: {root}")
    models = discover_models(
        root,
        arguments.models,
        list(arguments.scenarios),
        arguments.expected_models,
    )
    print("=" * 100)
    print("FORMAL ENERGY METRICS PRODUCTION AUDIT")
    print("=" * 100)
    print(f"Input root   : {root}")
    print(f"Models       : {len(models)}")
    print(f"Scenarios    : {', '.join(arguments.scenarios)}")
    print(f"Combinations : {len(models) * len(arguments.scenarios)}")
    print(f"Mode         : {'quick' if arguments.quick else 'full'}")

    rows: list[dict[str, object]] = []
    transition_warnings: list[dict[str, object]] = []
    for index, model in enumerate(models, start=1):
        print(f"[{index:02d}/{len(models):02d}] Auditing {model} ...")
        audit_model(
            root,
            model,
            list(arguments.scenarios),
            rows,
            transition_warnings,
            arguments,
        )
    audit_aggregate_csvs(
        get_path("results_tables"),
        len(models),
        len(arguments.scenarios),
        rows,
    )

    expected_netcdf = len(models) * 4 + len(models) * len(arguments.scenarios) * 4
    actual_netcdf = len(
        [path for path in root.rglob("*.nc") if "production_audit" not in path.parts]
    )
    if actual_netcdf != expected_netcdf:
        append_check(
            rows,
            "WARN" if actual_netcdf > expected_netcdf else "FAIL",
            "archive_file_count",
            "ALL",
            "ALL",
            message=f"Expected {expected_netcdf} NetCDF files; found {actual_netcdf}.",
            value=actual_netcdf,
        )

    warning_counter = Counter(
        (
            str(row.get("model", "")),
            str(row.get("scenario", "")),
            str(row.get("method", "")),
            str(row.get("variable", "")),
            str(row.get("month", "")),
        )
        for row in transition_warnings
    )
    failures = sum(row["severity"] == "FAIL" for row in rows)
    warnings = sum(row["severity"] == "WARN" for row in rows)
    status = "PASS" if failures == 0 else "FAIL"

    audit_root = get_path("results_audits")
    audit_root.mkdir(parents=True, exist_ok=True)
    checks_path = audit_root / "energy_production_audit_checks.csv"
    transition_path = audit_root / "energy_transition_z_gt_3_warnings.csv"
    summary_path = audit_root / "energy_production_audit_summary.json"
    report_path = audit_root / "energy_production_audit_report.txt"
    write_csv(checks_path, rows)
    write_csv(transition_path, transition_warnings)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_root": str(root),
        "models": models,
        "model_count": len(models),
        "scenarios": list(arguments.scenarios),
        "combination_count": len(models) * len(arguments.scenarios),
        "expected_netcdf": expected_netcdf,
        "actual_netcdf": actual_netcdf,
        "failures": failures,
        "warnings": warnings,
        "transition_z_gt_3_rows": len(transition_warnings),
        "transition_warning_groups": [
            {
                "model": key[0],
                "scenario": key[1],
                "method": key[2],
                "variable": key[3],
                "month": key[4],
                "rows": count,
            }
            for key, count in sorted(warning_counter.items())
        ],
        "quick_mode": bool(arguments.quick),
        "checks_csv": str(checks_path),
        "transition_warnings_csv": str(transition_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_lines = [
        "FORMAL ENERGY METRICS PRODUCTION AUDIT",
        "=" * 80,
        f"Status                    : {status}",
        f"Models                    : {len(models)}",
        f"Scenarios                 : {len(arguments.scenarios)}",
        f"Combinations              : {len(models) * len(arguments.scenarios)}",
        f"NetCDF files              : {actual_netcdf}/{expected_netcdf}",
        f"Failures                  : {failures}",
        f"Audit warnings            : {warnings}",
        f"Transition |z| > 3 rows   : {len(transition_warnings)}",
        f"Quick mode                : {arguments.quick}",
        "",
        "A transition |z| > 3 row is a diagnostic warning, not an automatic failure.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("=" * 100)
    print("FORMAL ENERGY METRICS PRODUCTION AUDIT COMPLETED")
    print("=" * 100)
    print(f"Status                  : {status}")
    print(f"NetCDF files            : {actual_netcdf}/{expected_netcdf}")
    print(f"Failures                : {failures}")
    print(f"Audit warnings          : {warnings}")
    print(f"Transition |z| > 3 rows : {len(transition_warnings)}")
    print(f"Checks CSV              : {checks_path}")
    print(f"Transition warnings     : {transition_path}")
    print(f"Summary                 : {summary_path}")
    print(f"Text report             : {report_path}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
