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
import sys
from pathlib import Path
from types import SimpleNamespace

import qm_qdm_holdout_smoke_test as core


DEFAULT_MODELS = (
    "CAS-ESM2-0",
    "ACCESS-CM2",
    "FGOALS-f3-L",
)
DEFAULT_VARIABLES = ("tas", "rsds", "sfcWind")
DEFAULT_TEMPERATURE_MODES = ("ratio", "additive")
SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "archive" / "abandoned_pilots" / "qm_qdm_holdout_pilot_3x3"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_combined_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the historical holdout QM/QDM pilot for three representative "
            "CMIP6 models and three variables."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model names; defaults to CAS-ESM2-0 ACCESS-CM2 FGOALS-f3-L.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=tuple(core.EXPECTED_UNITS),
        default=list(DEFAULT_VARIABLES),
    )
    parser.add_argument("--cmip6-root", type=Path, default=core.CMIP6_ROOT)
    parser.add_argument("--era5-root", type=Path, default=core.ERA5_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--bounds",
        action="append",
        default=None,
        metavar="LOWER,UPPER",
        help="Repeat to compare bounds; defaults to 0.02,0.98 and 0.05,0.95.",
    )
    parser.add_argument("--n-quantiles", type=int, default=99)
    parser.add_argument(
        "--temperature-modes",
        nargs="+",
        choices=("ratio", "additive"),
        default=list(DEFAULT_TEMPERATURE_MODES),
        help=(
            "Temperature scaling branches. Defaults to ratio (paper-style, "
            "Kelvin) and additive (recommended sensitivity)."
        ),
    )
    parser.add_argument(
        "--write-netcdf",
        action="store_true",
        help="Also write corrected fields. Omit for the first 3x3 diagnostic run.",
    )
    arguments = parser.parse_args()
    if arguments.n_quantiles < 3:
        parser.error("--n-quantiles must be at least 3")
    try:
        arguments.bounds = core.parse_bounds(
            arguments.bounds or ["0.02,0.98", "0.05,0.95"]
        )
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    return arguments


def main() -> int:
    arguments = parse_arguments()
    tasks: list[tuple[str, str, str]] = []
    for model in arguments.models:
        for variable in arguments.variables:
            modes = (
                arguments.temperature_modes
                if variable == "tas"
                else ["ratio"]
            )
            for correction_mode in modes:
                tasks.append((model, variable, correction_mode))
    total = len(tasks)
    completed = 0
    failures: list[dict[str, str]] = []
    combined_monthly: list[dict] = []
    combined_summary: list[dict] = []

    for model, variable, correction_mode in tasks:
        completed += 1
        print("=" * 100)
        print(
            f"[{completed}/{total}] {model} / {variable} / {correction_mode}"
        )
        print("=" * 100)
        single = SimpleNamespace(
            model=model,
            variable=variable,
            correction_mode=correction_mode,
            cmip6_root=arguments.cmip6_root,
            era5_root=arguments.era5_root,
            output_root=arguments.output_root,
            bounds=arguments.bounds,
            n_quantiles=arguments.n_quantiles,
            skip_netcdf=not arguments.write_netcdf,
        )
        try:
            core.run_validation(single)
            variable_directory = (
                f"{variable}_{correction_mode}"
                if variable == "tas"
                else variable
            )
            result_root = arguments.output_root / model / variable_directory
            monthly_rows = read_csv(result_root / "holdout_monthly_metrics.csv")
            summary_rows = read_csv(result_root / "holdout_method_summary.csv")
            for row in summary_rows:
                row = {
                    "model": model,
                    "variable": variable,
                    "correction_mode": correction_mode,
                    **row,
                }
                combined_summary.append(row)
            combined_monthly.extend(monthly_rows)
        except Exception as error:
            failures.append(
                {
                    "model": model,
                    "variable": variable,
                    "correction_mode": correction_mode,
                    "error": str(error),
                }
            )
            print(f"FAILED: {error}")

    if combined_monthly:
        write_combined_csv(
            arguments.output_root / "pilot_3x3_monthly_metrics.csv",
            combined_monthly,
        )
    if combined_summary:
        write_combined_csv(
            arguments.output_root / "pilot_3x3_method_summary.csv",
            combined_summary,
        )
    if failures:
        write_combined_csv(
            arguments.output_root / "pilot_3x3_failures.csv",
            failures,
        )

    print("=" * 100)
    print("3x3 PILOT SUMMARY")
    print("=" * 100)
    print(f"Expected combinations : {total}")
    print(f"Completed combinations: {total - len(failures)}")
    print(f"Failed combinations   : {len(failures)}")
    print(f"Output root           : {arguments.output_root}")
    if failures:
        print("The completed combinations were retained; inspect pilot_3x3_failures.csv.")
        return 1
    print("3x3 HOLDOUT PILOT COMPLETED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
