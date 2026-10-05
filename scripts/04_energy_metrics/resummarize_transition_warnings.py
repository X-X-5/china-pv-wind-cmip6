#!/usr/bin/env python
r"""Re-summarize the transition-warning audit CSV with intersection-area weights.

The committed ``results/audits/energy_transition_z_gt_3_warnings.csv`` was
derived from a national monthly series weighted by cos(latitude) within a
centre-point mask.  This migration-only tool re-derives that series with the
shared intersection-area weights (``china_intersection_area_km2``) and rewrites
the same CSV.  It reads the existing 272 energy NetCDF files and never rewrites
them.
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
    parser = argparse.ArgumentParser(description="Re-summarize transition warnings with intersection weights.")
    parser.add_argument("--energy-root", type=Path, help="Default: data/processed/energy_metrics")
    parser.add_argument("--audit-root", type=Path, help="Default: results/audits")
    parser.add_argument("--china-shapefile", type=Path, help="Default: data/boundaries/china_qc_boundary.shp")
    parser.add_argument("--models", nargs="+", help="Optional model subset.")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--expected-models", type=int, default=EXPECTED_MODELS)
    parser.add_argument("--dry-run", action="store_true")
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


def main() -> int:
    arguments = parse_arguments()
    energy_root = (
        arguments.energy_root.expanduser().resolve()
        if arguments.energy_root
        else get_path("data_processed_energy_metrics")
    )
    audit_root = (
        arguments.audit_root.expanduser().resolve()
        if arguments.audit_root
        else get_path("results_audits")
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
        for scenario in scenarios:
            for method in engine.METHODS:
                required.append(
                    energy_root / model / scenario / (
                        f"energy_monthly_{model}_{scenario}_{method}_199401-210012.nc"
                    )
                )
    missing = [p for p in required if not p.is_file()]
    if missing:
        preview = "\n  ".join(str(p) for p in missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} input NetCDF files:\n  {preview}")

    print("=" * 100)
    print("TRANSITION-WARNING RE-SUMMARIZATION (intersection-area weighting)")
    print("=" * 100)
    print(f"Models     : {len(models)}")
    print(f"Scenarios  : {', '.join(scenarios)}")
    print(f"Energy root: {energy_root}")
    print(f"Audit root : {audit_root}")
    print(f"QC region  : {QC_REGION}")

    if arguments.dry_run:
        print("DRY RUN COMPLETED: all inputs present; no CSV written.")
        return 0

    warnings: list[dict[str, object]] = []
    total_combos = len(models) * len(scenarios)
    for model_index, model in enumerate(models, start=1):
        probe_path = energy_root / model / "historical" / (
            f"historical_energy_{model}_qm_historical_199401-201412.nc"
        )
        with open_netcdf(probe_path) as probe:
            lat = np.asarray(probe.lat.values, dtype=np.float64)
            lon = np.asarray(probe.lon.values, dtype=np.float64)
        spatial_weights = china_intersection_weights(lat, lon, shapefile)[
            "china_intersection_area_km2"
        ]
        for scenario in scenarios:
            method_energies: dict[str, xr.Dataset] = {}
            try:
                for method in engine.METHODS:
                    path = energy_root / model / scenario / (
                        f"energy_monthly_{model}_{scenario}_{method}_199401-210012.nc"
                    )
                    method_energies[method] = open_netcdf(path).load()
                rows = engine.make_transition_rows(method_energies, spatial_weights, QC_REGION)
                for row in rows:
                    if bool(row["absolute_z_gt_3_warning"]):
                        warnings.append({"model": model, "scenario": scenario, **row})
            finally:
                for ds in method_energies.values():
                    ds.close()
        print(f"[{model_index:02d}/{len(models):02d}] {model}: {int((spatial_weights > 0.0).sum())} intersecting cells")

    audit_root.mkdir(parents=True, exist_ok=True)
    path = audit_root / "energy_transition_z_gt_3_warnings.csv"
    engine.write_csv(path, warnings)
    print("=" * 100)
    print("TRANSITION-WARNING RE-SUMMARIZATION COMPLETED")
    print("=" * 100)
    print(f"  {path.name}: {len(warnings)} warning rows (|z| > 3)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # pragma: no cover
        print(f"ERROR: {error}", file=sys.stderr)
        raise
