#!/usr/bin/env python
r"""Compute a historical-baseline-aware PV/WPD production pilot (version 2).

Place this file in::

    scripts/04_energy_metrics

Default pilot
-------------
* Model/scenario: CAS-ESM2-0 / ssp245
* Historical sensitivity outputs: historical multiplicative QM and ERA5
  (raw CMIP6 remains an internal input for fitting QM)
* Both future branches share historical-QM meteorology for 1993-12--2014-12
* Paper branch: qm_paper for tas, rsds, and sfcWind
* Optimized branch: qdm_improved tas, qm_paper rsds, qdm_improved sfcWind
* Main turbine hub height: 100 m; sensitivities: 80/100/120 m
* Paper periods: 1994-2014, 2040-2060, 2080-2100
* Monthly complementarity: complete 252-month sequence (primary) and
  12-month climatology (sensitivity)
* Seasonal complementarity: complete 84-season sequence (primary) and
  four-season climatology (sensitivity). December belongs to the following
  season-year; seasonal means are weighted by days in month.

Examples
--------
    python compute_energy_metrics_pilot_v2.py --dry-run
    python compute_energy_metrics_pilot_v2.py
    python compute_energy_metrics_pilot_v2.py --rectangular-domain --dry-run
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import xarray as xr
except ImportError as error:  # pragma: no cover - environment-specific
    raise RuntimeError(
        "xarray is required. Activate the pvwind conda environment before running."
    ) from error


# Reuse the tested monthly QM numerical core used by the production script.
SCRIPT_DIR = Path(__file__).resolve().parent
CORE_SCRIPT_NAME = "qm_qdm_holdout_smoke_test.py"
CORE_SEARCH_DIRECTORIES = (SCRIPT_DIR, SCRIPT_DIR.parent / "03_bias_correction")
for _candidate_directory in CORE_SEARCH_DIRECTORIES:
    if (_candidate_directory / CORE_SCRIPT_NAME).is_file():
        _candidate_text = str(_candidate_directory)
        if _candidate_text not in sys.path:
            sys.path.insert(0, _candidate_text)
        break
else:
    _searched = ", ".join(
        str(path / CORE_SCRIPT_NAME) for path in CORE_SEARCH_DIRECTORIES
    )
    raise FileNotFoundError(
        "Cannot find the tested QM/QDM core script. Searched: " + _searched
    )

import qm_qdm_holdout_smoke_test as core

try:
    from scipy.stats import spearmanr
except ImportError as error:  # pragma: no cover - environment-specific
    raise RuntimeError(
        "scipy is required. Install it in the pvwind conda environment."
    ) from error

# Shared intersection-area weighting (scripts/common/area_weights.py).  The
# centre-point mask and cos(latitude) weight are retired as formal statistics.
from area_weights import (  # noqa: E402
    area_weighted_mean as _area_weighted_mean,
    area_weighted_median as _area_weighted_median,
    china_intersection_weights as _china_intersection_weights,
    geodesic_gridcell_areas as _geodesic_gridcell_areas,
)


VARIABLES = ("tas", "rsds", "sfcWind")
METHODS = ("paper_qm", "optimized")
HISTORICAL_BASELINES = ("qm_historical", "era5_reference")
PERIODS = (
    ("historical", 1994, 2014),
    ("mid_century", 2040, 2060),
    ("late_century", 2080, 2100),
)
SEASONS = (
    ("DJF", ((-1, 12), (0, 1), (0, 2))),
    ("MAM", ((0, 3), (0, 4), (0, 5))),
    ("JJA", ((0, 6), (0, 7), (0, 8))),
    ("SON", ((0, 9), (0, 10), (0, 11))),
)
HISTORICAL_SUPPORT_START = (1993, 12)
HISTORICAL_END = (2014, 12)
HISTORICAL_FIT_START = (1959, 1)
HISTORICAL_FIT_END = (2014, 12)
FUTURE_START = (2015, 1)
FUTURE_END = (2100, 12)
MAIN_HUB_HEIGHT_M = 100.0
SENSITIVITY_HEIGHTS_M = (80.0, 100.0, 120.0)
REFERENCE_HEIGHT_M = 10.0
AIR_DENSITY_KG_M3 = 1.225
PV_GAMMA_PER_C = -0.005
PV_T_STC_C = 25.0
PV_C1_C = 4.3
PV_C2 = 0.943
PV_C3_M2_W = 0.028
PV_C4_C_S_M = -1.528
MIN_FULL_SEQUENCE_FRACTION = 0.80
DEFAULT_LOWER = 0.02
DEFAULT_UPPER = 0.98
DEFAULT_N_QUANTILES = 99


def cftime_decoder():
    """Return the current xarray cftime decoder without deprecated kwargs."""
    try:
        return xr.coders.CFDatetimeCoder(use_cftime=True)
    except AttributeError:  # pragma: no cover - supports older xarray releases
        return True


def drop_auxiliary_coordinates(data):
    """Remove scalar height/surface coordinates that conflict across variables."""
    auxiliary = [
        name for name in data.coords if name not in ("time", "lat", "lon")
    ]
    return data.reset_coords(auxiliary, drop=True) if auxiliary else data


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paper-QM and optimized PV/WPD metrics for one model/scenario, "
            "including period QC, hub-height sensitivity, and complementarity."
        )
    )
    parser.add_argument("--model", default="CAS-ESM2-0")
    parser.add_argument("--scenario", default="ssp245")
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root. Default: the centralized project root.",
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
        help=(
            "Root containing <model>/<scenario>/corrected_*.nc. Default: auto-detect "
            "the formal QM output under data/processed/bias_correction."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Default: <project-root>/archive/abandoned_pilots/energy_metrics_pilot_v2/<model>/<scenario>",
    )
    parser.add_argument(
        "--china-shapefile",
        type=Path,
        help=(
            "Optional WGS84 China boundary .shp. Default: auto-detect "
            "<project-root>/data/boundaries/china_qc_boundary.shp."
        ),
    )
    parser.add_argument(
        "--rectangular-domain",
        action="store_true",
        help="Do not use a China polygon even when the default shapefile exists.",
    )
    parser.add_argument(
        "--hub-height",
        type=float,
        default=MAIN_HUB_HEIGHT_M,
        help="Main turbine hub height in metres (default: 100).",
    )
    parser.add_argument(
        "--sensitivity-heights",
        nargs="+",
        type=float,
        default=list(SENSITIVITY_HEIGHTS_M),
        help="Hub heights for CSV sensitivity summary (default: 80 100 120).",
    )
    parser.add_argument("--lower", type=float, default=DEFAULT_LOWER)
    parser.add_argument("--upper", type=float, default=DEFAULT_UPPER)
    parser.add_argument("--n-quantiles", type=int, default=DEFAULT_N_QUANTILES)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.hub_height <= REFERENCE_HEIGHT_M:
        parser.error("--hub-height must be greater than the 10 m reference height")
    if any(value <= REFERENCE_HEIGHT_M for value in arguments.sensitivity_heights):
        parser.error("Every --sensitivity-heights value must be greater than 10 m")
    if not (0.0 < arguments.lower < arguments.upper < 1.0):
        parser.error("Require 0 < --lower < --upper < 1")
    if arguments.n_quantiles < 3:
        parser.error("--n-quantiles must be at least 3")
    arguments.bounds_label = (
        f"Q{round(arguments.lower * 100):02d}-"
        f"Q{round(arguments.upper * 100):02d}"
    )
    return arguments


def resolve_paths(arguments: argparse.Namespace) -> dict[str, Path | None]:
    script_dir = Path(__file__).resolve().parent
    project_root = (
        arguments.project_root.expanduser().resolve()
        if arguments.project_root
        else PROJECT_ROOT
    )
    cmip6_root = (
        arguments.cmip6_root.expanduser().resolve()
        if arguments.cmip6_root
        else get_path("data_interim_cmip6_china_clipped")
    )
    if not cmip6_root.is_dir():
        raise FileNotFoundError(f"CMIP6 processed root does not exist: {cmip6_root}")
    era5_root = (
        arguments.era5_root.expanduser().resolve()
        if arguments.era5_root
        else get_path("data_interim_era5_on_gcm_grid")
    )
    if not era5_root.is_dir():
        raise FileNotFoundError(f"ERA5-on-GCM-grid root does not exist: {era5_root}")

    if arguments.corrected_root:
        corrected_root = arguments.corrected_root.expanduser().resolve()
    else:
        candidates = (
            get_path("data_processed_bias_correction"),
            get_path("data_processed_bias_correction"),
            get_path("data_processed_bias_correction"),
        )
        corrected_root = next(
            (
                path
                for path in candidates
                if (path / arguments.model / arguments.scenario).is_dir()
            ),
            candidates[0],
        )
    if not (corrected_root / arguments.model / arguments.scenario).is_dir():
        raise FileNotFoundError(
            "Cannot find corrected model/scenario directory: "
            f"{corrected_root / arguments.model / arguments.scenario}. "
            "Supply --corrected-root explicitly if needed."
        )

    output_root = (
        arguments.output_root.expanduser().resolve()
        if arguments.output_root
        else PROJECT_ROOT / "archive" / "abandoned_pilots" / "energy_metrics_pilot_v2" / arguments.model / arguments.scenario
    )

    if arguments.rectangular_domain:
        china_shapefile = None
    elif arguments.china_shapefile:
        china_shapefile = arguments.china_shapefile.expanduser().resolve()
    else:
        candidate = get_path("china_shapefile")
        china_shapefile = candidate if candidate.is_file() else None

    return {
        "script_dir": script_dir,
        "project_root": project_root,
        "cmip6_root": cmip6_root,
        "era5_root": era5_root,
        "corrected_root": corrected_root,
        "output_root": output_root,
        "china_shapefile": china_shapefile,
    }


def unique_match(directory: Path, patterns: Iterable[str], label: str) -> Path:
    matches: set[Path] = set()
    for pattern in patterns:
        matches.update(path.resolve() for path in directory.glob(pattern) if path.is_file())
    ordered = sorted(matches)
    if len(ordered) != 1:
        display = "\n    ".join(str(path) for path in ordered[:10]) or "<none>"
        raise FileNotFoundError(
            f"Expected exactly one {label} in {directory}; found {len(ordered)}:\n"
            f"    {display}"
        )
    return ordered[0]


def discover_inputs(
    cmip6_root: Path,
    era5_root: Path,
    corrected_root: Path,
    model: str,
    scenario: str,
) -> dict[str, dict[str, Path]]:
    historical: dict[str, Path] = {}
    corrected: dict[str, Path] = {}
    era5: dict[str, Path] = {}
    corrected_directory = corrected_root / model / scenario
    era5_directory = era5_root / model
    era5_file = unique_match(
        era5_directory,
        (
            f"ERA5_{model}_195901-201412.nc",
            f"*ERA5*{model}*195901-201412*.nc",
        ),
        f"ERA5-on-{model}-grid file",
    )
    for variable in VARIABLES:
        historical_directory = cmip6_root / model / "historical" / variable
        historical[variable] = unique_match(
            historical_directory,
            (
                f"{variable}_Amon_{model}_historical_*195901-201412.nc",
                f"*{variable}*historical*195901-201412*.nc",
            ),
            f"historical {variable} file",
        )
        corrected[variable] = unique_match(
            corrected_directory,
            (
                f"corrected_{model}_{scenario}_{variable}_Q02-Q98_MW30_201501-210012.nc",
                f"corrected_{model}_{scenario}_{variable}_*MW30*201501-210012.nc",
            ),
            f"corrected future {variable} file",
        )
        era5[variable] = era5_file
    return {"historical": historical, "era5": era5, "corrected": corrected}


def standardize_dataset(dataset: xr.Dataset) -> xr.Dataset:
    rename = {}
    for source, target in (
        ("latitude", "lat"),
        ("longitude", "lon"),
        ("nav_lat", "lat"),
        ("nav_lon", "lon"),
    ):
        if source in dataset.dims or source in dataset.coords:
            if target not in dataset.dims and target not in dataset.coords:
                rename[source] = target
    if rename:
        dataset = dataset.rename(rename)
    required = ("time", "lat", "lon")
    missing = [name for name in required if name not in dataset.coords]
    if missing:
        raise ValueError(f"Dataset is missing required coordinates: {missing}")
    return dataset


def extract_historical(path: Path, variable: str) -> xr.DataArray:
    decoder = cftime_decoder()
    open_kwargs = (
        {"decode_times": decoder}
        if decoder is not True
        else {"decode_times": True, "use_cftime": True}
    )
    with xr.open_dataset(path, **open_kwargs) as source:
        dataset = standardize_dataset(source)
        if variable in dataset.data_vars:
            data = dataset[variable]
        else:
            candidates = [
                name
                for name, value in dataset.data_vars.items()
                if set(("time", "lat", "lon")).issubset(value.dims)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot identify {variable} in {path}; candidates={candidates}"
                )
            data = dataset[candidates[0]]
        data = drop_auxiliary_coordinates(data)
        data = data.transpose("time", "lat", "lon").load()
    data = data.rename(variable)
    return data


def extract_corrected(path: Path, variable: str) -> xr.Dataset:
    decoder = cftime_decoder()
    open_kwargs = (
        {"decode_times": decoder}
        if decoder is not True
        else {"decode_times": True, "use_cftime": True}
    )
    with xr.open_dataset(path, **open_kwargs) as source:
        dataset = standardize_dataset(source)
        required = ("raw", "qm_paper", "qdm_improved")
        missing = [name for name in required if name not in dataset.data_vars]
        if missing:
            raise ValueError(
                f"Corrected file {path} is missing {missing}; "
                f"found data variables={list(dataset.data_vars)}"
            )
        result = drop_auxiliary_coordinates(dataset[list(required)])
        result = result.transpose("time", "lat", "lon").load()
    result.attrs["source_variable"] = variable
    return result


def year_month_keys(time: xr.DataArray) -> np.ndarray:
    years = np.asarray(time.dt.year.values, dtype=np.int64)
    months = np.asarray(time.dt.month.values, dtype=np.int64)
    return years * 12 + months - 1


def expected_keys(start: tuple[int, int], end: tuple[int, int]) -> np.ndarray:
    first = start[0] * 12 + start[1] - 1
    last = end[0] * 12 + end[1] - 1
    return np.arange(first, last + 1, dtype=np.int64)


def validate_time_range(
    data: xr.DataArray | xr.Dataset,
    start: tuple[int, int],
    end: tuple[int, int],
    label: str,
) -> None:
    actual = year_month_keys(data.time)
    expected = expected_keys(start, end)
    if actual.size != np.unique(actual).size:
        raise ValueError(f"{label} contains duplicate year-month entries")
    if not np.array_equal(actual, expected):
        missing = sorted(set(expected.tolist()) - set(actual.tolist()))
        extra = sorted(set(actual.tolist()) - set(expected.tolist()))

        def show(keys: list[int]) -> list[str]:
            return [f"{key // 12:04d}-{key % 12 + 1:02d}" for key in keys[:6]]

        raise ValueError(
            f"{label} does not cover the expected period; "
            f"steps={actual.size}, missing={show(missing)}, extra={show(extra)}"
        )


def select_year_month_range(
    data: xr.DataArray | xr.Dataset,
    start: tuple[int, int],
    end: tuple[int, int],
) -> xr.DataArray | xr.Dataset:
    keys = year_month_keys(data.time)
    first = start[0] * 12 + start[1] - 1
    last = end[0] * 12 + end[1] - 1
    indices = np.flatnonzero((keys >= first) & (keys <= last))
    result = data.isel(time=indices)
    validate_time_range(result, start, end, f"selected {start}-{end}")
    return result


def validate_units(variable: str, data: xr.DataArray, label: str) -> str:
    unit = str(data.attrs.get("units", "")).strip()
    normalized = unit.lower().replace(" ", "")
    accepted = {
        "tas": {"k", "kelvin"},
        "rsds": {"wm-2", "w/m2", "wm**-2", "wm^-2"},
        "sfcWind": {"ms-1", "m/s", "ms**-1", "ms^-1"},
    }[variable]
    if normalized not in accepted:
        raise ValueError(
            f"Unexpected {variable} units in {label}: {unit!r}; accepted={sorted(accepted)}"
        )
    return unit


def grids_equal(left: xr.DataArray | xr.Dataset, right: xr.DataArray | xr.Dataset) -> bool:
    return (
        left.sizes["lat"] == right.sizes["lat"]
        and left.sizes["lon"] == right.sizes["lon"]
        and np.allclose(left.lat.values, right.lat.values, rtol=0.0, atol=1.0e-10)
        and np.allclose(left.lon.values, right.lon.values, rtol=0.0, atol=1.0e-10)
    )


def historical_qm(
    model: xr.DataArray,
    reference: xr.DataArray,
    variable: str,
    lower: float,
    upper: float,
    n_quantiles: int,
) -> xr.DataArray:
    """Apply the production paper-style monthly multiplicative QM to history."""
    model_months = np.asarray(model.time.dt.month.values, dtype=np.int64)
    reference_months = np.asarray(reference.time.dt.month.values, dtype=np.int64)
    output = np.full(model.shape, np.nan, dtype=np.float32)
    for month in range(1, 13):
        model_indices = np.flatnonzero(model_months == month)
        reference_indices = np.flatnonzero(reference_months == month)
        model_values = np.asarray(model.isel(time=model_indices).values)
        reference_values = np.asarray(reference.isel(time=reference_indices).values)
        result = core.correct_month(
            model_values,
            reference_values,
            model_values,
            variable,
            lower,
            upper,
            n_quantiles,
            "ratio",
        )
        output[model_indices] = result.qm.astype(np.float32)
    corrected = xr.DataArray(
        output,
        coords=model.coords,
        dims=model.dims,
        name=variable,
        attrs=dict(model.attrs),
    )
    corrected.attrs.update(
        correction_method="monthly multiplicative quantile mapping",
        historical_fit_period="1959-2014",
        reference="ERA5 on the model grid",
        quantile_bounds=(
            f"Q{round(lower * 100):02d}-Q{round(upper * 100):02d}"
        ),
        quantile_nodes=int(n_quantiles),
        role="common corrected historical baseline for QM/QDM comparison",
    )
    return corrected


def load_inputs(
    paths: dict[str, dict[str, Path]],
    lower: float,
    upper: float,
    n_quantiles: int,
) -> tuple[
    dict[str, dict[str, xr.DataArray]],
    dict[str, xr.Dataset],
]:
    historical_full: dict[str, xr.DataArray] = {}
    reference_full: dict[str, xr.DataArray] = {}
    corrected: dict[str, xr.Dataset] = {}

    for variable in VARIABLES:
        historical_full[variable] = extract_historical(
            paths["historical"][variable], variable
        )
        reference_full[variable] = extract_historical(paths["era5"][variable], variable)
        validate_time_range(
            historical_full[variable],
            HISTORICAL_FIT_START,
            HISTORICAL_FIT_END,
            f"historical CMIP6 {variable}",
        )
        validate_time_range(
            reference_full[variable],
            HISTORICAL_FIT_START,
            HISTORICAL_FIT_END,
            f"ERA5 {variable}",
        )
        validate_units(variable, historical_full[variable], f"historical {variable}")
        validate_units(variable, reference_full[variable], f"ERA5 {variable}")
        if not grids_equal(historical_full[variable], reference_full[variable]):
            raise ValueError(f"Historical CMIP6 and ERA5 grids differ for {variable}")

        corrected[variable] = extract_corrected(paths["corrected"][variable], variable)
        validate_time_range(
            corrected[variable], FUTURE_START, FUTURE_END, f"corrected {variable}"
        )
        validate_units(
            variable, corrected[variable]["raw"], f"corrected future {variable}"
        )
        if not grids_equal(historical_full[variable], corrected[variable]):
            raise ValueError(f"Historical and corrected grids differ for {variable}")

    baselines: dict[str, dict[str, xr.DataArray]] = {
        name: {} for name in HISTORICAL_BASELINES
    }
    for variable in VARIABLES:
        era5_support = select_year_month_range(
            reference_full[variable], HISTORICAL_SUPPORT_START, HISTORICAL_END
        )
        qm_full = historical_qm(
            historical_full[variable],
            reference_full[variable],
            variable,
            lower,
            upper,
            n_quantiles,
        )
        qm_support = select_year_month_range(
            qm_full, HISTORICAL_SUPPORT_START, HISTORICAL_END
        )
        baselines["qm_historical"][variable] = qm_support
        baselines["era5_reference"][variable] = era5_support

    for baseline_name, variables in baselines.items():
        for variable in VARIABLES[1:]:
            if not grids_equal(variables[VARIABLES[0]], variables[variable]):
                raise ValueError(
                    f"{baseline_name} grids differ: tas vs {variable}"
                )
            if not np.array_equal(
                year_month_keys(variables[VARIABLES[0]].time),
                year_month_keys(variables[variable].time),
            ):
                raise ValueError(
                    f"{baseline_name} time axes differ: tas vs {variable}"
                )
    return baselines, corrected


def build_china_weights(
    target: xr.DataArray,
    shapefile_path: Path | None,
) -> tuple[np.ndarray, str, dict[str, object]]:
    """Return per-cell China-overlap areas (km²) for intersection-area weighting.

    The returned array is ``china_intersection_area_km2``: positive where a grid
    cell overlaps the China boundary and zero elsewhere.  ``weights > 0`` selects
    the analysis cells; the weights themselves are the geographic weights used in
    every national spatial mean.  No centre-point mask is retained as a statistic.
    """
    shape = (target.sizes["lat"], target.sizes["lon"])
    latitudes = np.asarray(target.lat.values, dtype=np.float64)
    longitudes = np.asarray(target.lon.values, dtype=np.float64)
    if shapefile_path is None:
        weights = _geodesic_gridcell_areas(latitudes, longitudes)
        return weights, "rectangular_domain", {
            "shapefile": None,
            "selected_grid_cells": int((weights > 0.0).sum()),
            "total_grid_cells": int(np.prod(shape)),
            "selection_rule": "all rectangular-domain cells, weighted by geodesic cell area",
            "weighting": "geodesic grid-cell area (WGS84)",
        }

    if not shapefile_path.is_file():
        raise FileNotFoundError(f"China shapefile does not exist: {shapefile_path}")
    missing = [
        str(shapefile_path.with_suffix(suffix))
        for suffix in (".shx", ".dbf", ".prj")
        if not shapefile_path.with_suffix(suffix).is_file()
    ]
    if missing:
        raise FileNotFoundError("China shapefile sidecars are missing: " + ", ".join(missing))
    projection = shapefile_path.with_suffix(".prj").read_text(
        encoding="utf-8", errors="ignore"
    ).upper()
    if "WGS" not in projection and "4326" not in projection:
        raise ValueError("China shapefile must use WGS84 longitude/latitude coordinates")

    result = _china_intersection_weights(latitudes, longitudes, shapefile_path)
    weights = result["china_intersection_area_km2"]
    counts = result["counts"]
    if int(counts["intersects"]) == 0:
        raise ValueError("China shapefile overlaps zero target-grid cells")
    return weights, "china_intersection", {
        "shapefile": str(shapefile_path),
        "selected_grid_cells": int(counts["intersects"]),
        "total_grid_cells": int(counts["total"]),
        "selected_fraction": float(counts["intersects"] / counts["total"]),
        "selection_rule": "grid-cell area intersecting the China polygon",
        "weighting": "geodesic intersection area with the China boundary (WGS84)",
        "fully_inside_cells": int(counts["fully_inside"]),
        "partial_cells": int(counts["partial"]),
        "outside_cells": int(counts["outside"]),
        "china_boundary_area_km2": float(result["china_boundary_area_km2"]),
        "china_intersection_total_km2": float(weights.sum()),
    }


def concatenate_historical_future(
    historical: xr.DataArray,
    future: xr.DataArray,
    name: str,
) -> xr.DataArray:
    result = xr.concat([historical, future], dim="time").sortby("time")
    result = result.rename(name)
    result.attrs.update(future.attrs)
    validate_time_range(
        result, HISTORICAL_SUPPORT_START, FUTURE_END, f"combined {name}"
    )
    return result


def method_meteorology(
    historical_qm_data: dict[str, xr.DataArray],
    corrected: dict[str, xr.Dataset],
    method: str,
) -> xr.Dataset:
    if method == "paper_qm":
        selected = {variable: "qm_paper" for variable in VARIABLES}
    elif method == "optimized":
        selected = {
            "tas": "qdm_improved",
            "rsds": "qm_paper",
            "sfcWind": "qdm_improved",
        }
    else:
        raise ValueError(f"Unknown method: {method}")
    arrays = {
        variable: concatenate_historical_future(
            historical_qm_data[variable],
            corrected[variable][selected[variable]],
            variable,
        )
        for variable in VARIABLES
    }
    aligned = xr.align(*(arrays[name] for name in VARIABLES), join="exact")
    return xr.Dataset(dict(zip(VARIABLES, aligned)))


def historical_meteorology(
    variables: dict[str, xr.DataArray],
) -> xr.Dataset:
    aligned = xr.align(*(variables[name] for name in VARIABLES), join="exact")
    return xr.Dataset(dict(zip(VARIABLES, aligned)))


def wind_at_height(
    wind_10m: xr.DataArray,
    height_m: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    values = np.asarray(wind_10m.values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("sfcWind contains no finite values")
    if float(finite.min()) < -1.0e-6:
        raise ValueError(f"sfcWind contains negative values; minimum={finite.min():.6g}")
    nonnegative = xr.where(wind_10m > 0.0, wind_10m, 0.0)
    safe_positive = xr.where(nonnegative > 0.0, nonnegative, np.nan)
    exponent = (0.37 - 0.0881 * np.log(safe_positive)).rename(
        "roughness_exponent"
    )
    wind_hub = xr.where(
        nonnegative > 0.0,
        nonnegative * (height_m / REFERENCE_HEIGHT_M) ** exponent,
        0.0,
    ).rename("wind_speed_hub")
    exponent.attrs.update(
        units="1",
        long_name="Empirical surface roughness exponent",
        formula="0.37 - 0.0881 ln(v10) for h0=10 m",
    )
    wind_hub.attrs.update(
        units="m s-1",
        long_name=f"Wind speed extrapolated to {height_m:g} m",
        reference_height_m=REFERENCE_HEIGHT_M,
        target_height_m=float(height_m),
    )
    return exponent, wind_hub


def compute_energy(meteorology: xr.Dataset, hub_height_m: float) -> xr.Dataset:
    tas_k = meteorology["tas"].astype(np.float64)
    rsds = meteorology["rsds"].astype(np.float64)
    wind_10m = meteorology["sfcWind"].astype(np.float64)
    tas_c = (tas_k - 273.15).rename("tas_c")
    tas_c.attrs.update(units="degree_Celsius", long_name="Near-surface air temperature")
    tcell = (
        PV_C1_C + PV_C2 * tas_c + PV_C3_M2_W * rsds + PV_C4_C_S_M * wind_10m
    ).rename("tcell")
    tcell.attrs.update(units="degree_Celsius", long_name="Photovoltaic cell temperature")
    performance_ratio = (
        1.0 + PV_GAMMA_PER_C * (tcell - PV_T_STC_C)
    ).rename("performance_ratio")
    performance_ratio.attrs.update(units="1", long_name="PV performance ratio")
    pvpot = (performance_ratio * rsds).rename("pvpot")
    pvpot.attrs.update(units="W m-2", long_name="Photovoltaic power potential")
    exponent, wind_hub = wind_at_height(wind_10m, hub_height_m)
    wind_hub = wind_hub.rename("wind_speed_hub")
    wpd = (0.5 * AIR_DENSITY_KG_M3 * wind_hub**3).rename("wpd")
    wpd.attrs.update(
        units="W m-2",
        long_name=f"Wind power density at {hub_height_m:g} m",
        air_density_kg_m3=AIR_DENSITY_KG_M3,
        hub_height_m=float(hub_height_m),
    )
    output = xr.Dataset(
        {
            "tas_c": tas_c,
            "rsds": rsds,
            "sfcWind_10m": wind_10m,
            "tcell": tcell,
            "performance_ratio": performance_ratio,
            "pvpot": pvpot,
            "roughness_exponent": exponent,
            wind_hub.name: wind_hub,
            "wpd": wpd,
        }
    )
    output.attrs.update(
        hub_height_m=float(hub_height_m),
        reference_height_m=REFERENCE_HEIGHT_M,
        air_density_kg_m3=AIR_DENSITY_KG_M3,
        pv_gamma_per_c=PV_GAMMA_PER_C,
        pv_t_stc_c=PV_T_STC_C,
        note=(
            "Historical source is recorded by the caller; future meteorology "
            "follows the selected correction branch."
        ),
    )
    return output


def mask_values(data: xr.DataArray, spatial_weights: np.ndarray) -> np.ndarray:
    """Flatten per-cell values over the China-intersecting cells (weights > 0)."""
    values = np.asarray(data.values, dtype=np.float64)
    selection = spatial_weights > 0.0
    if values.ndim == 2:
        return values[selection]
    if values.ndim == 3:
        return values[:, selection]
    raise ValueError(f"Expected 2-D or 3-D data, got shape={values.shape}")


def statistics(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "finite_count": 0,
            "missing_percent": 100.0,
            "negative_percent": math.nan,
            "minimum": math.nan,
            "p01": math.nan,
            "median": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "p99": math.nan,
            "maximum": math.nan,
        }
    return {
        "finite_count": int(finite.size),
        "missing_percent": float(100.0 * (1.0 - finite.size / array.size)),
        "negative_percent": float(100.0 * np.mean(finite < 0.0)),
        "minimum": float(np.min(finite)),
        "p01": float(np.quantile(finite, 0.01)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p99": float(np.quantile(finite, 0.99)),
        "maximum": float(np.max(finite)),
    }


def area_weighted_mean_map(data: xr.DataArray, spatial_weights: np.ndarray) -> float:
    """China-intersection-area-weighted spatial mean of a 2-D (lat, lon) map."""
    values = np.asarray(data.values, dtype=np.float64)
    return float(_area_weighted_mean(values, spatial_weights))


def select_years(data: xr.DataArray | xr.Dataset, start: int, end: int):
    years = np.asarray(data.time.dt.year.values, dtype=np.int64)
    return data.isel(time=np.flatnonzero((years >= start) & (years <= end)))


def make_qc_rows(
    method: str,
    energy: xr.Dataset,
    spatial_weights: np.ndarray,
    qc_region: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    output = select_years(energy, 1994, 2100)
    for variable in output.data_vars:
        row: dict[str, object] = {
            "method": method,
            "variable": variable,
            "qc_region": qc_region,
            "time_steps": int(output.sizes["time"]),
            "unit": str(output[variable].attrs.get("units", "")),
        }
        row.update(statistics(mask_values(output[variable], spatial_weights)))
        rows.append(row)
    return rows


def make_period_rows(
    method: str,
    energy: xr.Dataset,
    spatial_weights: np.ndarray,
    qc_region: str,
    periods: tuple[tuple[str, int, int], ...] = PERIODS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for period_name, start, end in periods:
        period = select_years(energy, start, end)
        expected = (end - start + 1) * 12
        if period.sizes["time"] != expected:
            raise ValueError(
                f"{method}/{period_name} expected {expected} months, "
                f"found {period.sizes['time']}"
            )
        for variable in ("tcell", "performance_ratio", "pvpot", "wind_speed_hub", "wpd"):
            data = period[variable]
            sample_row: dict[str, object] = {
                "method": method,
                "period": period_name,
                "start_year": start,
                "end_year": end,
                "months": expected,
                "variable": variable,
                "aggregation": "all_monthly_grid_samples",
                "qc_region": qc_region,
                "unit": str(data.attrs.get("units", "")),
                "area_weighted_spatial_mean": math.nan,
            }
            sample_row.update(statistics(mask_values(data, spatial_weights)))
            rows.append(sample_row)

            mean_map = data.mean("time", skipna=True)
            map_row: dict[str, object] = {
                "method": method,
                "period": period_name,
                "start_year": start,
                "end_year": end,
                "months": expected,
                "variable": variable,
                "aggregation": "period_mean_map",
                "qc_region": qc_region,
                "unit": str(data.attrs.get("units", "")),
                "area_weighted_spatial_mean": area_weighted_mean_map(mean_map, spatial_weights),
            }
            map_row.update(statistics(mask_values(mean_map, spatial_weights)))
            rows.append(map_row)
    return rows


def make_height_sensitivity_rows(
    method: str,
    wind_10m: xr.DataArray,
    heights: list[float],
    spatial_weights: np.ndarray,
    qc_region: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for height in sorted(set(float(value) for value in heights)):
        _, wind_hub = wind_at_height(wind_10m, height)
        wpd = 0.5 * AIR_DENSITY_KG_M3 * wind_hub**3
        for period_name, start, end in PERIODS:
            period = select_years(wpd, start, end)
            mean_map = period.mean("time", skipna=True)
            row: dict[str, object] = {
                "method": method,
                "period": period_name,
                "start_year": start,
                "end_year": end,
                "hub_height_m": height,
                "qc_region": qc_region,
                "area_weighted_mean_wpd": area_weighted_mean_map(mean_map, spatial_weights),
            }
            row.update({f"wpd_{key}": value for key, value in statistics(mask_values(mean_map, spatial_weights)).items()})
            rows.append(row)
    reference = {
        (row["method"], row["period"]): row["area_weighted_mean_wpd"]
        for row in rows
        if math.isclose(float(row["hub_height_m"]), MAIN_HUB_HEIGHT_M)
    }
    for row in rows:
        baseline = reference.get((row["method"], row["period"]), math.nan)
        value = float(row["area_weighted_mean_wpd"])
        row["area_weighted_mean_change_from_100m_percent"] = (
            100.0 * (value / baseline - 1.0)
            if np.isfinite(value) and np.isfinite(baseline) and baseline != 0.0
            else math.nan
        )
    return rows


def time_indices(data: xr.Dataset, year_month_pairs: list[tuple[int, int]]) -> list[int]:
    years = np.asarray(data.time.dt.year.values, dtype=np.int64)
    months = np.asarray(data.time.dt.month.values, dtype=np.int64)
    lookup = {(int(year), int(month)): index for index, (year, month) in enumerate(zip(years, months))}
    missing = [pair for pair in year_month_pairs if pair not in lookup]
    if missing:
        raise ValueError(f"Missing months needed for seasonal aggregation: {missing[:6]}")
    return [lookup[pair] for pair in year_month_pairs]


def seasonal_samples(energy: xr.Dataset, start: int, end: int) -> xr.Dataset:
    samples = []
    labels = []
    season_names = []
    for year in range(start, end + 1):
        for season_name, offsets in SEASONS:
            pairs = [(year + offset, month) for offset, month in offsets]
            indices = time_indices(energy, pairs)
            subset = energy[["pvpot", "wpd"]].isel(time=indices)
            days = subset.time.dt.days_in_month.astype(np.float64)
            weighted = (subset * days).sum("time") / days.sum("time")
            samples.append(weighted)
            labels.append(f"{year:04d}-{season_name}")
            season_names.append(season_name)
    output = xr.concat(samples, dim="sample")
    output = output.assign_coords(
        sample=np.arange(len(samples), dtype=np.int32),
        sample_label=("sample", np.asarray(labels, dtype="U8")),
        season=("sample", np.asarray(season_names, dtype="U3")),
    )
    expected = (end - start + 1) * 4
    if output.sizes["sample"] != expected:
        raise AssertionError(f"Expected {expected} seasonal samples")
    return output


def monthly_climatology(monthly: xr.Dataset) -> xr.Dataset:
    grouped = monthly[["pvpot", "wpd"]].groupby("time.month").mean("time")
    return grouped.rename({"month": "sample"}).assign_coords(
        sample=np.arange(1, 13, dtype=np.int16)
    )


def seasonal_climatology(seasonal: xr.Dataset) -> xr.Dataset:
    pieces = []
    for index, (season_name, _) in enumerate(SEASONS):
        indices = np.flatnonzero(np.asarray(seasonal.season.values) == season_name)
        piece = seasonal[["pvpot", "wpd"]].isel(sample=indices).mean("sample")
        pieces.append(piece)
    return xr.concat(pieces, dim="sample").assign_coords(
        sample=np.arange(4, dtype=np.int16),
        season=("sample", np.asarray([name for name, _ in SEASONS], dtype="U3")),
    )


def spearman_grid(
    pvpot: xr.DataArray,
    wpd: xr.DataArray,
    minimum_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(pvpot.values, dtype=np.float64)
    y = np.asarray(wpd.values, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError(f"Spearman inputs must share (sample, lat, lon); got {x.shape}, {y.shape}")
    _, nlat, nlon = x.shape
    rho = np.full((nlat, nlon), np.nan, dtype=np.float32)
    p_value = np.full((nlat, nlon), np.nan, dtype=np.float32)
    count = np.zeros((nlat, nlon), dtype=np.int16)
    for i in range(nlat):
        for j in range(nlon):
            left = x[:, i, j]
            right = y[:, i, j]
            valid = np.isfinite(left) & np.isfinite(right)
            n = int(valid.sum())
            count[i, j] = n
            if n < minimum_count:
                continue
            left_valid = left[valid]
            right_valid = right[valid]
            if np.ptp(left_valid) == 0.0 or np.ptp(right_valid) == 0.0:
                continue
            result = spearmanr(left_valid, right_valid)
            rho[i, j] = np.float32(result.statistic)
            p_value[i, j] = np.float32(result.pvalue)
    return rho, p_value, count


def category_name(value: float) -> str | None:
    if not np.isfinite(value):
        return None
    if value < -0.9:
        return "very_strong_complementarity"
    if value < -0.6:
        return "strong_complementarity"
    if value < -0.3:
        return "moderate_complementarity"
    if value < 0.0:
        return "weak_complementarity"
    if value < 0.3:
        return "weak_similarity"
    if value < 0.6:
        return "moderate_similarity"
    if value < 0.9:
        return "strong_similarity"
    return "very_strong_similarity"


_CATEGORY_NAMES = (
    "very_strong_complementarity",  # 1
    "strong_complementarity",       # 2
    "moderate_complementarity",     # 3
    "weak_complementarity",         # 4
    "weak_similarity",              # 5
    "moderate_similarity",          # 6
    "strong_similarity",            # 7
    "very_strong_similarity",       # 8
)


def category_codes(values: np.ndarray) -> np.ndarray:
    """Vectorized 1..8 complementarity class, matching :func:`category_name`.

    ``rho == 0`` is class 5 (weak similarity), exactly as in ``category_name``.
    Non-finite values map to NaN.
    """
    data = np.asarray(values, dtype=np.float64)
    codes = np.full(data.shape, np.nan, dtype=np.float64)
    codes = np.where(data < -0.9, 1.0, codes)
    codes = np.where((data >= -0.9) & (data < -0.6), 2.0, codes)
    codes = np.where((data >= -0.6) & (data < -0.3), 3.0, codes)
    codes = np.where((data >= -0.3) & (data < 0.0), 4.0, codes)
    codes = np.where((data >= 0.0) & (data < 0.3), 5.0, codes)
    codes = np.where((data >= 0.3) & (data < 0.6), 6.0, codes)
    codes = np.where((data >= 0.6) & (data < 0.9), 7.0, codes)
    codes = np.where(data >= 0.9, 8.0, codes)
    return codes


def complementarity_summary_row(
    method: str,
    period_name: str,
    start: int,
    end: int,
    definition: str,
    total: int,
    minimum: int,
    rho: np.ndarray,
    spatial_weights: np.ndarray,
    qc_region: str,
) -> dict[str, object]:
    """Area-weighted national aggregation of a per-cell Spearman-rho map."""
    valid = np.isfinite(rho) & (spatial_weights > 0.0)
    valid_area = np.where(valid, spatial_weights, 0.0)
    total_area = float(valid_area.sum())
    codes = category_codes(rho)
    row: dict[str, object] = {
        "method": method,
        "period": period_name,
        "start_year": start,
        "end_year": end,
        "definition": definition,
        "expected_samples": total,
        "minimum_valid_samples": minimum,
        "qc_region": qc_region,
        "valid_grid_cells": int(valid.sum()),
        "missing_grid_cells": int((spatial_weights > 0.0).sum() - valid.sum()),
        "spatial_median_rho": _area_weighted_median(rho, spatial_weights),
        "spatial_mean_rho": float(_area_weighted_mean(rho, spatial_weights)),
    }
    for index, name in enumerate(_CATEGORY_NAMES, start=1):
        area_in_category = float((valid_area * (codes == index)).sum())
        row[f"{name}_percent"] = (
            100.0 * area_in_category / total_area if total_area > 0.0 else math.nan
        )
    return row


def compute_complementarity(
    method: str,
    energy: xr.Dataset,
    spatial_weights: np.ndarray,
    qc_region: str,
    periods: tuple[tuple[str, int, int], ...] = PERIODS,
) -> tuple[xr.Dataset, list[dict[str, object]]]:
    definitions = (
        "monthly_full",
        "monthly_climatology",
        "seasonal_full",
        "seasonal_climatology",
    )
    result_parts = []
    summary_rows: list[dict[str, object]] = []
    for period_name, start, end in periods:
        monthly = select_years(energy[["pvpot", "wpd"]], start, end)
        if monthly.sizes["time"] != 252:
            raise ValueError(f"{period_name} must contain 252 monthly samples")
        seasonal = seasonal_samples(energy, start, end)
        sample_sets = {
            "monthly_full": monthly.rename({"time": "sample"}),
            "monthly_climatology": monthly_climatology(monthly),
            "seasonal_full": seasonal,
            "seasonal_climatology": seasonal_climatology(seasonal),
        }
        for definition in definitions:
            samples = sample_sets[definition]
            total = int(samples.sizes["sample"])
            minimum = total if "climatology" in definition else int(math.ceil(total * MIN_FULL_SEQUENCE_FRACTION))
            rho, p_value, count = spearman_grid(
                samples["pvpot"].transpose("sample", "lat", "lon"),
                samples["wpd"].transpose("sample", "lat", "lon"),
                minimum,
            )
            piece = xr.Dataset(
                {
                    "spearman_rho": (("lat", "lon"), rho),
                    "spearman_p_value": (("lat", "lon"), p_value),
                    "valid_n": (("lat", "lon"), count),
                },
                coords={"lat": energy.lat, "lon": energy.lon},
            ).expand_dims(period=[period_name], definition=[definition])
            result_parts.append(piece)

            summary_rows.append(
                complementarity_summary_row(
                    method,
                    period_name,
                    start,
                    end,
                    definition,
                    total,
                    minimum,
                    rho,
                    spatial_weights,
                    qc_region,
                )
            )
    output = xr.combine_by_coords(result_parts, combine_attrs="drop_conflicts")
    output.attrs.update(
        method=method,
        primary_monthly_definition="Spearman correlation across complete 252-month sequence",
        sensitivity_monthly_definition="Spearman correlation across 12 monthly climatological means",
        primary_seasonal_definition="Spearman correlation across complete 84-season sequence",
        sensitivity_seasonal_definition="Spearman correlation across four seasonal climatological means",
        djf_rule="December belongs to the following season-year",
        seasonal_weighting="days in month",
        tie_handling="scipy.stats.spearmanr average ranks",
        zero_category_rule="rho == 0 is classified as weak similarity",
    )
    return output, summary_rows


def make_historical_baseline_rows(
    baseline_energies: dict[str, xr.Dataset],
    spatial_weights: np.ndarray,
    qc_region: str,
) -> list[dict[str, object]]:
    """Compare historical-QM and ERA5 energy diagnostics for 1994-2014."""
    rows: list[dict[str, object]] = []
    variables = (
        "tas_c",
        "rsds",
        "sfcWind_10m",
        "tcell",
        "performance_ratio",
        "pvpot",
        "wind_speed_hub",
        "wpd",
    )
    for baseline, energy_with_support in baseline_energies.items():
        energy = select_years(energy_with_support, 1994, 2014)
        if energy.sizes["time"] != 252:
            raise ValueError(f"{baseline} must contain 252 historical months")
        for variable in variables:
            data = energy[variable]
            mean_map = data.mean("time", skipna=True)
            row: dict[str, object] = {
                "historical_baseline": baseline,
                "start_year": 1994,
                "end_year": 2014,
                "months": 252,
                "variable": variable,
                "unit": str(data.attrs.get("units", "")),
                "qc_region": qc_region,
                "area_weighted_spatial_mean": area_weighted_mean_map(
                    mean_map, spatial_weights
                ),
            }
            row.update(
                {
                    f"all_samples_{key}": value
                    for key, value in statistics(
                        mask_values(data, spatial_weights)
                    ).items()
                }
            )
            row.update(
                {
                    f"mean_map_{key}": value
                    for key, value in statistics(
                        mask_values(mean_map, spatial_weights)
                    ).items()
                }
            )
            rows.append(row)

    reference = {
        row["variable"]: float(row["area_weighted_spatial_mean"])
        for row in rows
        if row["historical_baseline"] == "era5_reference"
    }
    for row in rows:
        value = float(row["area_weighted_spatial_mean"])
        observed = reference.get(str(row["variable"]), math.nan)
        row["era5_area_weighted_spatial_mean"] = observed
        row["absolute_difference_from_era5"] = value - observed
        row["relative_difference_from_era5_percent"] = (
            100.0 * (value / observed - 1.0)
            if np.isfinite(value) and np.isfinite(observed) and observed != 0.0
            else math.nan
        )
    return rows


def area_weighted_time_series(
    data: xr.DataArray,
    spatial_weights: np.ndarray,
) -> np.ndarray:
    """China-intersection-area-weighted national mean per time step."""
    values = np.asarray(data.values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected (time, lat, lon), got {values.shape}")
    return _area_weighted_mean(values, spatial_weights)


def make_transition_rows(
    method_energies: dict[str, xr.Dataset],
    spatial_weights: np.ndarray,
    qc_region: str,
) -> list[dict[str, object]]:
    """Diagnose 2010-2020 same-calendar-month anomalies around 2014/2015."""
    rows: list[dict[str, object]] = []
    variables = (
        "tas_c",
        "rsds",
        "sfcWind_10m",
        "tcell",
        "pvpot",
        "wind_speed_hub",
        "wpd",
    )
    for method, energy in method_energies.items():
        historical = select_years(energy, 1994, 2014)
        transition = select_years(energy, 2010, 2020)
        historical_months = np.asarray(
            historical.time.dt.month.values, dtype=np.int64
        )
        transition_years = np.asarray(
            transition.time.dt.year.values, dtype=np.int64
        )
        transition_months = np.asarray(
            transition.time.dt.month.values, dtype=np.int64
        )
        for variable in variables:
            historical_series = area_weighted_time_series(
                historical[variable], spatial_weights
            )
            transition_series = area_weighted_time_series(
                transition[variable], spatial_weights
            )
            climatology: dict[int, tuple[float, float, int]] = {}
            for month in range(1, 13):
                values = historical_series[historical_months == month]
                finite = values[np.isfinite(values)]
                mean = float(np.mean(finite)) if finite.size else math.nan
                std = (
                    float(np.std(finite, ddof=1))
                    if finite.size >= 2
                    else math.nan
                )
                climatology[month] = (mean, std, int(finite.size))
            for index, (year, month) in enumerate(
                zip(transition_years, transition_months)
            ):
                value = float(transition_series[index])
                climate_mean, climate_std, climate_n = climatology[int(month)]
                anomaly = value - climate_mean
                z_score = (
                    anomaly / climate_std
                    if np.isfinite(anomaly)
                    and np.isfinite(climate_std)
                    and climate_std > 0.0
                    else math.nan
                )
                rows.append(
                    {
                        "method": method,
                        "year": int(year),
                        "month": int(month),
                        "year_month": f"{int(year):04d}-{int(month):02d}",
                        "segment": "historical_qm" if year <= 2014 else "future_corrected",
                        "variable": variable,
                        "unit": str(transition[variable].attrs.get("units", "")),
                        "qc_region": qc_region,
                        "area_weighted_spatial_mean": value,
                        "historical_same_month_mean": climate_mean,
                        "historical_same_month_std": climate_std,
                        "historical_same_month_n": climate_n,
                        "anomaly_from_historical_same_month": anomaly,
                        "standardized_anomaly_z": z_score,
                        "absolute_z_gt_3_warning": bool(
                            np.isfinite(z_score) and abs(z_score) > 3.0
                        ),
                    }
                )
    return rows


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


def write_dataset(path: Path, dataset: xr.Dataset, time_chunks: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = dataset.copy()
    for coordinate in ("time", "lat", "lon"):
        if coordinate in output.coords and output[coordinate].attrs.get("bounds") not in output:
            output[coordinate].attrs.pop("bounds", None)
    encoding = {}
    for name, variable in output.data_vars.items():
        options: dict[str, object] = {"zlib": True, "complevel": 3}
        if np.issubdtype(variable.dtype, np.floating):
            options["dtype"] = "float32"
        if time_chunks and variable.dims == ("time", "lat", "lon"):
            options["chunksizes"] = (
                min(12, output.sizes["time"]),
                output.sizes["lat"],
                output.sizes["lon"],
            )
        encoding[name] = options
    try:
        output.to_netcdf(path, encoding=encoding)
    except (ValueError, TypeError):
        output.to_netcdf(path)


def print_inspection(
    arguments: argparse.Namespace,
    paths: dict[str, Path | None],
    input_paths: dict[str, dict[str, Path]],
    historical_baselines: dict[str, dict[str, xr.DataArray]],
    corrected: dict[str, xr.Dataset],
    mask: np.ndarray,
    qc_region: str,
    mask_metadata: dict[str, object],
) -> None:
    print("=" * 100)
    print("ENERGY METRICS PILOT INPUT INSPECTION")
    print("=" * 100)
    print(f"Model          : {arguments.model}")
    print(f"Scenario       : {arguments.scenario}")
    print(f"Hub height     : {arguments.hub_height:g} m")
    print(f"Sensitivity    : {', '.join(f'{x:g} m' for x in arguments.sensitivity_heights)}")
    print(f"Corrected root : {paths['corrected_root']}")
    print(f"ERA5 root      : {paths['era5_root']}")
    print(f"Output root    : {paths['output_root']}")
    print(f"QC region      : {qc_region}")
    if paths["china_shapefile"] is not None:
        print(f"China boundary : {paths['china_shapefile']}")
    print(
        f"QC grid cells  : {int((mask > 0.0).sum())}/{int(mask.size)} "
        f"({100.0 * (mask > 0.0).mean():.2f}%)"
    )
    print("History output : historical QM and ERA5 sensitivity")
    print("Raw CMIP6      : internal QM-fitting input; no Raw energy baseline output")
    print("Main histories : both paper_qm and optimized use historical QM")
    print("Methods        : paper_qm=(QM,QM,QM); optimized=(QDM,QM,QDM)")
    print(f"QM settings    : {arguments.bounds_label}; {arguments.n_quantiles} nodes")
    print("Periods        : 1994-2014, 2040-2060, 2080-2100")
    print("-" * 100)
    for variable in VARIABLES:
        hist = historical_baselines["qm_historical"][variable]
        future = corrected[variable]
        print(f"[{variable}]")
        print(f"  Historical : {input_paths['historical'][variable]}")
        print(f"  ERA5       : {input_paths['era5'][variable]}")
        print(f"  Corrected  : {input_paths['corrected'][variable]}")
        print(f"  Grid       : {hist.sizes['lat']} lat x {hist.sizes['lon']} lon")
        print(f"  Hist unit  : {hist.attrs.get('units', '')}")
        print(f"  Future unit: {future['raw'].attrs.get('units', '')}")
        print(f"  Hist steps : {hist.sizes['time']} (1993-12 support through 2014-12)")
        print(f"  Future     : {future.sizes['time']} steps (2015-01 through 2100-12)")
        print(f"  Data vars  : {', '.join(future.data_vars)}")
    print("-" * 100)
    print("Mask metadata:")
    for key, value in mask_metadata.items():
        print(f"  {key}: {value}")


def main() -> int:
    arguments = parse_arguments()
    try:
        paths = resolve_paths(arguments)
        input_paths = discover_inputs(
            paths["cmip6_root"],
            paths["era5_root"],
            paths["corrected_root"],
            arguments.model,
            arguments.scenario,
        )
        historical_baselines, corrected = load_inputs(
            input_paths,
            arguments.lower,
            arguments.upper,
            arguments.n_quantiles,
        )
        spatial_weights, qc_region, mask_metadata = build_china_weights(
            historical_baselines["qm_historical"]["tas"],
            paths["china_shapefile"],
        )
        print_inspection(
            arguments,
            paths,
            input_paths,
            historical_baselines,
            corrected,
            spatial_weights,
            qc_region,
            mask_metadata,
        )
        if arguments.dry_run:
            print("=" * 100)
            print("DRY RUN COMPLETED: inputs, variables, units, grids, periods, and mask are valid.")
            print("No output files were written.")
            return 0

        output_root: Path = paths["output_root"]
        output_root.mkdir(parents=True, exist_ok=True)
        all_qc_rows: list[dict[str, object]] = []
        all_period_rows: list[dict[str, object]] = []
        all_height_rows: list[dict[str, object]] = []
        all_complementarity_rows: list[dict[str, object]] = []
        historical_complementarity_rows: list[dict[str, object]] = []
        outputs: dict[str, object] = {}
        baseline_energies: dict[str, xr.Dataset] = {}
        method_energies: dict[str, xr.Dataset] = {}

        print("\nComputing historical baseline sensitivity ...")
        historical_outputs: dict[str, object] = {}
        historical_period = (("historical", 1994, 2014),)
        for baseline in HISTORICAL_BASELINES:
            meteorology = historical_meteorology(historical_baselines[baseline])
            energy_with_support = compute_energy(meteorology, arguments.hub_height)
            energy_output = select_years(energy_with_support, 1994, 2014)
            energy_output.attrs.update(
                model=arguments.model,
                scenario="historical",
                historical_baseline=baseline,
                output_period="1994-2014",
                qc_region=qc_region,
            )
            baseline_energies[baseline] = energy_with_support
            energy_path = output_root / (
                f"historical_energy_{arguments.model}_{baseline}_199401-201412.nc"
            )
            write_dataset(energy_path, energy_output)
            complementarity, rows = compute_complementarity(
                baseline,
                energy_with_support,
                spatial_weights,
                qc_region,
                periods=historical_period,
            )
            complementarity_path = output_root / (
                f"historical_complementarity_{arguments.model}_{baseline}.nc"
            )
            write_dataset(
                complementarity_path, complementarity, time_chunks=False
            )
            historical_complementarity_rows.extend(rows)
            historical_outputs[baseline] = {
                "energy_netcdf": str(energy_path),
                "complementarity_netcdf": str(complementarity_path),
            }
            print(f"  {baseline}: completed")
        outputs["historical_baselines"] = historical_outputs

        for method in METHODS:
            print(f"\nComputing {method} ...")
            meteorology = method_meteorology(
                historical_baselines["qm_historical"], corrected, method
            )
            energy_with_support = compute_energy(meteorology, arguments.hub_height)
            method_energies[method] = energy_with_support
            energy_output = select_years(energy_with_support, 1994, 2100)
            if energy_output.sizes["time"] != 1284:
                raise ValueError(
                    f"{method} energy output must contain 1284 months (1994-2100); "
                    f"found {energy_output.sizes['time']}"
                )
            energy_output.attrs.update(
                model=arguments.model,
                scenario=arguments.scenario,
                method=method,
                output_period="1994-2100",
                paper_periods="1994-2014, 2040-2060, 2080-2100",
                qc_region=qc_region,
            )
            energy_path = output_root / (
                f"energy_monthly_{arguments.model}_{arguments.scenario}_{method}_"
                "199401-210012.nc"
            )
            write_dataset(energy_path, energy_output)
            print(f"  Monthly energy: {energy_path}")

            all_qc_rows.extend(make_qc_rows(method, energy_output, spatial_weights, qc_region))
            all_period_rows.extend(
                make_period_rows(method, energy_output, spatial_weights, qc_region)
            )
            all_height_rows.extend(
                make_height_sensitivity_rows(
                    method,
                    meteorology["sfcWind"],
                    list(arguments.sensitivity_heights),
                    spatial_weights,
                    qc_region,
                )
            )
            complementarity, complementarity_rows = compute_complementarity(
                method, energy_with_support, spatial_weights, qc_region
            )
            complementarity_path = output_root / (
                f"complementarity_{arguments.model}_{arguments.scenario}_{method}.nc"
            )
            write_dataset(complementarity_path, complementarity, time_chunks=False)
            print(f"  Complementarity: {complementarity_path}")
            all_complementarity_rows.extend(complementarity_rows)
            outputs[method] = {
                "energy_netcdf": str(energy_path),
                "complementarity_netcdf": str(complementarity_path),
            }

        qc_path = output_root / "energy_qc_summary.csv"
        period_path = output_root / "energy_period_summary.csv"
        height_path = output_root / "hub_height_sensitivity.csv"
        complementarity_summary_path = output_root / "complementarity_summary.csv"
        historical_baseline_path = output_root / "historical_baseline_comparison.csv"
        historical_complementarity_path = (
            output_root / "historical_baseline_complementarity_summary.csv"
        )
        transition_path = output_root / "historical_future_transition.csv"
        historical_baseline_rows = make_historical_baseline_rows(
            baseline_energies, spatial_weights, qc_region
        )
        transition_rows = make_transition_rows(
            method_energies, spatial_weights, qc_region
        )
        write_csv(qc_path, all_qc_rows)
        write_csv(period_path, all_period_rows)
        write_csv(height_path, all_height_rows)
        write_csv(complementarity_summary_path, all_complementarity_rows)
        write_csv(historical_baseline_path, historical_baseline_rows)
        write_csv(
            historical_complementarity_path, historical_complementarity_rows
        )
        write_csv(transition_path, transition_rows)

        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "model": arguments.model,
            "scenario": arguments.scenario,
            "methods": {
                "paper_qm": {"tas": "qm_paper", "rsds": "qm_paper", "sfcWind": "qm_paper"},
                "optimized": {"tas": "qdm_improved", "rsds": "qm_paper", "sfcWind": "qdm_improved"},
            },
            "historical_rule": (
                "Both paper_qm and optimized branches use monthly multiplicative "
                "QM-corrected CMIP6 history for 1993-12 through 2014-12."
            ),
            "historical_sensitivity_baselines": list(HISTORICAL_BASELINES),
            "historical_qm": {
                "mode": "monthly multiplicative QM for tas, rsds, and sfcWind",
                "fit_period": "1959-2014",
                "reference": "ERA5 on each GCM grid",
                "quantile_bounds": arguments.bounds_label,
                "quantile_nodes": int(arguments.n_quantiles),
                "interpretation": "in-sample baseline alignment, not independent validation",
            },
            "main_hub_height_m": float(arguments.hub_height),
            "sensitivity_heights_m": [float(value) for value in arguments.sensitivity_heights],
            "paper_periods": [
                {"name": name, "start_year": start, "end_year": end}
                for name, start, end in PERIODS
            ],
            "mask": mask_metadata,
            "input_files": {
                group: {variable: str(path) for variable, path in files.items()}
                for group, files in input_paths.items()
            },
            "outputs": outputs,
            "csv_outputs": {
                "qc": str(qc_path),
                "period_summary": str(period_path),
                "hub_height_sensitivity": str(height_path),
                "complementarity_summary": str(complementarity_summary_path),
                "historical_baseline_comparison": str(historical_baseline_path),
                "historical_baseline_complementarity_summary": str(
                    historical_complementarity_path
                ),
                "historical_future_transition": str(transition_path),
            },
            "transition_warning_count": int(
                sum(bool(row["absolute_z_gt_3_warning"]) for row in transition_rows)
            ),
        }
        manifest_path = output_root / "energy_run_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("\n" + "=" * 100)
        print("ENERGY METRICS PILOT COMPLETED")
        print("=" * 100)
        print(f"Output root            : {output_root}")
        print(f"QC summary             : {qc_path}")
        print(f"Period summary         : {period_path}")
        print(f"Hub-height sensitivity : {height_path}")
        print(f"Complementarity summary: {complementarity_summary_path}")
        print(f"Historical baselines    : {historical_baseline_path}")
        print(f"Historical complement.  : {historical_complementarity_path}")
        print(f"2014/2015 transition    : {transition_path}")
        print(f"Manifest               : {manifest_path}")
        print("Inspect CSV diagnostics before expanding to all 17 models.")
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
