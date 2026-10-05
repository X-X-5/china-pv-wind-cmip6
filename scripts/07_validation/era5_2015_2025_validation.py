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
from area_weights import (  # noqa: E402
    area_weighted_mean,
    area_weighted_median,
    china_intersection_weights,
    geodesic_gridcell_areas,
)
# -------------------------------------------------------------------

"""Prepare ERA5 2015-2025 and validate one future QM/QDM pilot.

This single script performs four linked tasks:

1. Read the two ERA5 GRIB groups and align ``ssrd`` by ``valid_time``.
2. Convert to ``tas``, ``rsds`` and ``sfcWind`` and interpolate to the GCM grid.
3. Compare 2015-2025 Raw/QM/QDM monthly climatologies with independent ERA5.
4. Recalculate mid/late-period change signals relative to 2020-2039.

The default paths assume this script is stored in ``scripts/03_bias_correction``.
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    import cfgrib
except ImportError as error:
    raise RuntimeError(
        "cfgrib is required. Run this script with the pvwind conda environment."
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = PROJECT_ROOT
DEFAULT_MODEL = "CAS-ESM2-0"
DEFAULT_SCENARIO = "ssp245"
VARIABLES = ("tas", "rsds", "sfcWind")
METHOD_VARIABLES = {
    "RAW": "raw",
    "QM": "qm_paper",
    "QDM": "qdm_improved",
}
RATIO_DENOMINATOR_FLOOR = {
    "rsds": 1.0,
    "sfcWind": 0.1,
}
VALIDATION_START_YEAR = 2015
VALIDATION_END_YEAR = 2025
NEAR_PERIOD = (2020, 2039)
TARGET_PERIODS = {
    "mid": (2040, 2069),
    "late": (2070, 2099),
}


def month_keys(time: xr.DataArray) -> np.ndarray:
    years = np.asarray(time.dt.year.values, dtype=np.int64)
    months = np.asarray(time.dt.month.values, dtype=np.int64)
    return years * 12 + months - 1


def expected_month_keys(start_year: int, end_year: int) -> np.ndarray:
    return np.arange(start_year * 12, end_year * 12 + 12, dtype=np.int64)


def validate_months(
    time: xr.DataArray,
    start_year: int,
    end_year: int,
    label: str,
) -> None:
    actual = month_keys(time)
    expected = expected_month_keys(start_year, end_year)
    if actual.size != np.unique(actual).size:
        raise ValueError(f"{label} contains duplicate months")
    if not np.array_equal(actual, expected):
        raise ValueError(
            f"{label} must cover {start_year}-01 through {end_year}-12; "
            f"found {actual.size} months"
        )


def standardize_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    if "latitude" in dataset.coords:
        rename["latitude"] = "lat"
    if "longitude" in dataset.coords:
        rename["longitude"] = "lon"
    if rename:
        dataset = dataset.rename(rename)
    return dataset


def find_grib_groups(path: Path) -> tuple[xr.Dataset, xr.Dataset]:
    if not path.is_file():
        raise FileNotFoundError(f"ERA5 GRIB file does not exist: {path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        groups = cfgrib.open_datasets(
            str(path),
            backend_kwargs={"indexpath": ""},
        )
    atmospheric: xr.Dataset | None = None
    radiation: xr.Dataset | None = None
    for group in groups:
        names = set(group.data_vars)
        if {"u10", "v10", "t2m"}.issubset(names):
            atmospheric = standardize_coordinates(group)
        if "ssrd" in names:
            radiation = standardize_coordinates(group)
    if atmospheric is None:
        raise ValueError("Could not find the ERA5 u10/v10/t2m GRIB group")
    if radiation is None:
        raise ValueError("Could not find the ERA5 ssrd GRIB group")
    return atmospheric, radiation


def prepare_era5(path: Path) -> xr.Dataset:
    atmospheric, radiation = find_grib_groups(path)
    if (
        atmospheric.sizes["lat"] != radiation.sizes["lat"]
        or atmospheric.sizes["lon"] != radiation.sizes["lon"]
        or not np.allclose(
            atmospheric.lat.values, radiation.lat.values, rtol=0.0, atol=1.0e-10
        )
        or not np.allclose(
            atmospheric.lon.values, radiation.lon.values, rtol=0.0, atol=1.0e-10
        )
    ):
        raise ValueError("ERA5 atmospheric and ssrd grids differ")
    validate_months(
        atmospheric.time,
        VALIDATION_START_YEAR,
        VALIDATION_END_YEAR,
        "ERA5 atmospheric variables",
    )
    if "valid_time" not in radiation.coords:
        raise ValueError("ERA5 ssrd group has no valid_time coordinate")
    valid_time = radiation["valid_time"]
    validate_months(
        valid_time,
        VALIDATION_START_YEAR,
        VALIDATION_END_YEAR,
        "ERA5 ssrd valid_time",
    )
    if not np.array_equal(month_keys(valid_time), month_keys(atmospheric.time)):
        raise ValueError("ERA5 ssrd valid_time does not match atmospheric time")

    coords = {
        "time": np.asarray(atmospheric.time.values),
        "lat": np.asarray(atmospheric.lat.values, dtype=np.float64),
        "lon": np.asarray(atmospheric.lon.values, dtype=np.float64),
    }
    dims = ("time", "lat", "lon")
    u10 = np.asarray(atmospheric["u10"].values, dtype=np.float64)
    v10 = np.asarray(atmospheric["v10"].values, dtype=np.float64)
    t2m = np.asarray(atmospheric["t2m"].values, dtype=np.float64)
    ssrd = np.asarray(radiation["ssrd"].values, dtype=np.float64)
    if not (u10.shape == v10.shape == t2m.shape == ssrd.shape):
        raise ValueError(
            "ERA5 variable shapes differ after valid_time alignment: "
            f"u10={u10.shape}, v10={v10.shape}, t2m={t2m.shape}, ssrd={ssrd.shape}"
        )

    # ERA5 monthly-mean ssrd is a mean daily accumulation in J m-2.
    # Dividing by 86400 converts it to the monthly-mean flux in W m-2.
    era5 = xr.Dataset(
        {
            "tas": xr.DataArray(t2m, coords=coords, dims=dims),
            "rsds": xr.DataArray(ssrd / 86400.0, coords=coords, dims=dims),
            # This reproduces the historical reference convention used by the project.
            "sfcWind": xr.DataArray(
                np.sqrt(u10**2 + v10**2), coords=coords, dims=dims
            ),
        }
    )
    era5["tas"].attrs.update(
        units="K",
        source_variable="ERA5 t2m",
    )
    era5["rsds"].attrs.update(
        units="W m-2",
        source_variable="ERA5 ssrd",
        conversion="ssrd / 86400",
    )
    era5["sfcWind"].attrs.update(
        units="m s-1",
        source_variable="sqrt(monthly-mean u10^2 + monthly-mean v10^2)",
    )
    era5.attrs.update(
        source="ERA5 monthly averaged reanalysis",
        period="2015-2025",
        ssrd_time_coordinate="valid_time aligned to atmospheric time",
        wind_definition=(
            "magnitude of monthly-mean u10/v10; retained for consistency with "
            "the existing 1959-2014 ERA5 reference"
        ),
    )
    return era5.sortby("lat").sortby("lon")


def find_corrected_file(root: Path, model: str, scenario: str, variable: str) -> Path:
    matches = sorted(root.glob(f"corrected_{model}_{scenario}_{variable}_*MW30*.nc"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one MW30 corrected file for {variable} in {root}; "
            f"found {len(matches)}"
        )
    return matches[0]


def load_corrected(path: Path, variable: str) -> xr.Dataset:
    dataset = standardize_coordinates(xr.open_dataset(path, decode_times=True)).load()
    required = set(METHOD_VARIABLES.values())
    missing = required - set(dataset.data_vars)
    if missing:
        raise ValueError(f"{path.name} is missing variables: {sorted(missing)}")
    for name in required:
        if dataset[name].dims != ("time", "lat", "lon"):
            dataset[name] = dataset[name].transpose("time", "lat", "lon")
    dataset.attrs["validation_variable"] = variable
    return dataset


def same_grid(left: xr.Dataset, right: xr.Dataset) -> bool:
    return (
        left.sizes["lat"] == right.sizes["lat"]
        and left.sizes["lon"] == right.sizes["lon"]
        and np.allclose(left.lat.values, right.lat.values, rtol=0.0, atol=1.0e-10)
        and np.allclose(left.lon.values, right.lon.values, rtol=0.0, atol=1.0e-10)
    )


def interpolate_era5(era5: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    result = era5.interp(
        lat=target.lat,
        lon=target.lon,
        method="linear",
    )
    for variable in VARIABLES:
        values = np.asarray(result[variable].values)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values after ERA5 interpolation: {variable}")
    result.attrs.update(
        interpolation="xarray linear interpolation to GCM latitude/longitude grid",
        target_grid=f"{target.sizes['lat']} lat x {target.sizes['lon']} lon",
    )
    return result


def write_netcdf(path: Path, dataset: xr.Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = (
        min(12, dataset.sizes["time"]),
        dataset.sizes["lat"],
        dataset.sizes["lon"],
    )
    encoding = {
        variable: {
            "zlib": True,
            "complevel": 3,
            "dtype": "float32",
            "chunksizes": chunks,
        }
        for variable in dataset.data_vars
    }
    try:
        dataset.astype(np.float32).to_netcdf(path, encoding=encoding)
    except (ValueError, TypeError):
        dataset.astype(np.float32).to_netcdf(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _weighted_correlation(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if x.shape != y.shape or x.shape != w.shape:
        raise ValueError("correlation inputs must share the same shape")
    finite = np.isfinite(x) & np.isfinite(y) & (w > 0.0)
    x = x[finite]
    y = y[finite]
    w = w[finite]
    if w.size < 3:
        return float("nan")
    total = float(w.sum())
    if total <= 0.0:
        return float("nan")
    mean_x = float((x * w).sum() / total)
    mean_y = float((y * w).sum() / total)
    covariance = float((w * (x - mean_x) * (y - mean_y)).sum() / total)
    var_x = float((w * (x - mean_x) ** 2).sum() / total)
    var_y = float((w * (y - mean_y) ** 2).sum() / total)
    if var_x <= 0.0 or var_y <= 0.0:
        return float("nan")
    return float(covariance / np.sqrt(var_x * var_y))


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.broadcast_to(
        np.asarray(weights, dtype=np.float64), values.shape
    )
    finite = np.isfinite(values) & (weights > 0.0)
    ordered_values = values[finite]
    ordered_weights = weights[finite]
    if ordered_values.size == 0:
        return float("nan")
    order = np.argsort(ordered_values)
    ordered_values = ordered_values[order]
    ordered_weights = ordered_weights[order]
    cumulative = np.cumsum(ordered_weights)
    total = cumulative[-1]
    if total <= 0.0:
        return float("nan")
    index = int(np.searchsorted(cumulative, probability * total))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def build_spatial_weights(
    target: xr.Dataset,
    shapefile_path: Path | None,
) -> tuple[np.ndarray, str, dict]:
    """Return intersection-area weights (km²) on the target GCM grid.

    With a China shapefile, each cell's weight is the WGS84 geodesic area of its
    intersection with the boundary polygon (zero for cells with no overlap).
    Without a shapefile, the full rectangular domain is kept and weighted by each
    cell's own geodesic area.
    """
    shape = (target.sizes["lat"], target.sizes["lon"])
    latitudes = np.asarray(target.lat.values, dtype=np.float64)
    longitudes = np.asarray(target.lon.values, dtype=np.float64)
    if shapefile_path is None:
        weights = geodesic_gridcell_areas(latitudes, longitudes)
        return weights, "rectangular_domain", {
            "shapefile": None,
            "selection_rule": (
                "all grid cells, area-weighted by geodesic cell area (WGS84)"
            ),
        }

    shapefile_path = shapefile_path.expanduser().resolve()
    if not shapefile_path.is_file():
        raise FileNotFoundError(f"China shapefile does not exist: {shapefile_path}")
    if shapefile_path.suffix.lower() != ".shp":
        raise ValueError(f"--china-shapefile must point to a .shp file: {shapefile_path}")
    missing_sidecars = [
        str(shapefile_path.with_suffix(suffix))
        for suffix in (".shx", ".dbf", ".prj")
        if not shapefile_path.with_suffix(suffix).is_file()
    ]
    if missing_sidecars:
        raise FileNotFoundError(
            "China shapefile is missing required companion files: "
            + ", ".join(missing_sidecars)
        )

    projection_text = shapefile_path.with_suffix(".prj").read_text(
        encoding="utf-8", errors="ignore"
    ).upper()
    if "WGS" not in projection_text and "4326" not in projection_text:
        raise ValueError(
            "China shapefile must use WGS84 longitude/latitude coordinates; "
            f"could not confirm this from {shapefile_path.with_suffix('.prj')}"
        )

    result = china_intersection_weights(
        latitudes, longitudes, shapefile_path
    )
    weights = result["china_intersection_area_km2"]
    selected = int(np.count_nonzero(weights > 0.0))
    total = int(weights.size)
    if selected == 0:
        raise ValueError(
            "China intersection weighting selected zero target-grid cells. "
            "Check the shapefile CRS and target longitude convention."
        )
    boundary_area = float(result["china_boundary_area_km2"])
    coverage = float(weights.sum())
    return weights, "china_intersection", {
        "shapefile": str(shapefile_path),
        "selection_rule": (
            "geodesic intersection area between grid cell and China boundary "
            "polygon (WGS84); weight = overlap area in km^2"
        ),
        "selected_grid_cells": selected,
        "total_grid_cells": total,
        "selected_grid_cell_fraction": selected / total,
        "intersection_area_km2": coverage,
        "china_boundary_area_km2": boundary_area,
        "coverage_ratio": coverage / boundary_area if boundary_area > 0.0 else None,
        "cell_counts": result["counts"],
    }


def make_validation_rows(
    variable: str,
    corrected: xr.Dataset,
    era5: xr.Dataset,
    model: str,
    scenario: str,
    weights: np.ndarray,
    qc_region: str,
) -> list[dict]:
    years = np.asarray(corrected.time.dt.year.values, dtype=np.int64)
    selected = np.flatnonzero(
        (years >= VALIDATION_START_YEAR) & (years <= VALIDATION_END_YEAR)
    )
    model_period = corrected.isel(time=selected)
    validate_months(
        model_period.time,
        VALIDATION_START_YEAR,
        VALIDATION_END_YEAR,
        f"{variable} corrected validation period",
    )
    if not np.array_equal(month_keys(model_period.time), month_keys(era5.time)):
        raise ValueError(f"ERA5 and corrected month sequences differ for {variable}")

    model_months = np.asarray(model_period.time.dt.month.values, dtype=np.int64)
    era5_months = np.asarray(era5.time.dt.month.values, dtype=np.int64)
    observation = np.asarray(era5[variable].values, dtype=np.float64)
    if weights.shape != observation.shape[1:]:
        raise ValueError(
            f"Weights shape {weights.shape} does not match "
            f"{variable} grid {observation.shape[1:]}"
        )
    rows: list[dict] = []
    for month in range(1, 13):
        mi = np.flatnonzero(model_months == month)
        oi = np.flatnonzero(era5_months == month)
        obs_sample = observation[oi]
        obs_mean = np.mean(obs_sample, axis=0)
        obs_std = np.std(obs_sample, axis=0, ddof=1)
        for method, data_variable in METHOD_VARIABLES.items():
            sample = np.asarray(model_period[data_variable].isel(time=mi).values)
            sample_mean = np.mean(sample, axis=0)
            sample_std = np.std(sample, axis=0, ddof=1)
            bias = sample_mean - obs_mean
            absolute_bias = np.abs(bias)
            std_error = np.abs(sample_std - obs_std)
            rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "variable": variable,
                    "month": month,
                    "method": method,
                    "validation_period": "2015-2025",
                    "years_per_month": len(mi),
                    "qc_region": qc_region,
                    "unit": str(era5[variable].attrs.get("units", "")),
                    "bias_spatial_mean": area_weighted_mean(bias, weights),
                    "bias_spatial_median": area_weighted_median(bias, weights),
                    "absolute_bias_spatial_median": area_weighted_median(
                        absolute_bias, weights
                    ),
                    "absolute_bias_spatial_mean": area_weighted_mean(
                        absolute_bias, weights
                    ),
                    "rmse_of_climatology_field": float(
                        np.sqrt(area_weighted_mean(bias**2, weights))
                    ),
                    "climatology_spatial_correlation": _weighted_correlation(
                        sample_mean, obs_mean, weights
                    ),
                    "temporal_std_error_spatial_median": area_weighted_median(
                        std_error, weights
                    ),
                    "era5_climatology_spatial_mean": area_weighted_mean(
                        obs_mean, weights
                    ),
                    "method_climatology_spatial_mean": area_weighted_mean(
                        sample_mean, weights
                    ),
                    "era5_p10_all_samples": _weighted_quantile(
                        obs_sample, weights, 0.10
                    ),
                    "method_p10_all_samples": _weighted_quantile(
                        sample, weights, 0.10
                    ),
                    "era5_median_all_samples": _weighted_quantile(
                        obs_sample, weights, 0.50
                    ),
                    "method_median_all_samples": _weighted_quantile(
                        sample, weights, 0.50
                    ),
                    "era5_p90_all_samples": _weighted_quantile(
                        obs_sample, weights, 0.90
                    ),
                    "method_p90_all_samples": _weighted_quantile(
                        sample, weights, 0.90
                    ),
                }
            )
    return rows


def summarize_validation(rows: list[dict]) -> list[dict]:
    table = pd.DataFrame(rows)
    output: list[dict] = []
    for (variable, method), group in table.groupby(["variable", "method"], sort=False):
        output.append(
            {
                "variable": variable,
                "method": method,
                "validation_period": "2015-2025",
                "months": len(group),
                "qc_region": group["qc_region"].iloc[0],
                "median_monthly_absolute_bias_spatial_median": float(
                    group["absolute_bias_spatial_median"].median()
                ),
                "median_monthly_absolute_bias_spatial_mean": float(
                    group["absolute_bias_spatial_mean"].median()
                ),
                "median_monthly_rmse_of_climatology_field": float(
                    group["rmse_of_climatology_field"].median()
                ),
                "median_monthly_climatology_spatial_correlation": float(
                    group["climatology_spatial_correlation"].median()
                ),
                "median_monthly_temporal_std_error_spatial_median": float(
                    group["temporal_std_error_spatial_median"].median()
                ),
            }
        )
    return output


def change_field(
    future_mean: np.ndarray,
    baseline_mean: np.ndarray,
    variable: str,
) -> np.ndarray:
    if variable == "tas":
        return future_mean - baseline_mean
    floor = RATIO_DENOMINATOR_FLOOR[variable]
    valid = np.isfinite(baseline_mean) & (np.abs(baseline_mean) > floor)
    result = np.full(baseline_mean.shape, np.nan, dtype=np.float64)
    np.divide(future_mean, baseline_mean, out=result, where=valid)
    return (result - 1.0) * 100.0


def spatial_summary(
    values: np.ndarray,
    prefix: str,
    weights: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError(
            f"Spatial field shape {values.shape} does not match weights "
            f"shape {weights.shape}"
        )
    selected = np.isfinite(values) & (weights > 0.0)
    if not selected.any():
        return {
            f"{prefix}_spatial_median": np.nan,
            f"{prefix}_spatial_mean": np.nan,
            f"{prefix}_spatial_p10": np.nan,
            f"{prefix}_spatial_p90": np.nan,
            f"{prefix}_spatial_max": np.nan,
        }
    return {
        f"{prefix}_spatial_median": area_weighted_median(values, weights),
        f"{prefix}_spatial_mean": area_weighted_mean(values, weights),
        f"{prefix}_spatial_p10": _weighted_quantile(values, weights, 0.10),
        f"{prefix}_spatial_p90": _weighted_quantile(values, weights, 0.90),
        f"{prefix}_spatial_max": float(np.max(values[selected])),
    }


def make_near_baseline_rows(
    variable: str,
    corrected: xr.Dataset,
    model: str,
    scenario: str,
    weights: np.ndarray,
    qc_region: str,
) -> list[dict]:
    years = np.asarray(corrected.time.dt.year.values, dtype=np.int64)
    months = np.asarray(corrected.time.dt.month.values, dtype=np.int64)
    rows: list[dict] = []
    near_indices_by_month = {
        month: np.flatnonzero(
            (months == month)
            & (years >= NEAR_PERIOD[0])
            & (years <= NEAR_PERIOD[1])
        )
        for month in range(1, 13)
    }
    for period_name, (start_year, end_year) in TARGET_PERIODS.items():
        for month in range(1, 13):
            baseline_indices = near_indices_by_month[month]
            target_indices = np.flatnonzero(
                (months == month) & (years >= start_year) & (years <= end_year)
            )
            signals: dict[str, np.ndarray] = {}
            for method, data_variable in METHOD_VARIABLES.items():
                baseline = np.mean(
                    np.asarray(corrected[data_variable].isel(time=baseline_indices).values),
                    axis=0,
                )
                target = np.mean(
                    np.asarray(corrected[data_variable].isel(time=target_indices).values),
                    axis=0,
                )
                signals[method] = change_field(target, baseline, variable)
            raw_signal = signals["RAW"]
            for method, signal in signals.items():
                difference = np.abs(signal - raw_signal)
                row = {
                    "model": model,
                    "scenario": scenario,
                    "variable": variable,
                    "baseline_period": "2020-2039",
                    "target_period": period_name,
                    "target_start_year": start_year,
                    "target_end_year": end_year,
                    "month": month,
                    "method": method,
                    "signal_unit": "K" if variable == "tas" else "%",
                    "qc_region": qc_region,
                }
                row.update(
                    spatial_summary(signal, "change_signal", weights)
                )
                row.update(
                    spatial_summary(
                        difference,
                        "absolute_difference_from_raw_signal",
                        weights,
                    )
                )
                rows.append(row)
    return rows


def summarize_near_baseline(rows: list[dict]) -> list[dict]:
    table = pd.DataFrame(rows)
    output: list[dict] = []
    metric = "absolute_difference_from_raw_signal_spatial_median"
    grouped = table.groupby(["variable", "target_period", "method"], sort=False)
    for (variable, target_period, method), group in grouped:
        output.append(
            {
                "variable": variable,
                "target_period": target_period,
                "baseline_period": "2020-2039",
                "method": method,
                "signal_unit": group["signal_unit"].iloc[0],
                "qc_region": group["qc_region"].iloc[0],
                "median_of_monthly_spatial_median_absolute_difference": float(
                    group[metric].median()
                ),
                "mean_of_monthly_spatial_median_absolute_difference": float(
                    group[metric].mean()
                ),
                "maximum_monthly_spatial_median_absolute_difference": float(
                    group[metric].max()
                ),
            }
        )
    return output


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare ERA5 2015-2025, validate CAS-ESM2-0 QM/QDM, and compute "
            "near-period-relative future change-signal diagnostics."
        )
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--china-shapefile",
        type=Path,
        default=None,
        help=(
            "Optional WGS84 China .shp file. When supplied, spatial metrics "
            "use intersection-area weighting (each cell weighted by the WGS84 "
            "geodesic area of its overlap with the boundary polygon)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    arguments.era5_file = (
        get_path("data_raw_era5_2015_2025")
        / "ERA5_monthly_201501_202512.grib"
    )
    arguments.corrected_root = (
        get_path("data_processed_bias_correction")
        / arguments.model
        / arguments.scenario
    )
    output_scenario = (
        f"{arguments.scenario}_china_mask"
        if arguments.china_shapefile is not None
        else arguments.scenario
    )
    arguments.output_root = (
        get_path("data_processed_validation") / "era5_independent_validation"
        / arguments.model
        / output_scenario
    )
    return arguments


def main() -> int:
    arguments = parse_arguments()
    corrected_files = {
        variable: find_corrected_file(
            arguments.corrected_root,
            arguments.model,
            arguments.scenario,
            variable,
        )
        for variable in VARIABLES
    }
    corrected = {
        variable: load_corrected(path, variable)
        for variable, path in corrected_files.items()
    }
    target = corrected["tas"]
    for variable in VARIABLES[1:]:
        if not same_grid(target, corrected[variable]):
            raise ValueError(f"Corrected grids differ between tas and {variable}")

    print(f"ERA5 GRIB     : {arguments.era5_file}")
    print(f"Corrected root: {arguments.corrected_root}")
    print(f"Output root   : {arguments.output_root}")
    print(f"Model/scenario: {arguments.model} / {arguments.scenario}")
    print(f"Target grid   : {target.sizes['lat']} lat x {target.sizes['lon']} lon")
    for variable, path in corrected_files.items():
        print(f"  {variable:8s}: {path}")

    weights, qc_region, weights_metadata = build_spatial_weights(
        target,
        arguments.china_shapefile,
    )
    selected_cells = int(np.count_nonzero(weights > 0.0))
    print(f"QC region     : {qc_region}")
    if arguments.china_shapefile is not None:
        print(f"China boundary: {weights_metadata['shapefile']}")
    print(
        f"QC grid cells : {selected_cells}/{weights.size} "
        f"({100.0 * selected_cells / weights.size:.2f}%)"
    )

    era5_native = prepare_era5(arguments.era5_file)
    era5_on_grid = interpolate_era5(era5_native, target)
    validate_months(
        era5_on_grid.time,
        VALIDATION_START_YEAR,
        VALIDATION_END_YEAR,
        "interpolated ERA5",
    )
    print(
        f"ERA5 coverage : {era5_on_grid.sizes['time']} months, "
        f"{era5_on_grid.sizes['lat']} lat x {era5_on_grid.sizes['lon']} lon"
    )
    for variable in VARIABLES:
        values = np.asarray(era5_on_grid[variable].values)
        qc_values = values[:, weights > 0.0]
        print(
            f"  {variable:8s}: min={np.min(qc_values):.6g}, "
            f"mean={np.mean(qc_values):.6g}, max={np.max(qc_values):.6g}, "
            f"finite={100.0 * np.isfinite(qc_values).mean():.3f}% "
            f"within {qc_region}"
        )

    if arguments.dry_run:
        print("DRY RUN COMPLETED: no files were written.")
        return 0

    arguments.output_root.mkdir(parents=True, exist_ok=True)
    era5_path = arguments.output_root / (
        f"ERA5_{arguments.model}_201501-202512.nc"
    )
    write_netcdf(era5_path, era5_on_grid)

    validation_rows: list[dict] = []
    near_baseline_rows: list[dict] = []
    for variable in VARIABLES:
        validation_rows.extend(
            make_validation_rows(
                variable,
                corrected[variable],
                era5_on_grid,
                arguments.model,
                arguments.scenario,
                weights,
                qc_region,
            )
        )
        near_baseline_rows.extend(
            make_near_baseline_rows(
                variable,
                corrected[variable],
                arguments.model,
                arguments.scenario,
                weights,
                qc_region,
            )
        )
        print(f"Validated {variable}")

    validation_summary = summarize_validation(validation_rows)
    near_baseline_summary = summarize_near_baseline(near_baseline_rows)
    validation_monthly_path = arguments.output_root / "era5_validation_monthly_metrics.csv"
    validation_summary_path = arguments.output_root / "era5_validation_summary.csv"
    signal_monthly_path = arguments.output_root / "near_baseline_signal_monthly.csv"
    signal_summary_path = arguments.output_root / "near_baseline_signal_summary.csv"
    write_csv(validation_monthly_path, validation_rows)
    write_csv(validation_summary_path, validation_summary)
    write_csv(signal_monthly_path, near_baseline_rows)
    write_csv(signal_summary_path, near_baseline_summary)

    manifest = {
        "purpose": "ERA5 2015-2025 independent validation and near-baseline signal audit",
        "model": arguments.model,
        "scenario": arguments.scenario,
        "era5_grib": str(arguments.era5_file),
        "validation_period": [VALIDATION_START_YEAR, VALIDATION_END_YEAR],
        "near_baseline_period": list(NEAR_PERIOD),
        "target_periods": {name: list(period) for name, period in TARGET_PERIODS.items()},
        "qc_region": qc_region,
        "spatial_weights": weights_metadata,
        "ssrd_alignment": "radiation valid_time aligned to atmospheric time",
        "ssrd_conversion": "J m-2 mean daily accumulation divided by 86400 to W m-2",
        "wind_definition": "sqrt(monthly-mean u10^2 + monthly-mean v10^2)",
        "corrected_files": {key: str(value) for key, value in corrected_files.items()},
        "outputs": {
            "era5_on_gcm_grid": str(era5_path),
            "validation_monthly_metrics": str(validation_monthly_path),
            "validation_summary": str(validation_summary_path),
            "near_baseline_signal_monthly": str(signal_monthly_path),
            "near_baseline_signal_summary": str(signal_summary_path),
        },
    }
    manifest_path = arguments.output_root / "era5_validation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("ERA5 INDEPENDENT VALIDATION COMPLETED")
    print("=" * 100)
    print(f"ERA5 on grid  : {era5_path}")
    print(f"Monthly QC    : {validation_monthly_path}")
    print(f"Method summary: {validation_summary_path}")
    print(f"Signal summary: {signal_summary_path}")
    print(f"Manifest      : {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
