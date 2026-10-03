#!/usr/bin/env python
r"""Plot the revised paper-baseline (B) and optimized (D) figures.

Place this script in::

    scripts/06_plotting

Inputs are read from ``results/figure_data/final_analysis`` and outputs are
written to ``results/figures``.  Routes A/C are intentionally excluded.

Main percentage conventions
---------------------------
* Paper map analogues (Fig. 5 and Fig. 8): model/grid percentage change first,
  followed by equal-weight multi-model ensemble ("口径1").
* National bars and numerical conclusions: China area mean first, followed by
  percentage change ("口径2").

The script uses the paper-compatible B route for Fig. 2--10 and produces a
separate B-versus-D optimization section.  WPD is logarithmically transformed
only for the displayed period maps; WPD change maps use untransformed WPD.
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
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    import xarray as xr
except ImportError as error:  # pragma: no cover
    raise RuntimeError(
        "pandas and xarray are required. Activate the pvwind environment."
    ) from error

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
    from matplotlib.lines import Line2D
except ImportError as error:  # pragma: no cover
    raise RuntimeError(
        "matplotlib is required. Install it in the pvwind environment."
    ) from error

try:
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
except ImportError as error:  # pragma: no cover
    raise RuntimeError(
        "cartopy is required for the map figures. Install it in pvwind."
    ) from error


SCENARIOS = ("ssp126", "ssp245", "ssp585")
METHODS = ("paper_qm", "optimized")
PERIODS = ("historical", "mid_century", "late_century")
FUTURE_PERIODS = ("mid_century", "late_century")
PERIOD_LABELS = {
    "historical": "1994–2014",
    "mid_century": "2040–2060",
    "late_century": "2080–2100",
}
SCENARIO_LABELS = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp585": "SSP5-8.5",
}
SCENARIO_COLORS = {
    "ssp126": "#2878B5",
    "ssp245": "#F39B2F",
    "ssp585": "#C82423",
}
PAPER_SCENARIO_COLORS = {
    "ssp126": "#2CA02C",
    "ssp245": "#1F77B4",
    "ssp585": "#D62728",
}
COMPLEMENTARITY_COLORS = (
    "#A50026",  # 1 very strong complementarity
    "#D73027",  # 2 strong complementarity
    "#F46D43",  # 3 moderate complementarity
    "#FDAE61",  # 4 weak complementarity
    "#E0F3F8",  # 5 weak similarity
    "#ABD9E9",  # 6 moderate similarity
    "#74ADD1",  # 7 strong similarity
    "#313695",  # 8 very strong similarity
)
COMPLEMENTARITY_LABELS = (
    "Very strong complementarity",
    "Strong complementarity",
    "Moderate complementarity",
    "Weak complementarity",
    "Weak similarity",
    "Moderate similarity",
    "Strong similarity",
    "Very strong similarity",
)
METHOD_LABELS = {"paper_qm": "B: paper QM", "optimized": "D: optimized"}
METHOD_COLORS = {"paper_qm": "#4C4C4C", "optimized": "#0072B2"}
VARIABLE_LABELS = {
    "tas_c": "Near-surface air temperature (°C)",
    "rsds": "Surface solar radiation (W m⁻²)",
    "sfcWind_10m": "10 m wind speed (m s⁻¹)",
    "pvpot": "PV power potential (W m⁻²)",
    "wpd": "Wind power density (W m⁻²)",
}
EXPECTED_ANNUAL_YEARS = np.arange(2015, 2101)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot revised paper Fig. 2--10 reproductions and B/D optimization figures."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Default: <project-root>/results/figure_data/final_analysis",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Default: <project-root>/results/figures",
    )
    parser.add_argument(
        "--china-shapefile",
        type=Path,
        help="Default: <project-root>/data/boundaries/china_qc_boundary.shp",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=["png"],
        help="One or both output formats (default: png).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def required_files(input_root: Path) -> dict[str, Path]:
    return {
        "annual_model": input_root / "final_annual_model_timeseries.csv",
        "annual_ensemble": input_root / "final_annual_ensemble_timeseries.csv",
        "maps": input_root / "final_B_D_figure_maps_1deg.nc",
        "national": input_root / "final_national_change_summary.csv",
        "method": input_root / "final_D_minus_B_summary.csv",
        "analysis_manifest": input_root / "final_analysis_manifest.json",
    }


def load_geometries(shapefile: Path):
    reader = shpreader.Reader(str(shapefile))
    geometries = list(reader.geometries())
    close = getattr(reader, "close", None)
    if close is not None:
        close()
    if not geometries:
        raise ValueError(f"No geometry found in {shapefile}")
    return geometries


def validate_inputs(
    files: dict[str, Path], shapefile: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, xr.Dataset]:
    missing = [str(path) for path in files.values() if not path.is_file()]
    if not shapefile.is_file():
        missing.append(str(shapefile))
    for suffix in (".shx", ".dbf", ".prj"):
        sidecar = shapefile.with_suffix(suffix)
        if not sidecar.is_file():
            missing.append(str(sidecar))
    if missing:
        raise FileNotFoundError("Missing required input(s):\n  " + "\n  ".join(missing))

    annual_model = pd.read_csv(files["annual_model"])
    annual = pd.read_csv(files["annual_ensemble"])
    national = pd.read_csv(files["national"])
    maps = xr.open_dataset(files["maps"])

    annual_columns = {
        "scenario",
        "method",
        "year",
        "variable",
        "model_count",
        "mean",
        "minimum",
        "maximum",
        "p10",
        "p90",
    }
    annual_model_columns = {
        "model",
        "scenario",
        "method",
        "year",
        "variable",
        "china_area_weighted_annual_mean",
    }
    national_columns = {
        "scenario",
        "method",
        "future_period",
        "variable",
        "percent_change_of_area_mean",
        "area_mean_of_gridwise_percent_change",
    }
    missing_annual = annual_columns - set(annual.columns)
    missing_annual_model = annual_model_columns - set(annual_model.columns)
    missing_national = national_columns - set(national.columns)
    if missing_annual:
        raise KeyError(f"Annual CSV missing columns: {sorted(missing_annual)}")
    if missing_annual_model:
        raise KeyError(
            f"Annual model CSV missing columns: {sorted(missing_annual_model)}"
        )
    if missing_national:
        raise KeyError(f"National CSV missing columns: {sorted(missing_national)}")

    map_variables = {
        "period_mean",
        "wpd_log10",
        "gridwise_relative_change_percent",
        "change_sign_agreement_percent",
        "optimized_minus_paper",
        "spearman_rho",
        "spearman_category",
        "china_mask",
    }
    missing_maps = map_variables - set(maps.data_vars)
    if missing_maps:
        raise KeyError(f"Map NetCDF missing variables: {sorted(missing_maps)}")

    for coordinate, expected in (
        ("scenario", set(SCENARIOS)),
        ("method", set(METHODS)),
        ("period", set(PERIODS)),
    ):
        if coordinate not in maps.coords:
            raise KeyError(f"Map NetCDF missing coordinate: {coordinate}")
        found = {str(value) for value in maps.coords[coordinate].values.tolist()}
        if not expected.issubset(found):
            raise ValueError(f"{coordinate}: expected {sorted(expected)}, found {sorted(found)}")

    annual_years = np.sort(annual["year"].unique().astype(int))
    if not np.array_equal(annual_years, EXPECTED_ANNUAL_YEARS):
        raise ValueError(
            f"Annual series should cover 2015-2100; found {annual_years[0]}-{annual_years[-1]}"
        )
    if int(annual["model_count"].min()) < 14:
        raise ValueError("Annual ensemble contains fewer than 14 valid models")
    if annual_model["model"].nunique() != 17:
        raise ValueError(
            f"Annual model table should contain 17 models; found {annual_model['model'].nunique()}"
        )
    if not bool(np.asarray(maps["china_mask"].values).astype(bool).any()):
        raise ValueError("china_mask contains no selected cells")
    return annual_model, annual, national, maps


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def save_figure(
    figure: plt.Figure,
    stem: Path,
    formats: list[str],
    dpi: int,
    outputs: list[dict[str, str]],
    description: str,
) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        figure.savefig(path, dpi=dpi if extension == "png" else None)
        outputs.append({"file": str(path), "description": description})
    plt.close(figure)


def panel_letter(index: int) -> str:
    return f"({chr(ord('a') + index)})"


def annual_subset(
    annual: pd.DataFrame, variable: str, method: str, scenario: str
) -> pd.DataFrame:
    selected = annual[
        (annual["variable"] == variable)
        & (annual["method"] == method)
        & (annual["scenario"] == scenario)
    ].sort_values("year")
    if len(selected) != EXPECTED_ANNUAL_YEARS.size:
        raise ValueError(
            f"Incomplete annual series for {variable}/{method}/{scenario}: {len(selected)} rows"
        )
    return selected


def annual_model_subset(
    annual_model: pd.DataFrame, variable: str, method: str, scenario: str
) -> pd.DataFrame:
    selected = annual_model[
        (annual_model["variable"] == variable)
        & (annual_model["method"] == method)
        & (annual_model["scenario"] == scenario)
    ].copy()
    expected = 17 * EXPECTED_ANNUAL_YEARS.size
    if len(selected) != expected or selected["model"].nunique() != 17:
        raise ValueError(
            f"Incomplete model series for {variable}/{method}/{scenario}: "
            f"{len(selected)} rows, {selected['model'].nunique()} models"
        )
    return selected


def plot_paper_climate_annual(
    annual_model: pd.DataFrame,
    annual: pd.DataFrame,
    output: Path,
    formats: list[str],
    dpi: int,
    outputs: list,
) -> None:
    variables = ("rsds", "tas_c", "sfcWind_10m")
    model_colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, 17))
    figure, axes = plt.subplots(3, 3, figsize=(13.0, 10.0), sharex=True)
    for row, scenario in enumerate(SCENARIOS):
        for column, variable in enumerate(variables):
            axis = axes[row, column]
            data = annual_subset(annual, variable, "paper_qm", scenario)
            members = annual_model_subset(
                annual_model, variable, "paper_qm", scenario
            )
            years = data["year"].to_numpy(dtype=float)
            for color, (_, member) in zip(
                model_colors, members.groupby("model", sort=True)
            ):
                member = member.sort_values("year")
                axis.plot(
                    member["year"].to_numpy(dtype=float),
                    member["china_area_weighted_annual_mean"].to_numpy(dtype=float),
                    color=color,
                    linewidth=0.45,
                    alpha=0.38,
                    zorder=1,
                )
            axis.fill_between(
                years,
                data["minimum"].to_numpy(dtype=float),
                data["maximum"].to_numpy(dtype=float),
                color="#BDBDBD",
                alpha=0.18,
                linewidth=0,
                zorder=0,
            )
            axis.plot(
                years,
                data["maximum"].to_numpy(dtype=float),
                color="#E69F00",
                linewidth=1.0,
                label="Maximum" if row == 0 and column == 0 else None,
                zorder=2,
            )
            axis.plot(
                years,
                data["minimum"].to_numpy(dtype=float),
                color="#0072B2",
                linewidth=1.0,
                label="Minimum" if row == 0 and column == 0 else None,
                zorder=2,
            )
            axis.plot(
                years,
                data["mean"].to_numpy(dtype=float),
                color="#D55E00",
                linewidth=1.45,
                label="Mean" if row == 0 and column == 0 else None,
                zorder=3,
            )
            axis.set_title(
                f"{panel_letter(row * 3 + column)} {SCENARIO_LABELS[scenario]}"
            )
            axis.set_ylabel(VARIABLE_LABELS[variable])
            axis.grid(alpha=0.20, linewidth=0.5)
            if row == 2:
                axis.set_xlabel("Year")
    axes[0, 0].legend(ncol=3, loc="best", frameon=False)
    figure.suptitle(
        "Fig. 2 reproduction — projected climate factors from 17 GCMs (route B)",
        y=0.995,
    )
    figure.tight_layout()
    save_figure(
        figure,
        output / "Fig02_climate_factors_annual_route_B",
        formats,
        dpi,
        outputs,
        "Paper Fig. 2 reproduction: 17 member lines plus maximum, minimum, and mean, route B.",
    )


def plot_paper_energy_annual(
    annual: pd.DataFrame,
    variable: str,
    figure_number: str,
    output: Path,
    formats: list[str],
    dpi: int,
    outputs: list,
) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 4.3))
    for scenario in SCENARIOS:
        data = annual_subset(annual, variable, "paper_qm", scenario)
        years = data["year"].to_numpy(dtype=float)
        axis.plot(
            years,
            data["mean"].to_numpy(dtype=float),
            color=PAPER_SCENARIO_COLORS[scenario],
            linewidth=1.5,
            label=SCENARIO_LABELS[scenario],
        )
    axis.set_xlabel("Year")
    axis.set_ylabel(VARIABLE_LABELS[variable])
    axis.grid(alpha=0.22, linewidth=0.5)
    axis.legend(ncol=3, loc="best", frameon=False)
    axis.set_title(f"Fig. {figure_number} reproduction — annual {variable.upper()} (route B)")
    figure.tight_layout()
    save_figure(
        figure,
        output / f"Fig{figure_number}_{variable}_annual_route_B",
        formats,
        dpi,
        outputs,
        f"Paper Fig. {figure_number} reproduction: route-B annual {variable} ensemble mean.",
    )


def configure_map_axis(axis, geometries, extent: tuple[float, float, float, float]) -> None:
    axis.set_extent(extent, crs=ccrs.PlateCarree())
    axis.add_geometries(
        geometries,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="black",
        linewidth=0.55,
        zorder=4,
    )
    axis.set_xticks([75, 90, 105, 120, 135], crs=ccrs.PlateCarree())
    axis.set_yticks([20, 30, 40, 50], crs=ccrs.PlateCarree())
    axis.xaxis.set_major_formatter(LongitudeFormatter())
    axis.yaxis.set_major_formatter(LatitudeFormatter())
    axis.tick_params(length=2.5, pad=1.5)


def masked(data: xr.DataArray, maps: xr.Dataset) -> xr.DataArray:
    return data.where(maps["china_mask"].astype(bool))


def finite_values(arrays: list[xr.DataArray]) -> np.ndarray:
    pieces = []
    for data in arrays:
        values = np.asarray(data.values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            pieces.append(values)
    if not pieces:
        raise ValueError("No finite values available for map color scaling")
    return np.concatenate(pieces)


def sequential_limits(arrays: list[xr.DataArray]) -> tuple[float, float]:
    values = finite_values(arrays)
    low, high = np.quantile(values, [0.01, 0.99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(values)), float(np.max(values))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def symmetric_limit(arrays: list[xr.DataArray], quantile: float = 0.99) -> float:
    values = finite_values(arrays)
    limit = float(np.quantile(np.abs(values), quantile))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.max(np.abs(values)))
    return limit if limit > 0 else 1.0


def period_map_data(
    maps: xr.Dataset, variable: str, scenario: str, method: str, period: str
) -> xr.DataArray:
    if variable == "wpd":
        return masked(
            maps["wpd_log10"].sel(
                {"scenario": scenario, "method": method, "period": period}
            ),
            maps,
        )
    return masked(
        maps["period_mean"].sel(
            {
                "scenario": scenario,
                "method": method,
                "period": period,
                "energy_variable": variable,
            }
        ),
        maps,
    )


def nine_period_panels(maps: xr.Dataset, variable: str, method: str):
    panels = []
    for row, scenario in enumerate(SCENARIOS):
        for column, period in enumerate(PERIODS):
            panels.append((row, column, scenario, period))
    arrays = [
        period_map_data(maps, variable, scenario, method, period)
        for _, _, scenario, period in panels
    ]
    return panels, arrays


def plot_period_maps(
    maps: xr.Dataset,
    geometries,
    extent,
    variable: str,
    figure_number: str,
    output: Path,
    formats: list[str],
    dpi: int,
    outputs: list,
) -> None:
    panels, arrays = nine_period_panels(maps, variable, "paper_qm")
    low, high = sequential_limits(arrays)
    cmap = "YlOrRd" if variable == "pvpot" else "viridis"
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(11.4, 8.6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    mappable = None
    for (row, column, scenario, period), data in zip(panels, arrays):
        axis = axes[row, column]
        configure_map_axis(axis, geometries, extent)
        mappable = axis.pcolormesh(
            data.lon,
            data.lat,
            data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=low,
            vmax=high,
            shading="auto",
        )
        index = column * len(SCENARIOS) + row
        axis.set_title(
            f"{panel_letter(index)} {SCENARIO_LABELS[scenario]} | {PERIOD_LABELS[period]}"
        )
    label = (
        "PV power potential (W m⁻²)"
        if variable == "pvpot"
        else "log₁₀(WPD / W m⁻²)"
    )
    figure.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        fraction=0.030,
        pad=0.025,
        label=label,
        extend="both",
    )
    figure.suptitle(
        f"Fig. {figure_number} reproduction — {variable.upper()} spatial distribution (route B)",
        y=0.99,
    )
    figure.subplots_adjust(left=0.04, right=0.90, top=0.91, bottom=0.06, wspace=0.13, hspace=0.25)
    save_figure(
        figure,
        output / f"Fig{figure_number}_{variable}_period_maps_route_B",
        formats,
        dpi,
        outputs,
        f"Paper Fig. {figure_number} reproduction: 3 scenarios x 3 periods; WPD log10 is display-only.",
    )


def plot_change_maps(
    maps: xr.Dataset,
    geometries,
    extent,
    variable: str,
    figure_number: str,
    output: Path,
    formats: list[str],
    dpi: int,
    outputs: list,
) -> None:
    panels = []
    arrays = []
    for row, scenario in enumerate(SCENARIOS):
        for column, period in enumerate(FUTURE_PERIODS):
            data = masked(
                maps["gridwise_relative_change_percent"].sel(
                    {
                        "scenario": scenario,
                        "method": "paper_qm",
                        "future_period": period,
                        "energy_variable": variable,
                    }
                ),
                maps,
            )
            panels.append((row, column, scenario, period))
            arrays.append(data)
    limit = symmetric_limit(arrays)
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(8.8, 8.7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    mappable = None
    for (row, column, scenario, period), data in zip(panels, arrays):
        axis = axes[row, column]
        configure_map_axis(axis, geometries, extent)
        mappable = axis.pcolormesh(
            data.lon,
            data.lat,
            data,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            shading="auto",
        )
        index = column * len(SCENARIOS) + row
        axis.set_title(f"{panel_letter(index)} {SCENARIO_LABELS[scenario]} | {PERIOD_LABELS[period]}")
    figure.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        fraction=0.040,
        pad=0.030,
        label="Gridwise relative change (%)",
        extend="both",
    )
    figure.suptitle(
        f"Fig. {figure_number} reproduction — {variable.upper()} change from 1994–2014 "
        "(route B; grid-cell percentage definition)",
        y=0.99,
    )
    figure.subplots_adjust(left=0.06, right=0.88, top=0.91, bottom=0.06, wspace=0.15, hspace=0.26)
    save_figure(
        figure,
        output / f"Fig{figure_number}_{variable}_gridwise_change_route_B",
        formats,
        dpi,
        outputs,
        f"Paper Fig. {figure_number} reproduction: per-model/grid relative change then ensemble (definition 1).",
    )


def complementarity_panels(
    maps: xr.Dataset, definition: str, method: str
) -> tuple[list[tuple[int, int, str, str]], list[xr.DataArray]]:
    panels = []
    for row, scenario in enumerate(SCENARIOS):
        for column, period in enumerate(PERIODS):
            panels.append((row, column, scenario, period))
    arrays = []
    for _, _, scenario, period in panels:
        arrays.append(
            masked(
                maps["spearman_category"].sel(
                    {
                        "scenario": scenario,
                        "method": method,
                        "period": period,
                        "definition": definition,
                    }
                ),
                maps,
            )
        )
    return panels, arrays


def plot_complementarity(
    maps: xr.Dataset,
    geometries,
    extent,
    definition: str,
    figure_number: str,
    output: Path,
    formats: list[str],
    dpi: int,
    outputs: list,
) -> None:
    panels, arrays = complementarity_panels(maps, definition, "paper_qm")
    cmap = ListedColormap(COMPLEMENTARITY_COLORS, name="paper_complementarity")
    norm = BoundaryNorm(np.arange(0.5, 9.5, 1.0), cmap.N)
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(11.8, 8.8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    mappable = None
    for (row, column, scenario, period), data in zip(panels, arrays):
        axis = axes[row, column]
        configure_map_axis(axis, geometries, extent)
        mappable = axis.pcolormesh(
            data.lon,
            data.lat,
            data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            shading="auto",
        )
        index = column * len(SCENARIOS) + row
        axis.set_title(
            f"{panel_letter(index)} {SCENARIO_LABELS[scenario]} | {PERIOD_LABELS[period]}"
        )
    colorbar = figure.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        fraction=0.040,
        pad=0.025,
        ticks=np.arange(1, 9),
    )
    colorbar.ax.set_yticklabels(COMPLEMENTARITY_LABELS)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_label("Paper Table 3 similarity/complementarity class")
    scale = "seasonal" if definition == "seasonal_full" else "monthly"
    samples = "84 seasonal samples" if definition == "seasonal_full" else "252 monthly samples"
    figure.suptitle(
        f"Fig. {figure_number} reproduction — {scale} solar–wind complementarity "
        f"(route B; {samples})",
        y=0.99,
    )
    figure.subplots_adjust(left=0.04, right=0.79, top=0.91, bottom=0.06, wspace=0.13, hspace=0.25)
    save_figure(
        figure,
        output / f"Fig{figure_number}_{scale}_complementarity_route_B",
        formats,
        dpi,
        outputs,
        f"Paper Fig. {figure_number} reproduction: {definition} classified with paper Table 3, route B.",
    )


def plot_annual_B_D(
    annual: pd.DataFrame, output: Path, formats: list[str], dpi: int, outputs: list
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.0), sharex=True)
    for row, variable in enumerate(("pvpot", "wpd")):
        for column, scenario in enumerate(SCENARIOS):
            axis = axes[row, column]
            for method in METHODS:
                data = annual_subset(annual, variable, method, scenario)
                years = data["year"].to_numpy(dtype=float)
                axis.fill_between(
                    years,
                    data["p10"].to_numpy(dtype=float),
                    data["p90"].to_numpy(dtype=float),
                    color=METHOD_COLORS[method],
                    alpha=0.10,
                    linewidth=0,
                )
                axis.plot(
                    years,
                    data["mean"].to_numpy(dtype=float),
                    color=METHOD_COLORS[method],
                    linewidth=1.35,
                    linestyle="-" if method == "paper_qm" else "--",
                    label=METHOD_LABELS[method],
                )
            axis.set_title(f"{panel_letter(row * 3 + column)} {SCENARIO_LABELS[scenario]}")
            axis.grid(alpha=0.22, linewidth=0.5)
            if column == 0:
                axis.set_ylabel(VARIABLE_LABELS[variable])
            if row == 1:
                axis.set_xlabel("Year")
    axes[0, 0].legend(frameon=False, loc="best")
    figure.suptitle("Optimization comparison — annual route B versus route D", y=0.995)
    figure.tight_layout()
    save_figure(
        figure,
        output / "Opt01_annual_energy_B_vs_D",
        formats,
        dpi,
        outputs,
        "Annual PVpot and WPD: ensemble mean and P10-P90 for B versus D.",
    )


def plot_method_difference_maps(
    maps: xr.Dataset,
    geometries,
    extent,
    variable: str,
    output: Path,
    formats: list[str],
    dpi: int,
    outputs: list,
) -> None:
    panels = []
    arrays = []
    for row, period in enumerate(FUTURE_PERIODS):
        for column, scenario in enumerate(SCENARIOS):
            data = masked(
                maps["optimized_minus_paper"].sel(
                    {
                        "scenario": scenario,
                        "period": period,
                        "energy_variable": variable,
                        "method_difference_metric": "period_relative_difference_percent",
                    }
                ),
                maps,
            )
            panels.append((row, column, scenario, period))
            arrays.append(data)
    limit = symmetric_limit(arrays)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.8, 6.6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    mappable = None
    for index, ((row, column, scenario, period), data) in enumerate(zip(panels, arrays)):
        axis = axes[row, column]
        configure_map_axis(axis, geometries, extent)
        mappable = axis.pcolormesh(
            data.lon,
            data.lat,
            data,
            transform=ccrs.PlateCarree(),
            cmap="PuOr_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            shading="auto",
        )
        axis.set_title(
            f"{panel_letter(index)} {SCENARIO_LABELS[scenario]} | {PERIOD_LABELS[period]}"
        )
    figure.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.055,
        pad=0.08,
        label="Route D relative to route B (%)",
        extend="both",
    )
    figure.suptitle(f"Optimization difference — {variable.upper()} period mean (D − B)", y=0.99)
    figure.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.15, wspace=0.10, hspace=0.22)
    save_figure(
        figure,
        output / f"Opt02_{variable}_period_relative_difference_D_minus_B",
        formats,
        dpi,
        outputs,
        f"Spatial period-mean relative difference between optimized D and paper B for {variable}.",
    )


def plot_national_change(
    national: pd.DataFrame,
    output: Path,
    formats: list[str],
    dpi: int,
    outputs: list,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharey="row")
    width = 0.34
    x = np.arange(len(SCENARIOS), dtype=float)
    for row, variable in enumerate(("pvpot", "wpd")):
        for column, period in enumerate(FUTURE_PERIODS):
            axis = axes[row, column]
            for offset_index, method in enumerate(METHODS):
                values = []
                for scenario in SCENARIOS:
                    selected = national[
                        (national["scenario"] == scenario)
                        & (national["method"] == method)
                        & (national["future_period"] == period)
                        & (national["variable"] == variable)
                    ]
                    if len(selected) != 1:
                        raise ValueError(
                            f"Expected one national row for {scenario}/{method}/{period}/{variable}; found {len(selected)}"
                        )
                    values.append(float(selected.iloc[0]["percent_change_of_area_mean"]))
                offset = (-0.5 if offset_index == 0 else 0.5) * width
                axis.bar(
                    x + offset,
                    values,
                    width=width,
                    color=METHOD_COLORS[method],
                    alpha=0.86,
                    label=METHOD_LABELS[method],
                )
            axis.axhline(0.0, color="black", linewidth=0.7)
            axis.set_xticks(x, [SCENARIO_LABELS[s] for s in SCENARIOS])
            axis.set_title(f"{panel_letter(row * 2 + column)} {PERIOD_LABELS[period]}")
            axis.grid(axis="y", alpha=0.22, linewidth=0.5)
            if column == 0:
                axis.set_ylabel(f"{variable.upper()} national change (%)")
    axes[0, 0].legend(frameon=False, ncol=2, loc="best")
    figure.suptitle(
        "National changes — China area mean first, then percentage change (definition 2)",
        y=0.995,
    )
    figure.tight_layout()
    save_figure(
        figure,
        output / "Opt03_national_change_B_vs_D_definition2",
        formats,
        dpi,
        outputs,
        "National PVpot/WPD change using percent change of China area mean (definition 2).",
    )


def plot_complementarity_difference(
    maps: xr.Dataset,
    geometries,
    extent,
    output: Path,
    formats: list[str],
    dpi: int,
    outputs: list,
) -> None:
    definitions = ("seasonal_full", "monthly_full")
    arrays = []
    panels = []
    for row, definition in enumerate(definitions):
        for column, scenario in enumerate(SCENARIOS):
            optimized = maps["spearman_rho"].sel(
                {
                    "scenario": scenario,
                    "method": "optimized",
                    "period": "late_century",
                    "definition": definition,
                }
            )
            paper = maps["spearman_rho"].sel(
                {
                    "scenario": scenario,
                    "method": "paper_qm",
                    "period": "late_century",
                    "definition": definition,
                }
            )
            arrays.append(masked(optimized - paper, maps))
            panels.append((row, column, definition, scenario))
    limit = symmetric_limit(arrays)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.8, 6.6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    mappable = None
    for index, ((row, column, definition, scenario), data) in enumerate(zip(panels, arrays)):
        axis = axes[row, column]
        configure_map_axis(axis, geometries, extent)
        mappable = axis.pcolormesh(
            data.lon,
            data.lat,
            data,
            transform=ccrs.PlateCarree(),
            cmap="BrBG",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            shading="auto",
        )
        scale = "Seasonal" if definition == "seasonal_full" else "Monthly"
        axis.set_title(f"{panel_letter(index)} {scale} | {SCENARIO_LABELS[scenario]}")
    figure.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.055,
        pad=0.08,
        label="Spearman ρ difference (D − B)",
        extend="both",
    )
    figure.suptitle("Optimization difference — late-century complementarity (2080–2100)", y=0.99)
    figure.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.15, wspace=0.10, hspace=0.22)
    save_figure(
        figure,
        output / "Opt04_complementarity_rho_D_minus_B_late_century",
        formats,
        dpi,
        outputs,
        "Late-century seasonal/monthly Spearman-rho difference, optimized D minus paper B.",
    )


def main() -> int:
    arguments = parse_arguments()
    if arguments.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    script_dir = Path(__file__).resolve().parent
    project_root = PROJECT_ROOT
    input_root = (
        arguments.input_root.expanduser().resolve()
        if arguments.input_root
        else get_path("results_figure_data_final_analysis")
    )
    output_root = (
        arguments.output_root.expanduser().resolve()
        if arguments.output_root
        else get_path("results_figures")
    )
    shapefile = (
        arguments.china_shapefile.expanduser().resolve()
        if arguments.china_shapefile
        else get_path("china_shapefile")
    )
    files = required_files(input_root)

    print("=" * 100)
    print("FINAL PAPER-REPRODUCTION AND OPTIMIZATION FIGURES V2")
    print("=" * 100)
    print(f"Input root      : {input_root}")
    print(f"Output root     : {output_root}")
    print(f"China boundary  : {shapefile}")
    print("Paper route     : B = paper_qm, model-first")
    print("Optimized route : D = optimized, model-first")
    print("Excluded routes : A/C ensemble-first")
    print(f"Formats         : {', '.join(arguments.formats)}")
    print(f"PNG DPI         : {arguments.dpi}")

    annual_model, annual, national, maps = validate_inputs(files, shapefile)
    try:
        print(f"Annual model rows: {len(annual_model)}")
        print(f"Annual rows     : {len(annual)}")
        print(f"National rows   : {len(national)}")
        print(f"Map grid        : {maps.sizes['lat']} lat x {maps.sizes['lon']} lon")
        print(
            f"China cells     : {int(maps['china_mask'].astype(bool).sum().values)}/"
            f"{maps.sizes['lat'] * maps.sizes['lon']}"
        )
        if arguments.dry_run:
            print("DRY RUN COMPLETED: figure inputs, dimensions, columns, and masks are valid.")
            print("No figures were written.")
            return 0

        set_style()
        geometries = load_geometries(shapefile)
        valid_lon = np.asarray(maps.lon.values, dtype=float)
        valid_lat = np.asarray(maps.lat.values, dtype=float)
        extent = (
            max(70.0, float(valid_lon.min())),
            min(140.0, float(valid_lon.max())),
            max(14.0, float(valid_lat.min())),
            min(55.0, float(valid_lat.max())),
        )
        paper_root = output_root / "paper_baseline"
        optimization_root = output_root / "optimization"
        outputs: list[dict[str, str]] = []

        print("Plotting paper-baseline figures 2-10 ...")
        plot_paper_climate_annual(
            annual_model,
            annual,
            paper_root,
            arguments.formats,
            arguments.dpi,
            outputs,
        )
        plot_paper_energy_annual(annual, "pvpot", "03", paper_root, arguments.formats, arguments.dpi, outputs)
        plot_period_maps(maps, geometries, extent, "pvpot", "04", paper_root, arguments.formats, arguments.dpi, outputs)
        plot_change_maps(maps, geometries, extent, "pvpot", "05", paper_root, arguments.formats, arguments.dpi, outputs)
        plot_paper_energy_annual(annual, "wpd", "06", paper_root, arguments.formats, arguments.dpi, outputs)
        plot_period_maps(maps, geometries, extent, "wpd", "07", paper_root, arguments.formats, arguments.dpi, outputs)
        plot_change_maps(maps, geometries, extent, "wpd", "08", paper_root, arguments.formats, arguments.dpi, outputs)
        plot_complementarity(maps, geometries, extent, "seasonal_full", "09", paper_root, arguments.formats, arguments.dpi, outputs)
        plot_complementarity(maps, geometries, extent, "monthly_full", "10", paper_root, arguments.formats, arguments.dpi, outputs)

        print("Plotting B/D optimization figures ...")
        plot_annual_B_D(annual, optimization_root, arguments.formats, arguments.dpi, outputs)
        plot_method_difference_maps(maps, geometries, extent, "pvpot", optimization_root, arguments.formats, arguments.dpi, outputs)
        plot_method_difference_maps(maps, geometries, extent, "wpd", optimization_root, arguments.formats, arguments.dpi, outputs)
        plot_national_change(national, optimization_root, arguments.formats, arguments.dpi, outputs)
        plot_complementarity_difference(maps, geometries, extent, optimization_root, arguments.formats, arguments.dpi, outputs)

        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "input_root": str(input_root),
            "output_root": str(output_root),
            "routes": {
                "B": "paper_qm, model-level energy/complementarity before ensemble",
                "D": "optimized, model-level energy/complementarity before ensemble",
            },
            "excluded_routes": ["A", "C"],
            "paper_figures": [f"Fig{i:02d}" for i in range(2, 11)],
            "paper_layouts": {
                "Fig02": "3 scenarios x 3 climate variables; 17 members plus max/min/mean",
                "Fig03_Fig06": "three scenario ensemble-mean curves without uncertainty envelope",
                "Fig04_Fig07": "3 scenarios x 3 periods",
                "Fig05_Fig08": "3 scenarios x 2 future periods",
                "Fig09_Fig10": "3 scenarios x 3 periods; discrete paper Table-3 classes",
            },
            "periods": {key: PERIOD_LABELS[key] for key in PERIODS},
            "map_percentage_definition": (
                "per-model grid-cell relative change followed by equal-weight ensemble (口径1)"
            ),
            "national_percentage_definition": (
                "percent change of China cosine-latitude-weighted area mean (口径2)"
            ),
            "wpd_log_rule": (
                "log10 applied only to displayed WPD period maps; WPD changes use raw WPD"
            ),
            "complementarity": {
                "seasonal_full": "84 seasonal samples; DJF December assigned to following season-year",
                "monthly_full": "252 monthly samples",
                "display": "eight discrete similarity/complementarity classes from paper Table 3",
                "zero_rule": "rho == 0 is assigned to weak complementarity",
            },
            "color_scaling": (
                "shared within each multi-panel figure; sequential maps use 1st-99th percentiles; "
                "difference maps use symmetric 99th absolute percentile; clipped values are shown with extend"
            ),
            "formats": list(arguments.formats),
            "dpi": int(arguments.dpi),
            "outputs": outputs,
        }
        manifest_path = output_root / "figure_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print("=" * 100)
        print("FINAL FIGURES V2 COMPLETED")
        print("=" * 100)
        print(f"Paper figures       : {paper_root}")
        print(f"Optimization figures: {optimization_root}")
        print(f"Figure manifest     : {manifest_path}")
        print(f"Files written       : {len(outputs)} figure file(s)")
        print("Next: visually inspect the figures and then prepare the project README/results text.")
        return 0
    finally:
        maps.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # pragma: no cover
        print(f"ERROR: {error}", file=sys.stderr)
        raise
