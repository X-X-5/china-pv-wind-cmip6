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
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, PathPatch
    from matplotlib.path import Path as MplPath
except ImportError as error:  # pragma: no cover
    raise RuntimeError(
        "matplotlib is required. Install it in the pvwind environment."
    ) from error

try:
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
    from shapely.geometry.polygon import orient
except ImportError as error:  # pragma: no cover
    raise RuntimeError(
        "cartopy and shapely are required for the map figures. Install them in pvwind."
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
METHOD_COLORS = {"paper_qm": "#D55E00", "optimized": "#0072B2"}
ROUTE_TAG = {"paper_qm": "route_B", "optimized": "route_D"}
ROUTE_NAME = {"paper_qm": "B", "optimized": "D"}
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
        "china_intersects",
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
    if not bool(np.asarray(maps["china_intersects"].values).astype(bool).any()):
        raise ValueError("china_intersects contains no selected cells")
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
    method: str,
    ylims: dict[str, tuple[float, float]],
) -> None:
    variables = ("rsds", "tas_c", "sfcWind_10m")
    model_colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, 17))
    figure, axes = plt.subplots(3, 3, figsize=(13.0, 10.0), sharex=True)
    for row, scenario in enumerate(SCENARIOS):
        for column, variable in enumerate(variables):
            axis = axes[row, column]
            data = annual_subset(annual, variable, method, scenario)
            members = annual_model_subset(
                annual_model, variable, method, scenario
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
            axis.set_ylim(ylims[variable])
            if row == 2:
                axis.set_xlabel("Year")
    axes[0, 0].legend(ncol=3, loc="best", frameon=False)
    figure.suptitle(
        f"Fig. 2 reproduction — projected climate factors from 17 GCMs "
        f"(route {ROUTE_NAME[method]})",
        y=0.995,
    )
    figure.tight_layout()
    save_figure(
        figure,
        output / f"Fig02_climate_factors_annual_{ROUTE_TAG[method]}",
        formats,
        dpi,
        outputs,
        f"Paper Fig. 2 reproduction: 17 member lines plus maximum, minimum, and mean, "
        f"route {ROUTE_NAME[method]}.",
    )


def plot_paper_energy_annual(
    annual: pd.DataFrame,
    variable: str,
    figure_number: str,
    output: Path,
    formats: list[str],
    dpi: int,
    outputs: list,
    method: str,
    ylim: tuple[float, float],
) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 4.3))
    for scenario in SCENARIOS:
        data = annual_subset(annual, variable, method, scenario)
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
    axis.set_ylim(ylim)
    axis.grid(alpha=0.22, linewidth=0.5)
    axis.legend(ncol=3, loc="best", frameon=False)
    axis.set_title(
        f"Fig. {figure_number} reproduction — annual {variable.upper()} "
        f"(route {ROUTE_NAME[method]})"
    )
    figure.tight_layout()
    save_figure(
        figure,
        output / f"Fig{figure_number}_{variable}_annual_{ROUTE_TAG[method]}",
        formats,
        dpi,
        outputs,
        f"Paper Fig. {figure_number} reproduction: route-{ROUTE_NAME[method]} "
        f"annual {variable} ensemble mean.",
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


_CHINA_BOUNDARY_PATHS: dict[int, MplPath] = {}


def _china_boundary_to_path(geometries) -> MplPath:
    """Convert the China boundary Shapely geometry into one matplotlib Path.

    The full Polygon/MultiPolygon is kept — mainland, islands, and any interior
    holes — so the clip uses exactly the geometry drawn as the black outline.
    Coordinates remain in lon/lat (PlateCarree data space).
    """
    rings: list[tuple[np.ndarray, list[int]]] = []

    def add_polygon(polygon) -> None:
        polygon = orient(polygon, sign=1.0)  # exterior CCW, holes CW
        for ring in (polygon.exterior, *polygon.interiors):
            coords = np.asarray(ring.coords)  # keep the repeated closing vertex
            codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(coords) - 2) + [MplPath.CLOSEPOLY]
            rings.append((coords, codes))

    def add_geometry(geometry) -> None:
        if geometry.is_empty:
            return
        if geometry.geom_type == "Polygon":
            add_polygon(geometry)
        elif geometry.geom_type == "MultiPolygon":
            for polygon in geometry.geoms:
                add_polygon(polygon)
        elif geometry.geom_type == "GeometryCollection":
            for part in geometry.geoms:
                add_geometry(part)

    for geometry in geometries:
        add_geometry(geometry)

    if not rings:
        raise ValueError("China boundary shapefile contains no polygon geometry")
    vertices = np.concatenate([coords for coords, _ in rings], axis=0)
    codes = np.concatenate([code for _, code in rings])
    return MplPath(vertices, codes)


def clip_mesh_to_china(mappable, geometries, axis) -> None:
    """Clip a map QuadMesh to the China boundary, preserving the 1° grid.

    Selection uses every cell whose area intersects the China polygon
    (``china_intersects``); this final polygon clip only trims the drawn mesh
    to the true coastline so grid colours stop spilling past the border.
    """
    key = id(geometries)
    path = _CHINA_BOUNDARY_PATHS.get(key)
    if path is None:
        path = _china_boundary_to_path(geometries)
        _CHINA_BOUNDARY_PATHS[key] = path
    mappable.set_clip_path(
        PathPatch(path, transform=axis.transData, facecolor="none", edgecolor="none")
    )


def masked(data: xr.DataArray, maps: xr.Dataset) -> xr.DataArray:
    return data.where(maps["china_intersects"].astype(bool))


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
    method: str,
    low: float,
    high: float,
) -> None:
    panels, arrays = nine_period_panels(maps, variable, method)
    cmap = "YlOrRd" if variable == "pvpot" else "viridis"
    figure = plt.figure(figsize=(11.4, 8.6))
    grid = GridSpec(
        3,
        4,
        figure=figure,
        left=0.04,
        right=0.90,
        top=0.91,
        bottom=0.06,
        width_ratios=[1.0, 1.0, 1.0, 0.07],
        wspace=0.13,
        hspace=0.25,
    )
    axes = [
        [figure.add_subplot(grid[row, column], projection=ccrs.PlateCarree()) for column in range(3)]
        for row in range(3)
    ]
    colorbar_axis = figure.add_subplot(grid[:, 3])
    mappable = None
    for (row, column, scenario, period), data in zip(panels, arrays):
        axis = axes[row][column]
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
        clip_mesh_to_china(mappable, geometries, axis)
        index = column * len(SCENARIOS) + row
        axis.set_title(
            f"{panel_letter(index)} {SCENARIO_LABELS[scenario]} | {PERIOD_LABELS[period]}"
        )
    label = (
        "PV power potential (W m⁻²)"
        if variable == "pvpot"
        else "log₁₀(WPD / W m⁻²)"
    )
    colorbar = figure.colorbar(
        mappable,
        cax=colorbar_axis,
        orientation="vertical",
        label=label,
        extend="both",
    )
    colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        f"Fig. {figure_number} reproduction — {variable.upper()} spatial distribution "
        f"(route {ROUTE_NAME[method]})",
        y=0.99,
    )
    save_figure(
        figure,
        output / f"Fig{figure_number}_{variable}_period_maps_{ROUTE_TAG[method]}",
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
    method: str,
    limit: float,
) -> None:
    panels = [
        (row, column, scenario, period)
        for row, scenario in enumerate(SCENARIOS)
        for column, period in enumerate(FUTURE_PERIODS)
    ]
    arrays = change_arrays(maps, variable, method)
    figure = plt.figure(figsize=(8.8, 8.7))
    grid = GridSpec(
        3,
        3,
        figure=figure,
        left=0.06,
        right=0.90,
        top=0.91,
        bottom=0.06,
        width_ratios=[1.0, 1.0, 0.07],
        wspace=0.15,
        hspace=0.26,
    )
    axes = [
        [figure.add_subplot(grid[row, column], projection=ccrs.PlateCarree()) for column in range(2)]
        for row in range(3)
    ]
    colorbar_axis = figure.add_subplot(grid[:, 2])
    mappable = None
    for (row, column, scenario, period), data in zip(panels, arrays):
        axis = axes[row][column]
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
        clip_mesh_to_china(mappable, geometries, axis)
        index = column * len(SCENARIOS) + row
        axis.set_title(f"{panel_letter(index)} {SCENARIO_LABELS[scenario]} | {PERIOD_LABELS[period]}")
    colorbar = figure.colorbar(
        mappable,
        cax=colorbar_axis,
        orientation="vertical",
        label="Gridwise relative change (%)",
        extend="both",
    )
    colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        f"Fig. {figure_number} reproduction — {variable.upper()} change from 1994–2014 "
        f"(route {ROUTE_NAME[method]}; grid-cell percentage definition)",
        y=0.99,
    )
    save_figure(
        figure,
        output / f"Fig{figure_number}_{variable}_gridwise_change_{ROUTE_TAG[method]}",
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
    method: str,
) -> None:
    panels, arrays = complementarity_panels(maps, definition, method)
    cmap = ListedColormap(COMPLEMENTARITY_COLORS, name="paper_complementarity")
    norm = BoundaryNorm(np.arange(0.5, 9.5, 1.0), cmap.N)
    figure = plt.figure(figsize=(13.4, 8.8))
    grid = GridSpec(
        3,
        4,
        figure=figure,
        left=0.04,
        right=0.82,
        top=0.91,
        bottom=0.06,
        width_ratios=[1.0, 1.0, 1.0, 0.14],
        wspace=0.13,
        hspace=0.25,
    )
    axes = [
        [figure.add_subplot(grid[row, column], projection=ccrs.PlateCarree()) for column in range(3)]
        for row in range(3)
    ]
    colorbar_axis = figure.add_subplot(grid[:, 3])
    mappable = None
    for (row, column, scenario, period), data in zip(panels, arrays):
        axis = axes[row][column]
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
        clip_mesh_to_china(mappable, geometries, axis)
        index = column * len(SCENARIOS) + row
        axis.set_title(
            f"{panel_letter(index)} {SCENARIO_LABELS[scenario]} | {PERIOD_LABELS[period]}"
        )
    colorbar = figure.colorbar(
        mappable,
        cax=colorbar_axis,
        orientation="vertical",
        ticks=np.arange(1, 9),
    )
    colorbar.ax.set_yticklabels(COMPLEMENTARITY_LABELS)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_label("Paper Table 3 similarity/complementarity class")
    scale = "seasonal" if definition == "seasonal_full" else "monthly"
    samples = "84 seasonal samples" if definition == "seasonal_full" else "252 monthly samples"
    figure.suptitle(
        f"Fig. {figure_number} reproduction — {scale} solar–wind complementarity "
        f"(route {ROUTE_NAME[method]}; {samples})",
        y=0.99,
    )
    save_figure(
        figure,
        output / f"Fig{figure_number}_{scale}_complementarity_{ROUTE_TAG[method]}",
        formats,
        dpi,
        outputs,
        f"Paper Fig. {figure_number} reproduction: {definition} classified with paper Table 3, "
        f"route {ROUTE_NAME[method]}.",
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
                    alpha=0.14,
                    linewidth=0.6,
                    edgecolor=METHOD_COLORS[method],
                )
                axis.plot(
                    years,
                    data["mean"].to_numpy(dtype=float),
                    color=METHOD_COLORS[method],
                    linewidth=1.35,
                    linestyle="-" if method == "paper_qm" else "--",
                )
            axis.set_title(f"{panel_letter(row * 3 + column)} {SCENARIO_LABELS[scenario]}")
            axis.grid(alpha=0.22, linewidth=0.5)
            if column == 0:
                axis.set_ylabel(VARIABLE_LABELS[variable])
            if row == 1:
                axis.set_xlabel("Year")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS["paper_qm"],
            linewidth=1.35,
            linestyle="-",
            label="Route B ensemble mean",
        ),
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS["optimized"],
            linewidth=1.35,
            linestyle="--",
            label="Route D ensemble mean",
        ),
        Patch(
            facecolor=METHOD_COLORS["paper_qm"],
            alpha=0.14,
            label="Route B inter-model P10–P90",
        ),
        Patch(
            facecolor=METHOD_COLORS["optimized"],
            alpha=0.14,
            label="Route D inter-model P10–P90",
        ),
        Patch(
            facecolor="#808080",
            alpha=0.30,
            label="Overlap of Route B and Route D P10–P90 bands",
        ),
    ]
    figure.suptitle("Optimization comparison — annual route B versus route D", y=0.995)
    figure.tight_layout(rect=[0.0, 0.20, 1.0, 0.985])
    figure.legend(
        handles=legend_handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.14),
        frameon=False,
        columnspacing=1.4,
        handlelength=1.8,
        handletextpad=0.6,
    )
    figure.text(
        0.5,
        0.03,
        "Lines show the 17-model ensemble means; shading shows the inter-model P10–P90 ranges. "
        "Grey shading indicates overlap between the two ranges.",
        ha="center",
        va="center",
        fontsize=8,
    )
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
        clip_mesh_to_china(mappable, geometries, axis)
        axis.set_title(
            f"{panel_letter(index)} {SCENARIO_LABELS[scenario]} | {PERIOD_LABELS[period]}"
        )
    figure.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.22, wspace=0.10, hspace=0.22)
    cax = figure.add_axes([0.25, 0.09, 0.50, 0.02])
    figure.colorbar(
        mappable,
        cax=cax,
        orientation="horizontal",
        label="Route D relative to route B (%)",
        extend="both",
    )
    figure.suptitle(f"Optimization difference — {variable.upper()} period mean (D − B)", y=0.99)
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
                )
            axis.axhline(0.0, color="black", linewidth=0.7)
            axis.set_xticks(x, [SCENARIO_LABELS[s] for s in SCENARIOS])
            axis.set_title(f"{panel_letter(row * 2 + column)} {PERIOD_LABELS[period]}")
            axis.grid(axis="y", alpha=0.22, linewidth=0.5)
            if column == 0:
                axis.set_ylabel(f"{variable.upper()} national change (%)")
    legend_handles = [
        Patch(facecolor=METHOD_COLORS["paper_qm"], alpha=0.86, label=METHOD_LABELS["paper_qm"]),
        Patch(facecolor=METHOD_COLORS["optimized"], alpha=0.86, label=METHOD_LABELS["optimized"]),
    ]
    figure.suptitle(
        "National changes — China area mean first, then percentage change (definition 2)",
        y=0.995,
    )
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    figure.legend(
        handles=legend_handles,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        frameon=False,
        columnspacing=1.4,
        handlelength=1.6,
        handletextpad=0.6,
    )
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
        clip_mesh_to_china(mappable, geometries, axis)
        scale = "Seasonal" if definition == "seasonal_full" else "Monthly"
        axis.set_title(f"{panel_letter(index)} {scale} | {SCENARIO_LABELS[scenario]}")
    figure.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.22, wspace=0.10, hspace=0.22)
    cax = figure.add_axes([0.25, 0.09, 0.50, 0.02])
    figure.colorbar(
        mappable,
        cax=cax,
        orientation="horizontal",
        label="Spearman ρ difference (D − B)",
        extend="both",
    )
    figure.suptitle("Optimization difference — late-century complementarity (2080–2100)", y=0.99)
    save_figure(
        figure,
        output / "Opt04_complementarity_rho_D_minus_B_late_century",
        formats,
        dpi,
        outputs,
        "Late-century seasonal/monthly Spearman-rho difference, optimized D minus paper B.",
    )


def padded_ylim(values: np.ndarray, frac: float = 0.04) -> tuple[float, float]:
    """Expand a data range by ``frac`` on both ends so curves/bands don't touch axes."""
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return lo, hi
    span = hi - lo
    if span <= 0:
        pad = max(abs(hi) * 0.01, 1e-6)
        return lo - pad, hi + pad
    pad = frac * span
    return lo - pad, hi + pad


def joint_period_limits(maps: xr.Dataset, variable: str) -> tuple[float, float]:
    """1st-99th-percentile color limits over BOTH routes so B and D share one scale."""
    arrays = (
        nine_period_panels(maps, variable, "paper_qm")[1]
        + nine_period_panels(maps, variable, "optimized")[1]
    )
    return sequential_limits(arrays)


def change_arrays(
    maps: xr.Dataset, variable: str, method: str
) -> list[xr.DataArray]:
    arrays = []
    for scenario in SCENARIOS:
        for period in FUTURE_PERIODS:
            arrays.append(
                masked(
                    maps["gridwise_relative_change_percent"].sel(
                        {
                            "scenario": scenario,
                            "method": method,
                            "future_period": period,
                            "energy_variable": variable,
                        }
                    ),
                    maps,
                )
            )
    return arrays


def joint_change_limit(maps: xr.Dataset, variable: str) -> float:
    """Symmetric (0-centred) change-map limit over BOTH routes."""
    return symmetric_limit(
        change_arrays(maps, variable, "paper_qm")
        + change_arrays(maps, variable, "optimized")
    )


def joint_energy_ylim(annual: pd.DataFrame, variable: str) -> tuple[float, float]:
    """Shared y-range for the annual energy line plot (ensemble mean only)."""
    pieces = [
        annual_subset(annual, variable, method, scenario)["mean"].to_numpy(dtype=float)
        for method in METHODS
        for scenario in SCENARIOS
    ]
    return padded_ylim(np.concatenate(pieces))


def joint_climate_ylims(
    annual_model: pd.DataFrame, annual: pd.DataFrame
) -> dict[str, tuple[float, float]]:
    """Shared per-variable y-range for Fig02 (17 members + min/max band + mean)."""
    ylims = {}
    for variable in ("rsds", "tas_c", "sfcWind_10m"):
        pieces = []
        for method in METHODS:
            for scenario in SCENARIOS:
                data = annual_subset(annual, variable, method, scenario)
                pieces.append(data["minimum"].to_numpy(dtype=float))
                pieces.append(data["maximum"].to_numpy(dtype=float))
                members = annual_model_subset(annual_model, variable, method, scenario)
                pieces.append(
                    members["china_area_weighted_annual_mean"].to_numpy(dtype=float)
                )
        ylims[variable] = padded_ylim(np.concatenate(pieces))
    return ylims


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
            f"China cells     : {int(maps['china_intersects'].astype(bool).sum().values)}/"
            f"{maps.sizes['lat'] * maps.sizes['lon']} intersecting"
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
        optimized_root = output_root / "optimized_route"
        comparison_root = output_root / "method_comparison"
        outputs: list[dict[str, str]] = []

        # Joint B+D color/y limits so both routes share identical scales.
        period_limits = {
            variable: joint_period_limits(maps, variable)
            for variable in ("pvpot", "wpd")
        }
        change_limits = {
            variable: joint_change_limit(maps, variable)
            for variable in ("pvpot", "wpd")
        }
        energy_ylims = {
            variable: joint_energy_ylim(annual, variable)
            for variable in ("pvpot", "wpd")
        }
        climate_ylims = joint_climate_ylims(annual_model, annual)

        print("Shared B/D color/y limits (identical for both routes):")
        for variable in ("rsds", "tas_c", "sfcWind_10m"):
            print(f"  Fig02       {variable:12s} ylim = {climate_ylims[variable]}")
        for variable in ("pvpot", "wpd"):
            print(f"  Fig03/06    {variable:12s} ylim = {energy_ylims[variable]}")
            print(f"  Fig04/07    {variable:12s} colorbar = {period_limits[variable]}")
            print(f"  Fig05/08    {variable:12s} symmetric limit = {change_limits[variable]}")

        for method, root in (("paper_qm", paper_root), ("optimized", optimized_root)):
            print(f"Plotting route-{ROUTE_NAME[method]} figures 2-10 ...")
            plot_paper_climate_annual(
                annual_model,
                annual,
                root,
                arguments.formats,
                arguments.dpi,
                outputs,
                method,
                climate_ylims,
            )
            plot_paper_energy_annual(annual, "pvpot", "03", root, arguments.formats, arguments.dpi, outputs, method, energy_ylims["pvpot"])
            plot_period_maps(maps, geometries, extent, "pvpot", "04", root, arguments.formats, arguments.dpi, outputs, method, *period_limits["pvpot"])
            plot_change_maps(maps, geometries, extent, "pvpot", "05", root, arguments.formats, arguments.dpi, outputs, method, change_limits["pvpot"])
            plot_paper_energy_annual(annual, "wpd", "06", root, arguments.formats, arguments.dpi, outputs, method, energy_ylims["wpd"])
            plot_period_maps(maps, geometries, extent, "wpd", "07", root, arguments.formats, arguments.dpi, outputs, method, *period_limits["wpd"])
            plot_change_maps(maps, geometries, extent, "wpd", "08", root, arguments.formats, arguments.dpi, outputs, method, change_limits["wpd"])
            plot_complementarity(maps, geometries, extent, "seasonal_full", "09", root, arguments.formats, arguments.dpi, outputs, method)
            plot_complementarity(maps, geometries, extent, "monthly_full", "10", root, arguments.formats, arguments.dpi, outputs, method)

        print("Plotting B/D method-comparison figures ...")
        plot_annual_B_D(annual, comparison_root, arguments.formats, arguments.dpi, outputs)
        plot_method_difference_maps(maps, geometries, extent, "pvpot", comparison_root, arguments.formats, arguments.dpi, outputs)
        plot_method_difference_maps(maps, geometries, extent, "wpd", comparison_root, arguments.formats, arguments.dpi, outputs)
        plot_national_change(national, comparison_root, arguments.formats, arguments.dpi, outputs)
        plot_complementarity_difference(maps, geometries, extent, comparison_root, arguments.formats, arguments.dpi, outputs)

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
            "paper_figure_routes": {
                "paper_baseline": "route B (paper_qm)",
                "optimized_route": "route D (optimized)",
            },
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
                "percent change of China intersection-area-weighted area mean (口径2)"
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
                "shared within each multi-panel figure AND jointly across routes B and D; "
                "sequential maps use 1st-99th percentiles of the joint B+D data; "
                "difference maps use symmetric 99th absolute percentile of the joint B+D data; "
                "annual line plots share a joint B+D y-range with 4% padding; "
                "clipped values are shown with extend"
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
        print(f"Paper-baseline figures (route B): {paper_root}")
        print(f"Optimized-route figures (route D): {optimized_root}")
        print(f"Method-comparison figures        : {comparison_root}")
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
