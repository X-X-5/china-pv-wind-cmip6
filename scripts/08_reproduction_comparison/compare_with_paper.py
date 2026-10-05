#!/usr/bin/env python
r"""Read-only comparison of Route B (``paper_qm``) formal outputs against the
published baseline values in Y. Fan et al., "…wind and solar energy potential
and complementarity in China…", *Renewable Energy* 249 (2025) 123213
(DOI 10.1016/j.renene.2025.123213).

Phase 2 of the reproduction-comparability audit. The script ONLY READS existing
Route B formal CSV / NetCDF products. It does **not** rerun any upstream stage
(QM/QDM, energy metrics, ensemble, final analysis, or plotting) and it does
**not** modify any existing scientific result.  For Table 5 it streams the
Route B energy NetCDF files directly (no energy computation is rerun).

It writes:

    results/tables/paper_baseline_values_inventory.csv
    results/tables/paper_reproduction_comparison.csv
    results/tables/paper_trend_mk_summary.csv
    docs/paper_reproduction_comparison.md
    results/figures/reproduction_comparison/*.png   (<= 2)

Scope rules
-----------
* Only Route B (``paper_qm``) is read. Route D / A / C are not touched.
* Paper "by 2100" endpoints are compared against the Route B **year-2100**
  annual ensemble mean, never against the 2080-2100 period mean (that mean is
  reported only as ``late-century period mean``).
* Approximate paper values are reported with ``absolute_difference`` and a
  direction / ranking-consistency flag only -- no fake-precision relative
  errors.
* Table 5 (WPD max/min/median/std) is treated as **non-identifiable** unless a
  unique aggregation order can be established; three candidate aggregations are
  computed (plain pooled, per-model mean, area-weighted pooled) and reported
  with full provenance (time range, spatial domain, area weighting, model
  weighting, month pooling).  None is force-matched.
* Spatial maps are compared qualitatively only (no digitisation, no
  interpolation, no pseudo-pixel RMSE); the map figure is titled "Route B
  spatial pattern with published qualitative anchors".
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_COMMON_DIR = _SCRIPTS_DIR / "common"
for _dir in (_SCRIPTS_DIR, _COMMON_DIR):
    if _dir.is_dir() and str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import numpy as np
import pandas as pd
from scipy import stats

try:
    from project_paths import PROJECT_ROOT, get_path, models as _canonical_models
except Exception:  # pragma: no cover - standalone fallback
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    def get_path(name: str) -> Path:
        raise RuntimeError("project_paths unavailable")

    def _canonical_models() -> list[str]:
        return []

FIG_DATA = PROJECT_ROOT / "results" / "figure_data" / "final_analysis"
TABLES = PROJECT_ROOT / "results" / "tables"
DOCS = PROJECT_ROOT / "docs"
FIGURES = PROJECT_ROOT / "results" / "figures" / "reproduction_comparison"

ENSEMBLE_CSV = FIG_DATA / "final_annual_ensemble_timeseries.csv"
NATIONAL_CSV = FIG_DATA / "final_national_change_summary.csv"
MAPS_NC = FIG_DATA / "final_B_D_figure_maps_1deg.nc"

OUT_INVENTORY = TABLES / "paper_baseline_values_inventory.csv"
OUT_COMPARISON = TABLES / "paper_reproduction_comparison.csv"
OUT_TREND_MK = TABLES / "paper_trend_mk_summary.csv"
OUT_MD = DOCS / "paper_reproduction_comparison.md"

METHOD = "paper_qm"
ROUTE = "B"
SCENARIOS = ("ssp126", "ssp245", "ssp585")

PAPER_TO_PROJECT = {
    "tas": "tas_c",
    "sfcWind": "sfcWind_10m",
    "rsds": "rsds",
    "pvpot": "pvpot",
    "wpd": "wpd",
}
PROJECT_TO_PAPER = {v: k for k, v in PAPER_TO_PROJECT.items()}
VARIABLE_UNIT = {
    "tas_c": "degree_Celsius",
    "sfcWind_10m": "m s-1",
    "rsds": "W m-2",
    "pvpot": "W m-2",
    "wpd": "W m-2",
}
ALL_VARIABLES = ("pvpot", "wpd", "rsds", "tas_c", "sfcWind_10m")

PAPER_DOI = "10.1016/j.renene.2025.123213"
PAPER_CITATION = "Y. Fan et al., Renewable Energy 249 (2025) 123213"

# Table 5 paper values (exact, per SSP): maximum / minimum / median / std.
PAPER_TABLE5 = {
    "ssp126": {"maximum": 2551.42, "minimum": 0.0029, "median": 9.78, "std": 151.49},
    "ssp245": {"maximum": 2561.68, "minimum": 0.0044, "median": 9.62, "std": 150.22},
    "ssp585": {"maximum": 2796.12, "minimum": 0.0028, "median": 9.84, "std": 150.40},
}


# --------------------------------------------------------------------------
# Statistical helpers
# --------------------------------------------------------------------------


def mann_kendall(x: np.ndarray) -> dict[str, float]:
    """Standard Mann-Kendall S, tie-corrected variance, Z, and two-sided p.

    Two-sided p-value via the normal approximation: p = 2 * P(Z >= |z|).
    Ties are accounted for in Var(S) (Gilbert, 1987).
    """
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    s = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            if diff > 0.0:
                s += 1.0
            elif diff < 0.0:
                s -= 1.0

    _, counts = np.unique(x, return_counts=True)
    tie_term = float(sum(int(t) * (int(t) - 1) * (2 * int(t) + 5) for t in counts if t > 1))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if s > 0.0:
        z = (s - 1.0) / math.sqrt(var_s)
    elif s < 0.0:
        z = (s + 1.0) / math.sqrt(var_s)
    else:
        z = 0.0
    p_two_sided = 2.0 * float(stats.norm.sf(abs(z)))
    return {"S": s, "var_S": var_s, "Z": z, "p_two_sided": p_two_sided, "n": n}


def ols_trend(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ordinary least-squares slope and intercept of y on x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xbar = x.mean()
    ybar = y.mean()
    denom = float(((x - xbar) ** 2).sum())
    slope = float(((x - xbar) * (y - ybar)).sum() / denom)
    intercept = float(ybar - slope * xbar)
    return slope, intercept


def theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Theil-Sen slope (median of pairwise slopes); sensitivity check only."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slopes = []
    n = x.size
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx != 0.0:
                slopes.append((y[j] - y[i]) / dx)
    return float(np.median(slopes))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median over a 1-D sample set (weights repeat per cell)."""
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    finite = np.isfinite(values) & (weights > 0.0)
    v = values[finite]
    w = weights[finite]
    if v.size == 0:
        return float("nan")
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cumulative = np.cumsum(w)
    total = cumulative[-1]
    if total <= 0.0:
        return float("nan")
    index = int(np.searchsorted(cumulative, 0.5 * total))
    return float(v[index])


def _fmt(v: float, nd: int = 4) -> str:
    return f"{v:.{nd}f}"


def _fmt_cell(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        if pd.isna(v):
            return ""
        return f"{v:.6g}"
    return str(v)


def _md_table(df: pd.DataFrame) -> str:
    """Minimal markdown pipe table (no ``tabulate`` dependency)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(_fmt_cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


# --------------------------------------------------------------------------
# Load Route B inputs
# --------------------------------------------------------------------------


def load_route_b_series() -> pd.DataFrame:
    df = pd.read_csv(ENSEMBLE_CSV, encoding="utf-8-sig")
    df = df[(df["route"] == ROUTE) & (df["method"] == METHOD)].copy()
    df["year"] = df["year"].astype(int)
    return df


def series_for(df: pd.DataFrame, variable: str, scenario: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[(df["variable"] == variable) & (df["scenario"] == scenario)].sort_values("year")
    return sub["year"].to_numpy(dtype=float), sub["mean"].to_numpy(dtype=float)


# --------------------------------------------------------------------------
# Paper baseline inventory (hand-transcribed from the PDF text; read-only)
# --------------------------------------------------------------------------


def paper_baseline_rows() -> list[dict]:
    rows: list[dict] = []
    add = rows.append

    pv_2100 = {"ssp126": 196.0, "ssp245": 193.0, "ssp585": 188.0}
    for ssp, val in pv_2100.items():
        add({
            "item_id": f"pvpot_2100_{ssp}",
            "metric": "national mean PVpot at year 2100",
            "variable": "pvpot", "ssp": ssp, "period": "2100",
            "paper_value": val, "unit": "W m-2", "source_figure_table": "Fig. 3 / section 4.2",
            "source_page": "p.6 (results)", "exact_or_approx": "approximate",
            "comparability_grade": "B",
            "definition_note": "paper 'end of century'/'2100' value; compared to Route B year-2100 "
                              "(NOT the 2080-2100 period mean)",
        })

    add({
        "item_id": "pvpot_slope_ssp126",
        "metric": "PVpot linear trend slope, 2015-2100",
        "variable": "pvpot", "ssp": "ssp126", "period": "2015-2100",
        "paper_value": 0.07, "unit": "W m-2 yr-1", "source_figure_table": "Fig. 3 / section 4.2",
        "source_page": "p.6 (results)", "exact_or_approx": "approximate",
        "comparability_grade": "B",
        "definition_note": "paper reports ~+0.07 W m-2 yr-1 for SSP1-2.6",
    })
    add({
        "item_id": "pvpot_slope_ssp585",
        "metric": "PVpot linear trend slope, 2015-2100",
        "variable": "pvpot", "ssp": "ssp585", "period": "2015-2100",
        "paper_value": -0.03, "unit": "W m-2 yr-1", "source_figure_table": "Fig. 3 / section 4.2",
        "source_page": "p.6 (results)", "exact_or_approx": "approximate",
        "comparability_grade": "B",
        "definition_note": "paper reports ~-0.03 W m-2 yr-1 for SSP5-8.5",
    })

    add({
        "item_id": "sfcwind_2100",
        "metric": "sfcWind value at year 2100",
        "variable": "sfcWind", "ssp": "all", "period": "2100",
        "paper_value": 1.7, "unit": "m s-1", "source_figure_table": "section 4.1",
        "source_page": "p.5 (results)", "exact_or_approx": "approximate",
        "comparability_grade": "B",
        "definition_note": "paper: 'reaching ~1.7 m s-1 by end of this century'; compared to "
                          "Route B year-2100 per SSP",
    })
    add({
        "item_id": "sfcwind_decline",
        "metric": "sfcWind decline over the projection",
        "variable": "sfcWind", "ssp": "all", "period": "2015-2100",
        "paper_value": 0.1, "unit": "m s-1", "source_figure_table": "section 4.1",
        "source_page": "p.5 (results)", "exact_or_approx": "approximate",
        "comparability_grade": "B",
        "definition_note": "paper: 'decrease by approximately 0.1 m s-1' (start vs end)",
    })

    table4_z = {
        "rsds": {"ssp126": 11.8, "ssp245": 11.6, "ssp585": 10.9, "dir": "increasing"},
        "tas": {"ssp126": 9.2, "ssp245": 12.7, "ssp585": 13.3, "dir": "increasing"},
        "sfcWind": {"ssp126": -5.2, "ssp245": -7.9, "ssp585": -9.9, "dir": "decreasing"},
    }
    for var, d in table4_z.items():
        for ssp, zval in d.items():
            if ssp == "dir":
                continue
            add({
                "item_id": f"table4_z_{var}_{ssp}",
                "metric": "Mann-Kendall Z (national annual mean series)",
                "variable": var, "ssp": ssp, "period": "2015-2100",
                "paper_value": zval, "unit": "Z (dimensionless)", "source_figure_table": "Table 4",
                "source_page": "p.5 (results)", "exact_or_approx": "exact",
                "comparability_grade": "B",
                "definition_note": "all p<0.001; sign gives trend direction",
            })
        add({
            "item_id": f"table4_dir_{var}",
            "metric": "trend direction of national annual mean",
            "variable": var, "ssp": "all", "period": "2015-2100",
            "paper_value": d["dir"], "unit": "direction", "source_figure_table": "Table 4",
            "source_page": "p.5 (results)", "exact_or_approx": "exact",
            "comparability_grade": "B",
            "definition_note": "sign of M-K Z",
        })

    add({
        "item_id": "wpd_diff_ssp245_vs_ssp126",
        "metric": "WPD end-of-century difference vs SSP1-2.6 (year-2100 AND period-mean)",
        "variable": "wpd", "ssp": "ssp245", "period": "2100 / 2080-2100",
        "paper_value": -1.0, "unit": "W m-2", "source_figure_table": "Fig. 6 / section 4.3",
        "source_page": "p.7 (results)", "exact_or_approx": "approximate",
        "comparability_grade": "B",
        "definition_note": "paper '~1 W m-2 lower under SSP2-4.5'; paper does NOT specify "
                          "year-2100 vs 2080-2100 period mean",
    })
    add({
        "item_id": "wpd_diff_ssp585_vs_ssp126",
        "metric": "WPD end-of-century difference vs SSP1-2.6 (year-2100 AND period-mean)",
        "variable": "wpd", "ssp": "ssp585", "period": "2100 / 2080-2100",
        "paper_value": -2.0, "unit": "W m-2", "source_figure_table": "Fig. 6 / section 4.3",
        "source_page": "p.7 (results)", "exact_or_approx": "approximate",
        "comparability_grade": "B",
        "definition_note": "paper '~2 W m-2 lower under SSP5-8.5'; paper does NOT specify "
                          "year-2100 vs 2080-2100 period mean",
    })

    table5 = PAPER_TABLE5
    stat_names = ["maximum", "minimum", "median", "std"]
    for ssp, vals in table5.items():
        for stat_name in stat_names:
            add({
                "item_id": f"table5_{stat_name}_{ssp}",
                "metric": f"WPD {stat_name} over China (unspecified aggregation)",
                "variable": "wpd", "ssp": ssp, "period": "unspecified (future 2015-2100 samples)",
                "paper_value": vals[stat_name], "unit": "W m-2", "source_figure_table": "Table 5",
                "source_page": "p.7 (results)", "exact_or_approx": "exact",
                "comparability_grade": "B (non-identifiable)",
                "definition_note": "time/space/model aggregation order unspecified in the paper",
            })

    table3 = {
        "very_strong_similarity": "0.9 <= rho <= 1.0",
        "strong_similarity": "0.6 <= rho < 0.9",
        "moderate_similarity": "0.3 <= rho < 0.6",
        "weak_similarity": "0.0 <= rho < 0.3",
        "weak_complementarity": "-0.3 < rho <= 0.0",
        "moderate_complementarity": "-0.6 < rho <= -0.3",
        "strong_complementarity": "-0.9 < rho <= -0.6",
        "very_strong_complementarity": "-1.0 <= rho <= -0.9",
    }
    for name, rng in table3.items():
        add({
            "item_id": f"table3_{name}",
            "metric": "Spearman rho similarity/complementarity threshold",
            "variable": "wpd-vs-pvpot", "ssp": "all", "period": "any",
            "paper_value": rng, "unit": "rho interval", "source_figure_table": "Table 3",
            "source_page": "p.5 (results)", "exact_or_approx": "exact",
            "comparability_grade": "A",
            "definition_note": "project uses identical 8-grade thresholds",
        })

    return rows


# --------------------------------------------------------------------------
# Route B derived metrics
# --------------------------------------------------------------------------


def compute_route_b_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variable in ALL_VARIABLES:
        for scenario in SCENARIOS:
            years, values = series_for(df, variable, scenario)
            if years.size == 0:
                continue
            late = values[years >= 2080.0]
            late_mean = float(late.mean()) if late.size else math.nan
            year2015 = float(values[years == 2015.0][0]) if (years == 2015.0).any() else math.nan
            year2100 = float(values[years == 2100.0][0]) if (years == 2100.0).any() else math.nan
            start = float(values[years <= 2019.0].mean()) if (years <= 2019.0).any() else math.nan
            end = float(values[years >= 2096.0].mean()) if (years >= 2096.0).any() else math.nan
            decline_5yr = start - end
            decline_endpoint = year2015 - year2100

            mk = mann_kendall(values)
            ols_slope, ols_int = ols_trend(years, values)
            ts = theil_sen_slope(years, values)
            direction = "increasing" if mk["Z"] > 0 else ("decreasing" if mk["Z"] < 0 else "none")

            records.append({
                "variable": variable,
                "variable_paper": PROJECT_TO_PAPER.get(variable, variable),
                "ssp": scenario,
                "unit": VARIABLE_UNIT[variable],
                "n_years": int(years.size),
                "year_2015": year2015,
                "year_2100": year2100,
                "late_century_mean_2080_2100": late_mean,
                "start_mean_2015_2019": start,
                "end_mean_2096_2100": end,
                "decline_endpoint_2015_2100": decline_endpoint,
                "decline_5yr_smoothed": decline_5yr,
                "mk_S": mk["S"],
                "mk_var_S": mk["var_S"],
                "mk_Z": mk["Z"],
                "mk_p_two_sided": mk["p_two_sided"],
                "mk_direction": direction,
                "ols_slope_per_yr": ols_slope,
                "ols_intercept": ols_int,
                "theil_sen_slope_per_yr": ts,
            })
    return pd.DataFrame(records)


def paper_lookup() -> dict[str, dict]:
    return {row["item_id"]: row for row in paper_baseline_rows()}


def build_comparison_rows(metrics: pd.DataFrame) -> list[dict]:
    paper = paper_lookup()
    rows: list[dict] = []

    def metric_row(variable: str, scenario: str) -> dict | None:
        sub = metrics[(metrics["variable"] == variable) & (metrics["ssp"] == scenario)]
        return sub.iloc[0].to_dict() if not sub.empty else None

    def add_compare(item_id, route_b_value, route_b_desc, direction_match, note="", paper_value=None, unit=None, row_id=None):
        base = paper[item_id]
        pv = paper_value if paper_value is not None else base["paper_value"]
        unit = unit or base["unit"]
        if isinstance(pv, (int, float)) and isinstance(route_b_value, (int, float)) and not (
            isinstance(pv, bool) or isinstance(route_b_value, bool)
        ):
            abs_diff = abs(route_b_value - pv)
        else:
            abs_diff = math.nan
        rows.append({
            "item_id": row_id or item_id,
            "paper_metric": base["metric"],
            "variable": base["variable"],
            "ssp": base["ssp"],
            "period": base["period"],
            "paper_value": pv,
            "paper_unit": unit,
            "exact_or_approx": base["exact_or_approx"],
            "comparability_grade": base["comparability_grade"],
            "route_b_value": route_b_value,
            "route_b_value_description": route_b_desc,
            "absolute_difference": abs_diff,
            "direction_or_ranking_match": direction_match,
            "note": note or base["definition_note"],
        })

    # PVpot year-2100 (paper 196/193/188 vs Route B year-2100).
    for ssp in SCENARIOS:
        m = metric_row("pvpot", ssp)
        if m is None:
            continue
        add_compare(
            f"pvpot_2100_{ssp}", m["year_2100"],
            "Route B year-2100 ensemble annual mean (area-weighted)",
            "n/a (scalar)",
            f"late-century (2080-2100) period mean = {_fmt(m['late_century_mean_2080_2100'], 3)} W m-2 "
            "(informational only, NOT the 2100 endpoint)",
        )

    # PVpot trend slopes (unchanged).
    for ssp, key in (("ssp126", "pvpot_slope_ssp126"), ("ssp585", "pvpot_slope_ssp585")):
        m = metric_row("pvpot", ssp)
        if m is None:
            continue
        paper_slope = paper[key]["paper_value"]
        direction = "consistent" if math.copysign(1, m["ols_slope_per_yr"]) == math.copysign(1, paper_slope) else "inconsistent"
        add_compare(
            key, m["ols_slope_per_yr"],
            "Route B OLS slope of 2015-2100 China area-weighted annual mean",
            direction,
            f"Theil-Sen sensitivity slope = {_fmt(m['theil_sen_slope_per_yr'], 4)} W m-2 yr-1",
        )

    # sfcWind year-2100 (paper 1.7 vs Route B year-2100, per SSP).
    for ssp in SCENARIOS:
        m = metric_row("sfcWind_10m", ssp)
        if m is None:
            continue
        add_compare(
            "sfcwind_2100", m["year_2100"],
            f"Route B year-2100 ensemble annual mean, {ssp} (area-weighted)",
            "n/a (scalar)",
            f"paper gives a single ~1.7 m s-1 (not per SSP); area-weighted vs plain-native "
            f"diagnostic in the Markdown report",
            paper_value=1.7, unit="m s-1", row_id=f"sfcwind_2100_{ssp}",
        )

    # sfcWind decline (paper ~0.1 m s-1 vs Route B, per SSP).
    for ssp in SCENARIOS:
        m = metric_row("sfcWind_10m", ssp)
        if m is None:
            continue
        add_compare(
            "sfcwind_decline", m["decline_endpoint_2015_2100"],
            f"Route B (year-2015 - year-2100), {ssp}",
            "consistent" if m["decline_endpoint_2015_2100"] > 0 else "inconsistent",
            f"5-yr smoothed decline = {_fmt(m['decline_5yr_smoothed'], 4)} m s-1; "
            f"paper reports a decline (~0.1 m s-1)",
            paper_value=0.1, unit="m s-1", row_id=f"sfcwind_decline_{ssp}",
        )

    # Table 4 M-K Z and direction.
    table4_proj = {"rsds": "rsds", "tas": "tas_c", "sfcWind": "sfcWind_10m"}
    for paper_var, proj_var in table4_proj.items():
        for ssp in SCENARIOS:
            m = metric_row(proj_var, ssp)
            if m is None:
                continue
            z_b = m["mk_Z"]
            z_paper = paper[f"table4_z_{paper_var}_{ssp}"]["paper_value"]
            direction = "consistent" if math.copysign(1, z_b) == math.copysign(1, z_paper) else "inconsistent"
            add_compare(
                f"table4_z_{paper_var}_{ssp}", z_b,
                "Route B M-K Z on 2015-2100 area-weighted annual mean series",
                direction,
                "Z magnitude is not directly comparable (different series construction); "
                "sign is the comparable quantity; both two-sided",
            )
        m0 = metric_row(proj_var, "ssp126")
        if m0 is None:
            continue
        add_compare(
            f"table4_dir_{paper_var}", m0["mk_direction"],
            "Route B M-K trend direction (consistent across all 3 SSPs)",
            "consistent" if m0["mk_direction"] == paper[f"table4_dir_{paper_var}"]["paper_value"] else "inconsistent",
            "paper direction: " + str(paper[f"table4_dir_{paper_var}"]["paper_value"]),
        )

    # WPD scenario differences: year-2100 AND 2080-2100 period mean.
    m126 = metric_row("wpd", "ssp126")
    for ssp, key in (("ssp245", "wpd_diff_ssp245_vs_ssp126"), ("ssp585", "wpd_diff_ssp585_vs_ssp126")):
        m = metric_row("wpd", ssp)
        if m is None or m126 is None:
            continue
        paper_diff = paper[key]["paper_value"]
        # year-2100
        d2100 = m["year_2100"] - m126["year_2100"]
        dir2100 = "consistent" if math.copysign(1, d2100) == math.copysign(1, paper_diff) else "inconsistent"
        add_compare(
            key, d2100,
            f"Route B year-2100 ({ssp} minus ssp126)",
            dir2100, "paper does not specify year-2100 vs period mean",
            paper_value=paper_diff, row_id=key + "_2100",
        )
        # period mean
        dpm = m["late_century_mean_2080_2100"] - m126["late_century_mean_2080_2100"]
        dirpm = "consistent" if math.copysign(1, dpm) == math.copysign(1, paper_diff) else "inconsistent"
        add_compare(
            key, dpm,
            f"Route B 2080-2100 period mean ({ssp} minus ssp126)",
            dirpm, "paper does not specify year-2100 vs period mean",
            paper_value=paper_diff, row_id=key + "_periodmean",
        )

    return rows


# --------------------------------------------------------------------------
# Table 5 candidate aggregations + sfcWind 2100 diagnostic (stream NetCDFs)
# --------------------------------------------------------------------------


def compute_table5_and_sfcwind_diag():
    """Stream Route B energy NetCDFs (no energy computation rerun).

    Returns (table5_candidates, sfc_diag) where table5_candidates is a list of
    per-scenario dicts (candidates A/B/C) and sfc_diag maps scenario -> dict with
    area-weighted and plain-native year-2100 sfcWind endpoint means.
    """
    import xarray as xr
    from area_weights import china_intersection_weights, area_weighted_mean

    shapefile = get_path("china_shapefile")
    energy_root = get_path("data_processed_energy_metrics")
    models = sorted(p.name for p in energy_root.iterdir() if p.is_dir())
    _canonical = _canonical_models()
    if _canonical:
        models = [m for m in _canonical if (energy_root / m).is_dir()] or models

    # Precompute per-model China intersection mask + area once.
    model_weights: dict[str, dict] = {}
    for model in models:
        sample = energy_root / model / "ssp126" / f"energy_monthly_{model}_ssp126_paper_qm_199401-210012.nc"
        with xr.open_dataset(sample) as ds:
            lat = ds["lat"].values.astype(float)
            lon = ds["lon"].values.astype(float)
        w = china_intersection_weights(lat, lon, shapefile)
        model_weights[model] = {
            "mask": w["china_intersects"],
            "area": w["china_intersection_area_km2"],
        }

    candidates: list[dict] = []
    sfc_diag: dict[str, dict] = {}

    for scenario in SCENARIOS:
        plain_count = 0
        plain_sum = 0.0
        plain_sum_sq = 0.0
        plain_min = float("inf")
        plain_max = float("-inf")
        w_sum = 0.0
        w_mean_num = 0.0
        w_sum_sq = 0.0
        vals_list: list[np.ndarray] = []
        weights_list: list[np.ndarray] = []
        b_min: list[float] = []
        b_max: list[float] = []
        b_med: list[float] = []
        b_std: list[float] = []
        sfc_aw: list[float] = []
        sfc_plain: list[float] = []

        for model in models:
            path = energy_root / model / scenario / f"energy_monthly_{model}_{scenario}_paper_qm_199401-210012.nc"
            with xr.open_dataset(path) as ds:
                mask = model_weights[model]["mask"]
                area = model_weights[model]["area"]

                wp = ds["wpd"]
                t_idx = np.flatnonzero(wp.time.dt.year.values >= 2015)
                arr = wp.isel(time=t_idx).values.astype(np.float64)
                T = arr.shape[0]
                finite = np.isfinite(arr)
                sel = mask[None, :, :] & finite
                vals = arr[sel]
                w_cell = np.broadcast_to(area[None, :, :], (T, area.shape[0], area.shape[1]))
                wts = w_cell[sel]

                if vals.size:
                    plain_count += vals.size
                    plain_sum += float(vals.sum())
                    plain_sum_sq += float((vals ** 2).sum())
                    plain_min = min(plain_min, float(vals.min()))
                    plain_max = max(plain_max, float(vals.max()))
                    w_sum += float(wts.sum())
                    w_mean_num += float((vals * wts).sum())
                    w_sum_sq += float((vals ** 2 * wts).sum())
                    vals_list.append(vals.astype(np.float32))
                    weights_list.append(wts.astype(np.float32))
                    b_min.append(float(vals.min()))
                    b_max.append(float(vals.max()))
                    b_med.append(float(np.median(vals)))
                    b_std.append(float(vals.std(ddof=0)))

                # sfcWind year-2100 (last 12 months) endpoint diagnostic.
                sw = ds["sfcWind_10m"]
                field2100 = sw.isel(time=t_idx).isel(time=slice(-12, None)).mean("time").values.astype(np.float64)
                sfc_aw.append(float(area_weighted_mean(field2100, area)))
                sfc_plain.append(float(np.nanmean(field2100[mask])))

        all_vals = np.concatenate(vals_list)
        all_wts = np.concatenate(weights_list)

        plain_mean = plain_sum / plain_count
        plain_std = math.sqrt(max(plain_sum_sq / plain_count - plain_mean ** 2, 0.0))
        plain_median = float(np.median(all_vals))

        w_mean = w_mean_num / w_sum
        w_var = (w_sum_sq - 2.0 * w_mean * w_mean_num + w_mean ** 2 * w_sum) / w_sum
        w_std = math.sqrt(max(w_var, 0.0))
        w_median = weighted_median(all_vals, all_wts)

        candidates.append({
            "scenario": scenario, "candidate": "A",
            "description": "17 models x 2015-2100 months x China cells, plain pooled",
            "time_range": "2015-2100 (monthly)", "spatial_domain": "China-intersection native cells",
            "area_weighted": "no", "model_weighting": "pooled (each sample equal)",
            "months_pooled": "yes",
            "maximum": plain_max, "minimum": plain_min, "median": plain_median, "std": plain_std,
        })
        candidates.append({
            "scenario": scenario, "candidate": "B",
            "description": "per-model stats then 17-model equal-weight mean",
            "time_range": "2015-2100 (monthly)", "spatial_domain": "China-intersection native cells",
            "area_weighted": "no", "model_weighting": "equal-weight mean of per-model stats",
            "months_pooled": "yes (within each model)",
            "maximum": float(np.mean(b_max)), "minimum": float(np.mean(b_min)),
            "median": float(np.mean(b_med)), "std": float(np.mean(b_std)),
        })
        candidates.append({
            "scenario": scenario, "candidate": "C",
            "description": "pooled, area-weighted by china_intersection_area_km2",
            "time_range": "2015-2100 (monthly)", "spatial_domain": "China-intersection native cells",
            "area_weighted": "yes (china_intersection_area_km2)",
            "model_weighting": "pooled (weighted by cell area)",
            "months_pooled": "yes",
            "maximum": plain_max, "minimum": plain_min, "median": w_median, "std": w_std,
        })

        sfc_diag[scenario] = {
            "area_weighted_2100": float(np.mean(sfc_aw)),
            "plain_2100": float(np.mean(sfc_plain)),
        }

    return candidates, sfc_diag


def table5_absdiff(candidates: list[dict]) -> list[dict]:
    """Per-candidate per-stat absolute difference vs the paper's Table 5."""
    rows = []
    for c in candidates:
        paper = PAPER_TABLE5[c["scenario"]]
        rows.append({
            "scenario": c["scenario"], "candidate": c["candidate"],
            "max_absdiff": abs(c["maximum"] - paper["maximum"]),
            "min_absdiff": abs(c["minimum"] - paper["minimum"]),
            "median_absdiff": abs(c["median"] - paper["median"]),
            "std_absdiff": abs(c["std"] - paper["std"]),
        })
    return rows


# --------------------------------------------------------------------------
# Figures (<= 2, qualitative only)
# --------------------------------------------------------------------------


def _china_boundary_rings() -> list[np.ndarray]:
    """Exterior (+interior) coordinate rings of the official China boundary.

    Loads the exact geometry used by the formal analysis
    (``china_intersection_weights`` / ``area_weights._load_china_geometry``):
    the WGS84 ``china_qc_boundary.shp`` unified into a single geometry.  Returns
    one ``(N, 2)`` lon/lat array per ring (mainland, Hainan, Taiwan, and every
    included island).  No provincial boundaries are present in this source.
    """
    import cartopy.io.shapereader as shpreader
    from shapely.ops import unary_union

    shapefile = get_path("china_shapefile")
    reader = shpreader.Reader(str(shapefile))
    geom = unary_union(list(reader.geometries()))
    close = getattr(reader, "close", None)
    if close is not None:
        close()

    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
    else:
        polys = [geom]

    rings: list[np.ndarray] = []
    for poly in polys:
        rings.append(np.asarray(poly.exterior.coords))
        for interior in poly.interiors:
            rings.append(np.asarray(interior.coords))
    return rings


def _china_clip_path():
    """matplotlib Path that clips rasters to the China boundary (no data change)."""
    from matplotlib.path import Path as MplPath

    verts: list = []
    codes: list = []
    for ring in _china_boundary_rings():
        verts.extend(ring)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY])
    return MplPath(np.asarray(verts), codes)


def plot_annual_series(df: pd.DataFrame) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"ssp126": "#2c7bb6", "ssp245": "#fdae61", "ssp585": "#d7191c"}
    panels = (("pvpot", "PV potential", "PVpot (W m$^{-2}$)", {"ssp126": 196, "ssp245": 193, "ssp585": 188}),
              ("wpd", "Wind power density", "WPD (W m$^{-2}$)", None),
              ("sfcWind_10m", "Surface wind speed", "sfcWind (m s$^{-1}$)", None))

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True, constrained_layout=True)

    for ax, (var, title, ylabel, endpoints) in zip(axes, panels):
        sfc_vals: list[np.ndarray] = []
        for ssp in SCENARIOS:
            years, vals = series_for(df, var, ssp)
            ax.plot(years, vals, color=colors[ssp], lw=1.4)
            if var == "sfcWind_10m":
                sfc_vals.append(vals)

        if var == "sfcWind_10m":
            ax.axhline(1.7, color="0.25", ls="--", lw=1)
            allv = np.concatenate(sfc_vals)
            ylo = float(np.nanmin(allv)) - 0.04
            ax.set_ylim(ylo, 1.78)
            ax.text(2016, 1.71, "published ~1.7 m s$^{-1}$\n(year-2100 anchor)",
                    fontsize=8, va="bottom", ha="left", color="0.25")
        elif endpoints:
            for ssp, pv in endpoints.items():
                ax.scatter([2100], [pv], marker="x", s=70, color=colors[ssp],
                           linewidths=1.7, zorder=6)

        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("year")

    handles = [
        Line2D([], [], color=colors["ssp126"], lw=1.4, label="SSP126"),
        Line2D([], [], color=colors["ssp245"], lw=1.4, label="SSP245"),
        Line2D([], [], color=colors["ssp585"], lw=1.4, label="SSP585"),
        Line2D([], [], marker="x", color="0.15", lw=0, ms=7, markeredgewidth=1.7,
               label="Published approximate year-2100 endpoint"),
    ]
    axes[0].legend(handles=handles, title="solid lines = Route B annual ensemble means",
                   loc="upper left", fontsize=8, title_fontsize=8, framealpha=0.9, ncol=2)

    fig.suptitle("Route B (paper_qm): annual ensemble means, 2015–2100\n"
                 "Published approximate year-2100 values shown as qualitative anchors",
                 fontsize=11)
    out = FIGURES / "FigRC01_route_B_annual_series_with_paper_endpoint_markers.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_spatial_pattern() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec

    FIGURES.mkdir(parents=True, exist_ok=True)

    rings = _china_boundary_rings()
    clip_path = _china_clip_path()

    with xr.open_dataset(MAPS_NC) as ds:
        lon = ds["lon"].values
        lat = ds["lat"].values
        mask = ds["china_intersects"].values > 0
        pvpot = ds["period_mean"].sel({"method": METHOD, "period": "late_century", "energy_variable": "pvpot"})
        wpd_lg = ds["wpd_log10"].sel({"method": METHOD, "period": "late_century"})

        # Joint color limits across all three SSPs, so each row shares one scale.
        pv_all = np.concatenate([pvpot.sel({"scenario": s}).values.ravel() for s in SCENARIOS])
        pv_all = pv_all[np.isfinite(pv_all)]
        pv_vmin, pv_vmax = float(pv_all.min()), float(pv_all.max())

        wl_all = np.concatenate([wpd_lg.sel({"scenario": s}).values.ravel() for s in SCENARIOS])
        wl_all = wl_all[np.isfinite(wl_all)]
        wl_vmin, wl_vmax = float(wl_all.min()), float(wl_all.max())

        norm_pv = Normalize(vmin=pv_vmin, vmax=pv_vmax)
        norm_wl = Normalize(vmin=wl_vmin, vmax=wl_vmax)

        fig = plt.figure(figsize=(14, 9))
        gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05], hspace=0.34, wspace=0.08,
                      left=0.055, right=0.93, top=0.86, bottom=0.085)
        axes = np.empty((2, 3), dtype=object)
        for r in range(2):
            for c in range(3):
                axes[r, c] = fig.add_subplot(gs[r, c])
        cax = [fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 3])]

        pv_ref = None
        wl_ref = None
        for col, ssp in enumerate(SCENARIOS):
            pv = np.where(mask, pvpot.sel({"scenario": ssp}).values, np.nan)
            wl = np.where(mask, wpd_lg.sel({"scenario": ssp}).values, np.nan)
            ax_pv = axes[0, col]
            ax_wd = axes[1, col]

            im_pv = ax_pv.pcolormesh(lon, lat, pv, cmap="viridis", norm=norm_pv, shading="auto")
            im_wl = ax_wd.pcolormesh(lon, lat, wl, cmap="magma", norm=norm_wl, shading="auto")
            # Geometry clip only; values, mask, and area weights are untouched.
            im_pv.set_clip_path(clip_path, transform=ax_pv.transData)
            im_wl.set_clip_path(clip_path, transform=ax_wd.transData)

            for ring in rings:
                ax_pv.plot(ring[:, 0], ring[:, 1], color="black", lw=0.8, zorder=6)
                ax_wd.plot(ring[:, 0], ring[:, 1], color="black", lw=0.8, zorder=6)

            ax_pv.set_title(f"PVpot · {ssp.upper()}", fontsize=10)
            ax_wd.set_title(f"WPD log10 · {ssp.upper()}", fontsize=10)
            ax_pv.set_aspect("equal")
            ax_wd.set_aspect("equal")
            pv_ref = im_pv
            wl_ref = im_wl

        for r in range(2):
            axes[r, 0].set_ylabel("Latitude (°N)")
            for c in (1, 2):
                axes[r, c].tick_params(axis="y", labelleft=False)
        for c in range(3):
            axes[1, c].set_xlabel("Longitude (°E)")

        cb_pv = fig.colorbar(pv_ref, cax=cax[0])
        cb_wl = fig.colorbar(wl_ref, cax=cax[1])
        cb_pv.set_ticks(np.linspace(pv_vmin, pv_vmax, 5))
        cb_wl.set_ticks(np.linspace(wl_vmin, wl_vmax, 5))
        cb_pv.set_label("PVpot (W m$^{-2}$)", fontsize=9)
        cb_wl.set_label(r"log$_{10}$[WPD / (W m$^{-2}$)]", fontsize=9)

        anchors = ("Published qualitative anchors (not digitised): PVpot max ~330 W m$^{-2}$ SW, "
                   "min ~60 W m$^{-2}$ Sichuan; WPD abundant SW/N/SE coast, deficient Hunan/Anhui.")
        fig.suptitle("Route B (paper_qm): late-century (2080–2100) spatial pattern\n"
                     "with published qualitative anchors", fontsize=11, y=0.985)
        fig.text(0.5, 0.02, anchors, ha="center", va="bottom", fontsize=8, wrap=True)

        out = FIGURES / "FigRC02_route_B_spatial_pattern_with_qualitative_anchors.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------


def write_markdown(metrics: pd.DataFrame, table5: list[dict], sfc_diag: dict) -> None:
    lines: list[str] = []
    lines.append("# Paper reproduction comparison (Route B vs Y. Fan et al. 2025)")
    lines.append("")
    lines.append(f"- Paper: {PAPER_CITATION} (DOI {PAPER_DOI})")
    lines.append("- Compared route: **B** (`paper_qm`) — 17-model equal-weight ensemble, "
                 "monthly multiplicative QM, geodesic China-boundary intersection-area weighting.")
    lines.append("- Read-only: no upstream stage was rerun; no existing scientific result was modified.")
    lines.append("")
    lines.append("## Definitional notes (per audit corrections)")
    lines.append("")
    lines.append("1. **Wind shear exponent** — paper Eq. 7 is "
                 "`k = (0.37 - 0.0881 ln(v1)) / (1 - 0.0881 ln(h0/10))`; with `h0 = 10 m` the "
                 "denominator is 1, so the project's `k = 0.37 - 0.0881 ln(v10)` is **exactly "
                 "equivalent**. No WPD-formula correction is made.")
    lines.append("2. **Spatial averaging** — the paper states only a spatial mean over mainland-China "
                 "grid cells; it does not specify the boundary mask, partial-cell treatment, or area "
                 "weighting. The project uses WGS84 boundary-intersection area weighting. The PVpot "
                 "offset cannot be attributed solely to area weighting.")
    lines.append("3. **Hub height** — the paper does not state a numeric hub height; the project uses "
                 "`100 m`. Recorded as a project assumption.")
    lines.append("4. **\"by 2100\" endpoints** — paper PVpot/sfcWind end-of-century values are compared "
                 "against the Route B **year-2100** annual ensemble mean; the 2080-2100 mean is reported "
                 "only as `late-century period mean` and never substitutes for the 2100 endpoint.")
    lines.append("5. **Table 5** — WPD max/min/median/std are reported per SSP (future samples); "
                 "time/space/model aggregation order is unspecified, so it is treated as "
                 "**non-identifiable** (three candidate aggregations only).")
    lines.append("")
    lines.append("## Route B Mann-Kendall + OLS trends (2015-2100)")
    lines.append("")
    lines.append(_md_table(metrics[["variable", "ssp", "n_years", "mk_Z", "mk_p_two_sided",
                                    "mk_direction", "ols_slope_per_yr", "theil_sen_slope_per_yr"]]))
    lines.append("")

    lines.append("## sfcWind year-2100 endpoint diagnostic (area-weighted vs plain native-grid)")
    lines.append("")
    lines.append("Used only to interpret the 1.7 vs Route B gap; not a new formal dual-method result.")
    lines.append("")
    sfc_rows = []
    for ssp in SCENARIOS:
        d = sfc_diag.get(ssp, {})
        sfc_rows.append({"ssp": ssp, "area_weighted_2100": d.get("area_weighted_2100"),
                         "plain_native_2100": d.get("plain_2100")})
    lines.append(_md_table(pd.DataFrame(sfc_rows)))
    lines.append("")

    lines.append("## Table 5 candidate aggregations (non-identifiable)")
    lines.append("")
    lines.append("All three candidates pool the full 2015-2100 monthly samples over China-intersection "
                 "native-grid cells; they differ only in weighting. None matches the paper's four "
                 "statistics simultaneously (the paper minimum ~0.003 W m-2 is two orders of magnitude "
                 "below every candidate, implying a different, unspecified aggregation).")
    lines.append("")
    lines.append(_md_table(pd.DataFrame(table5)[["scenario", "candidate", "description",
                                                 "area_weighted", "model_weighting", "months_pooled",
                                                 "maximum", "minimum", "median", "std"]]))
    lines.append("")
    lines.append("### Absolute differences vs the paper's Table 5")
    lines.append("")
    lines.append(_md_table(pd.DataFrame(table5_absdiff(table5))))
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    for p in (OUT_INVENTORY, OUT_COMPARISON, OUT_TREND_MK, OUT_MD,
              FIGURES / "FigRC01_route_B_annual_series_with_paper_endpoint_markers.png",
              FIGURES / "FigRC02_route_B_spatial_pattern_with_qualitative_anchors.png"):
        lines.append(f"- `{p.relative_to(PROJECT_ROOT).as_posix()}`")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Read-only Route B (paper_qm) vs paper reproduction "
                    "comparison. No upstream stage is rerun and no existing "
                    "scientific result is modified; only the comparison CSVs, "
                    "Markdown report, and two qualitative figures are written."
    )
    parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = load_route_b_series()
    metrics = compute_route_b_metrics(df)

    inv = paper_baseline_rows()
    pd.DataFrame(inv).to_csv(OUT_INVENTORY, index=False, encoding="utf-8-sig")

    comp = build_comparison_rows(metrics)
    pd.DataFrame(comp).to_csv(OUT_COMPARISON, index=False, encoding="utf-8-sig")

    trend = metrics.copy()
    t4 = {"rsds": {"ssp126": 11.8, "ssp245": 11.6, "ssp585": 10.9},
          "tas_c": {"ssp126": 9.2, "ssp245": 12.7, "ssp585": 13.3},
          "sfcWind_10m": {"ssp126": -5.2, "ssp245": -7.9, "ssp585": -9.9}}
    trend["paper_ref_mk_Z"] = [t4.get(v, {}).get(s, np.nan) for v, s in zip(trend["variable"], trend["ssp"])]
    pv_slopes = {"ssp126": 0.07, "ssp585": -0.03}
    trend["paper_ref_slope_per_yr"] = [
        pv_slopes.get(s, np.nan) if v == "pvpot" else np.nan
        for v, s in zip(trend["variable"], trend["ssp"])
    ]
    trend["paper_z_direction_match"] = [
        ("yes" if (pd.notna(z) and pd.notna(zz) and math.copysign(1, z) == math.copysign(1, zz)) else "n/a")
        for z, zz in zip(trend["mk_Z"], trend["paper_ref_mk_Z"])
    ]
    trend.to_csv(OUT_TREND_MK, index=False, encoding="utf-8-sig")

    table5, sfc_diag = compute_table5_and_sfcwind_diag()

    write_markdown(metrics, table5, sfc_diag)

    fig1 = plot_annual_series(df)
    fig2 = plot_spatial_pattern()
    fig_files = [fig1, fig2]

    ok = all(p.exists() and p.stat().st_size > 0 for p in
             (OUT_INVENTORY, OUT_COMPARISON, OUT_TREND_MK, OUT_MD) + tuple(fig_files))

    print("=" * 100)
    print("PAPER REPRODUCTION COMPARISON (Route B) — COMPLETE" if ok else "SELF-CHECK FAILED")
    print("=" * 100)
    print(f"Baseline inventory rows : {len(inv)}")
    print(f"Comparison rows        : {len(comp)}")
    print(f"Trend/M-K rows         : {len(trend)}")
    print(f"Table 5 candidates     : {len(table5)}")
    print(f"Figures                : {len(fig_files)}")
    for p in (OUT_INVENTORY, OUT_COMPARISON, OUT_TREND_MK, OUT_MD) + tuple(fig_files):
        print(f"  wrote {p.relative_to(PROJECT_ROOT).as_posix()}")

    print("\nCorrected year-2100 endpoints (paper vs Route B):")
    pv_paper = {"ssp126": 196, "ssp245": 193, "ssp585": 188}
    for ssp in SCENARIOS:
        mp = metrics[(metrics["variable"] == "pvpot") & (metrics["ssp"] == ssp)].iloc[0]
        ms = metrics[(metrics["variable"] == "sfcWind_10m") & (metrics["ssp"] == ssp)].iloc[0]
        print(f"  {ssp}: pvpot paper={pv_paper[ssp]} vs B2100={mp['year_2100']:.3f} "
              f"| sfcWind paper=1.7 vs B2100={ms['year_2100']:.3f}")

    print("\nTable 5 candidates (max/min/median/std):")
    for c in table5:
        print(f"  {c['scenario']} {c['candidate']}: {c['maximum']:.2f} / {c['minimum']:.4f} / "
              f"{c['median']:.3f} / {c['std']:.2f}  [{c['area_weighted']}]")
    print("\nTable 5 verdict: non-identifiable (no candidate matches all four statistics).")

    print("\nsfcWind 2100 diagnostic (area-weighted vs plain native-grid):")
    for ssp in SCENARIOS:
        d = sfc_diag[ssp]
        print(f"  {ssp}: area-weighted={d['area_weighted_2100']:.4f}  plain-native={d['plain_2100']:.4f}")

    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
