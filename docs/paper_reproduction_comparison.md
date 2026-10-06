# Paper reproduction comparison (Route B vs Y. Fan et al. 2025)

- Paper: Y. Fan et al., Renewable Energy 249 (2025) 123213 (DOI 10.1016/j.renene.2025.123213)
- Compared route: **B** (`paper_qm`) — 17-model equal-weight ensemble, monthly multiplicative QM, geodesic China-boundary intersection-area weighting.
- **Status: Route B is a paper-like baseline (a re-implementation of the paper's described method), not an exact re-run of the paper's own code.** Absolute-value differences are characterised below, not scored as errors.
- Read-only: no upstream stage was rerun; no existing scientific result was modified.
- **Scope — default formal run:** native-grid diagnostics (Table-5 candidate aggregations and the sfcWind year-2100 area-weighting diagnostic) are **not** included; enable them with `--include-native-grid-diagnostics` (requires `data/processed/energy_metrics/`).

## Evidence classification

Every paper value is classified by evidence source; only `reported_numeric_value` rows are eligible for quantitative (definition-consistent) comparison.

| class | definition | treatment |
| --- | --- | --- |
| `reported_numeric_value` | explicit number in the paper body text, Table 4, or Table 5 | quantitative comparison allowed where metric definitions match |
| `approximate_anchor` | rough textual/figure anchor (e.g. "~1.7 m s-1", "~0.1 m s-1") | retained as an anchor only; direction-only; excluded from strict error / RMSE / overall score |
| (pixel-digitised series) | automated Fig. 2 pixel extraction | abandoned; not present in any formal output |

## Definitional notes (per audit corrections)

1. **Wind shear exponent** — paper Eq. 7 is `k = (0.37 - 0.0881 ln(v1)) / (1 - 0.0881 ln(h0/10))`; with `h0 = 10 m` the denominator is 1, so the project's `k = 0.37 - 0.0881 ln(v10)` is **exactly equivalent**. No WPD-formula correction is made.
2. **Spatial averaging** — the paper states only a spatial mean over mainland-China grid cells; it does not specify the boundary mask, partial-cell treatment, or area weighting. The project uses WGS84 boundary-intersection area weighting. The PVpot offset cannot be attributed solely to area weighting; see the M1/M5 spatial-domain sensitivity below.
3. **Hub height** — the paper does not state a numeric hub height; the project uses `100 m`. Recorded as a project assumption.
4. **"by 2100" endpoints** — paper PVpot/sfcWind end-of-century values are compared against the Route B **year-2100** annual ensemble mean; the 2080-2100 mean is reported only as `late-century period mean` and never substitutes for the 2100 endpoint. The sfcWind `~1.7 m s-1` value is an **approximate textual anchor**, not a strict quantitative endpoint.
5. **Table 5** — WPD max/min/median/std are **reported numerical values** from the paper's Table 5, but the time/space/model aggregation order is unspecified, so they are treated as **non-identifiable** (three candidate aggregations are shown for transparency, not force-matched).

## Spatial-domain sensitivity (M1 vs M5)

PVpot year-2100 national mean under two aggregation domains, against the paper's approximate endpoints (W m-2):

| aggregation | ssp126 | ssp245 | ssp585 |
| --- | --- | --- | --- |
| M1 — China intersection-area weighted | 204.55 | 201.83 | 197.94 |
| M5 — rectangular simple domain | 197.38 | 194.58 | 190.79 |
| Paper (approximate) | ~196 | ~193 | ~188 |

> Rectangular-domain aggregation explains approximately 72–84% of the absolute PVpot difference. This evidence is consistent with, but does not prove, a difference in spatial aggregation domain.

## Route B Mann-Kendall + OLS trends (2015-2100)

Route B results with the paper's Table 4 Mann–Kendall Z; the **sign** is the comparable quantity (Z magnitude is not directly comparable because the series construction differs).

| variable | ssp | n_years | mk_Z | paper_table4_Z | mk_p_two_sided | mk_direction | ols_slope_per_yr | theil_sen_slope_per_yr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pvpot | ssp126 | 86 | 10.5928 |  | 3.21945e-26 | increasing | 0.0510399 | 0.0478904 |
| pvpot | ssp245 | 86 | 9.10828 |  | 8.36948e-20 | increasing | 0.0292607 | 0.0295404 |
| pvpot | ssp585 | 86 | -7.60143 |  | 2.92884e-14 | decreasing | -0.0231649 | -0.0245842 |
| wpd | ssp126 | 86 | -5.11735 |  | 3.0986e-07 | decreasing | -0.0195417 | -0.0202415 |
| wpd | ssp245 | 86 | -7.37764 |  | 1.61125e-13 | decreasing | -0.0208915 | -0.0197843 |
| wpd | ssp585 | 86 | -9.07099 |  | 1.17945e-19 | decreasing | -0.0315958 | -0.0310037 |
| rsds | ssp126 | 86 | 10.6226 | 11.8 | 2.33947e-26 | increasing | 0.0664488 | 0.0619947 |
| rsds | ssp245 | 86 | 11.391 | 11.6 | 4.63902e-30 | increasing | 0.062339 | 0.062111 |
| rsds | ssp585 | 86 | 10.727 | 10.9 | 7.59932e-27 | increasing | 0.0476793 | 0.0475518 |
| tas_c | ssp126 | 86 | 9.29478 | 9.2 | 1.47515e-20 | increasing | 0.0142987 | 0.013535 |
| tas_c | ssp245 | 86 | 12.7635 | 12.7 | 2.62038e-37 | increasing | 0.0326559 | 0.0333076 |
| tas_c | ssp585 | 86 | 13.3528 | 13.3 | 1.14003e-40 | increasing | 0.0720216 | 0.0725359 |
| sfcWind_10m | ssp126 | 86 | -5.923 | -5.2 | 3.16129e-09 | decreasing | -0.000631677 | -0.000638018 |
| sfcWind_10m | ssp245 | 86 | -8.51151 | -7.9 | 1.71684e-17 | decreasing | -0.000583783 | -0.000552279 |
| sfcWind_10m | ssp585 | 86 | -10.451 | -9.9 | 1.44944e-25 | decreasing | -0.00098429 | -0.000976079 |

## Approximate textual/figure anchors (direction-only)

The following paper values are approximate textual/figure anchors and are **not** used for strict error or an overall reproduction score.

- **sfcWind `~1.7 m s-1` at year 2100** (paper section 4.1) — approximate anchor. Route B year-2100 area-weighted mean: 1.272 / 1.269 / 1.241 m s-1.
- **sfcWind decline `~0.1 m s-1` over 2015-2100** (paper section 4.1) — approximate anchor. Route B (year-2015 − year-2100): 0.045 / 0.035 / 0.059 m s-1.
- **WPD `~1 W m-2` lower (SSP2-4.5) and `~2 W m-2` lower (SSP5-8.5) vs SSP1-2.6** (paper section 4.3) — approximate anchors.

## Table 5 (reported numerical value, non-identifiable aggregation)

The paper's Table 5 reports WPD maximum/minimum/median/std per SSP; the aggregation order (time/space/model) is unspecified, so the values are reported but **not** force-matched.

| scenario | maximum | minimum | median | std |
| --- | --- | --- | --- | --- |
| ssp126 | 2551.42 | 0.0029 | 9.78 | 151.49 |
| ssp245 | 2561.68 | 0.0044 | 9.62 | 150.22 |
| ssp585 | 2796.12 | 0.0028 | 9.84 | 150.4 |

## Outputs

- `results/tables/paper_baseline_values_inventory.csv`
- `results/tables/paper_reproduction_comparison.csv`
- `results/tables/paper_trend_mk_summary.csv`
- `docs/paper_reproduction_comparison.md`
- `results/figures/reproduction_comparison/FigRC01_route_B_annual_series_with_paper_endpoint_markers.png`
- `results/figures/reproduction_comparison/FigRC02_route_B_spatial_pattern_with_qualitative_anchors.png`
