# Paper reproduction comparison (Route B vs Y. Fan et al. 2025)

- Paper: Y. Fan et al., Renewable Energy 249 (2025) 123213 (DOI 10.1016/j.renene.2025.123213)
- Compared route: **B** (`paper_qm`) — 17-model equal-weight ensemble, monthly multiplicative QM, geodesic China-boundary intersection-area weighting.
- Read-only: no upstream stage was rerun; no existing scientific result was modified.

## Definitional notes (per audit corrections)

1. **Wind shear exponent** — paper Eq. 7 is `k = (0.37 - 0.0881 ln(v1)) / (1 - 0.0881 ln(h0/10))`; with `h0 = 10 m` the denominator is 1, so the project's `k = 0.37 - 0.0881 ln(v10)` is **exactly equivalent**. No WPD-formula correction is made.
2. **Spatial averaging** — the paper states only a spatial mean over mainland-China grid cells; it does not specify the boundary mask, partial-cell treatment, or area weighting. The project uses WGS84 boundary-intersection area weighting. The PVpot offset cannot be attributed solely to area weighting.
3. **Hub height** — the paper does not state a numeric hub height; the project uses `100 m`. Recorded as a project assumption.
4. **"by 2100" endpoints** — paper PVpot/sfcWind end-of-century values are compared against the Route B **year-2100** annual ensemble mean; the 2080-2100 mean is reported only as `late-century period mean` and never substitutes for the 2100 endpoint.
5. **Table 5** — WPD max/min/median/std are reported per SSP (future samples); time/space/model aggregation order is unspecified, so it is treated as **non-identifiable** (three candidate aggregations only).

## Route B Mann-Kendall + OLS trends (2015-2100)

| variable | ssp | n_years | mk_Z | mk_p_two_sided | mk_direction | ols_slope_per_yr | theil_sen_slope_per_yr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pvpot | ssp126 | 86 | 10.5928 | 3.21945e-26 | increasing | 0.0510399 | 0.0478904 |
| pvpot | ssp245 | 86 | 9.10828 | 8.36948e-20 | increasing | 0.0292607 | 0.0295404 |
| pvpot | ssp585 | 86 | -7.60143 | 2.92884e-14 | decreasing | -0.0231649 | -0.0245842 |
| wpd | ssp126 | 86 | -5.11735 | 3.0986e-07 | decreasing | -0.0195417 | -0.0202415 |
| wpd | ssp245 | 86 | -7.37764 | 1.61125e-13 | decreasing | -0.0208915 | -0.0197843 |
| wpd | ssp585 | 86 | -9.07099 | 1.17945e-19 | decreasing | -0.0315958 | -0.0310037 |
| rsds | ssp126 | 86 | 10.6226 | 2.33947e-26 | increasing | 0.0664488 | 0.0619947 |
| rsds | ssp245 | 86 | 11.391 | 4.63902e-30 | increasing | 0.062339 | 0.062111 |
| rsds | ssp585 | 86 | 10.727 | 7.59932e-27 | increasing | 0.0476793 | 0.0475518 |
| tas_c | ssp126 | 86 | 9.29478 | 1.47515e-20 | increasing | 0.0142987 | 0.013535 |
| tas_c | ssp245 | 86 | 12.7635 | 2.62038e-37 | increasing | 0.0326559 | 0.0333076 |
| tas_c | ssp585 | 86 | 13.3528 | 1.14003e-40 | increasing | 0.0720216 | 0.0725359 |
| sfcWind_10m | ssp126 | 86 | -5.923 | 3.16129e-09 | decreasing | -0.000631677 | -0.000638018 |
| sfcWind_10m | ssp245 | 86 | -8.51151 | 1.71684e-17 | decreasing | -0.000583783 | -0.000552279 |
| sfcWind_10m | ssp585 | 86 | -10.451 | 1.44944e-25 | decreasing | -0.00098429 | -0.000976079 |

## sfcWind year-2100 endpoint diagnostic (area-weighted vs plain native-grid)

Used only to interpret the 1.7 vs Route B gap; not a new formal dual-method result.

| ssp | area_weighted_2100 | plain_native_2100 |
| --- | --- | --- |
| ssp126 | 1.27232 | 1.40148 |
| ssp245 | 1.26923 | 1.39672 |
| ssp585 | 1.24133 | 1.36912 |

## Table 5 candidate aggregations (non-identifiable)

All three candidates pool the full 2015-2100 monthly samples over China-intersection native-grid cells; they differ only in weighting. None matches the paper's four statistics simultaneously (the paper minimum ~0.003 W m-2 is two orders of magnitude below every candidate, implying a different, unspecified aggregation).

| scenario | candidate | description | area_weighted | model_weighting | months_pooled | maximum | minimum | median | std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ssp126 | A | 17 models x 2015-2100 months x China cells, plain pooled | no | pooled (each sample equal) | yes | 7616.54 | 0.000254926 | 9.51462 | 145.998 |
| ssp126 | B | per-model stats then 17-model equal-weight mean | no | equal-weight mean of per-model stats | yes (within each model) | 5361.02 | 0.0010734 | 9.50935 | 152.505 |
| ssp126 | C | pooled, area-weighted by china_intersection_area_km2 | yes (china_intersection_area_km2) | pooled (weighted by cell area) | yes | 7616.54 | 0.000254926 | 8.60844 | 72.868 |
| ssp245 | A | 17 models x 2015-2100 months x China cells, plain pooled | no | pooled (each sample equal) | yes | 6426.45 | 0.000251781 | 9.78288 | 144.999 |
| ssp245 | B | per-model stats then 17-model equal-weight mean | no | equal-weight mean of per-model stats | yes (within each model) | 5081.57 | 0.0011007 | 9.69049 | 151.624 |
| ssp245 | C | pooled, area-weighted by china_intersection_area_km2 | yes (china_intersection_area_km2) | pooled (weighted by cell area) | yes | 6426.45 | 0.000251781 | 8.8278 | 72.5166 |
| ssp585 | A | 17 models x 2015-2100 months x China cells, plain pooled | no | pooled (each sample equal) | yes | 6629.43 | 0.000193931 | 9.67364 | 144.81 |
| ssp585 | B | per-model stats then 17-model equal-weight mean | no | equal-weight mean of per-model stats | yes (within each model) | 5216.44 | 0.00104671 | 9.55051 | 151.582 |
| ssp585 | C | pooled, area-weighted by china_intersection_area_km2 | yes (china_intersection_area_km2) | pooled (weighted by cell area) | yes | 6629.43 | 0.000193931 | 8.69554 | 72.423 |

### Absolute differences vs the paper's Table 5

| scenario | candidate | max_absdiff | min_absdiff | median_absdiff | std_absdiff |
| --- | --- | --- | --- | --- | --- |
| ssp126 | A | 5065.12 | 0.00264507 | 0.265382 | 5.49228 |
| ssp126 | B | 2809.6 | 0.0018266 | 0.270654 | 1.01469 |
| ssp126 | C | 5065.12 | 0.00264507 | 1.17156 | 78.622 |
| ssp245 | A | 3864.77 | 0.00414822 | 0.162879 | 5.22137 |
| ssp245 | B | 2519.89 | 0.0032993 | 0.0704883 | 1.40391 |
| ssp245 | C | 3864.77 | 0.00414822 | 0.792195 | 77.7034 |
| ssp585 | A | 3833.31 | 0.00260607 | 0.166363 | 5.58951 |
| ssp585 | B | 2420.32 | 0.00175329 | 0.289494 | 1.18175 |
| ssp585 | C | 3833.31 | 0.00260607 | 1.14446 | 77.977 |

## Outputs

- `results/tables/paper_baseline_values_inventory.csv`
- `results/tables/paper_reproduction_comparison.csv`
- `results/tables/paper_trend_mk_summary.csv`
- `docs/paper_reproduction_comparison.md`
- `results/figures/reproduction_comparison/FigRC01_route_B_annual_series_with_paper_endpoint_markers.png`
- `results/figures/reproduction_comparison/FigRC02_route_B_spatial_pattern_with_qualitative_anchors.png`
