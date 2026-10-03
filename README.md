# China PV & Wind Power under CMIP6 Scenarios

A multimodel (17-model) assessment of photovoltaic (PV) and wind-energy
potential and their complementarity over China, with Quantile-Mapping / Quantile
-Delta-Mapping (QM/QDM) bias correction against ERA5 reanalysis. This repository
reproduces the study's baseline route and an optimized variant.

> The repository ships the **code, configuration, and small results** — not the
> multi-GB climate data. See [Data availability](#data-availability).

---

## 1. Purpose

- **Reproduce** the baseline (paper) route of the PV/wind study.
- **Optimize** the bias-correction chain with a QDM variant and evaluate the
  resulting change in PV potential, wind power density, and complementarity.
- Produce 14 final figures (9 paper-baseline + 5 optimization).

## 2. Data

- **Models:** 17 CMIP6 global climate models (ACCESS-CM2, ACCESS-ESM1-5,
  AWI-CM-1-1-MR, BCC-CSM2-MR, CAS-ESM2-0, CESM2-WACCM, CMCC-CM2-SR5,
  CMCC-ESM2, CanESM5, FGOALS-f3-L, FIO-ESM-2-0, IPSL-CM6A-LR, KACE-1-0-G,
  MPI-ESM1-2-HR, MPI-ESM1-2-LR, MRI-ESM2-0, TaiESM1).
- **Scenarios:** SSP1-2.6, SSP2-4.5, SSP5-8.5 (`ssp126`, `ssp245`, `ssp585`).
- **Variables:** near-surface air temperature `tas`, downwelling shortwave
  radiation `rsds`, 10 m wind speed `sfcWind`.
- **Reanalysis:** ERA5 (used as the bias-correction reference and for the
  validation window).
- **Domain:** mainland China (clipped to a China polygon boundary).

## 3. Periods

- **Bias-correction calibration (fit) period:** 1959–2014.
- **Historical analysis baseline:** 1994–2014.
- **Future:** 2015–2100, summarised as mid-century (2040–2060) and
  late-century (2080–2100).

## 4. Routes B and D

- **Route B (paper baseline):** the study's QM method — reproduces the paper
  figures `Fig02`–`Fig10`.
- **Route D (optimized):** the QDM variant — compared against B in
  `Opt01`–`Opt04`.
- **Two aggregation definitions:** grid-wise maps (definition 1) and national
  aggregates (definition 2).

## 5. Pipeline (7 stages)

1. **Download** — CMIP6 (CEDA) and ERA5 (CDS); `scripts/01_download/`.
2. **Prepare** — clip to China, regrid ERA5 onto each model grid, build
   reference periods; `scripts/02_data_preparation/`.
3. **Bias correction** — QM/QDM, quantiles Q02–Q98, 30-year window;
   `scripts/03_bias_correction/`.
4. **Energy metrics** — PV potential and wind power density (incl. hub-height
   sensitivity and complementarity); `scripts/04_energy_metrics/`.
5. **Figure data** — multimodel ensemble + final-analysis products;
   `scripts/05_figure_data/`.
6. **Plotting** — the 14 final figures; `scripts/06_plotting/`.
7. **Validation** — ERA5 2015–2025 holdout / validation;
   `scripts/07_validation/`.

## 6. Running the pipeline

All production scripts read their paths from `config/project_paths.json` via
`scripts/common/project_paths.py` (no absolute paths, no hard-coded locations).
Run from the repository root:

```bash
conda activate pvwind

# Stage 3 — bias correction (51 model x scenario combinations)
python scripts/03_bias_correction/qm_qdm_batch_production.py --dry-run
python scripts/03_bias_correction/qm_qdm_batch_production.py

# Stage 4 — energy metrics
python scripts/04_energy_metrics/compute_energy_metrics_batch.py --dry-run
python scripts/04_energy_metrics/compute_energy_metrics_batch.py

# Stage 5 — multimodel ensemble, then final-analysis products
python scripts/05_figure_data/build_multimodel_ensemble.py --dry-run
python scripts/05_figure_data/build_multimodel_ensemble.py
python scripts/05_figure_data/build_final_analysis_products.py --dry-run
python scripts/05_figure_data/build_final_analysis_products.py

# Stage 6 — plot the 14 figures
python scripts/06_plotting/plot_final_figures.py --dry-run
python scripts/06_plotting/plot_final_figures.py
```

Stage 1–2 require access to the CMIP6 (CEDA) and ERA5 (CDS) archives and are
run before Stage 3. Stage 7 (`era5_2015_2025_validation.py`,
`run_qm_qdm_holdout_pilot_3x3.py`) is an independent validation pass.

Useful options on the main scripts: `--models` / `--scenarios` (run a subset),
`--expected-model-count` / `--expected-models` (assert the model count),
`--hub-height`, `--lower`/`--upper` (quantile bounds), `--force` (recompute).

## 7. Dry-run mode

Every production script supports `--dry-run`, which validates every input path,
grid, calendar, unit, dimension, and column **without writing any output**:

```bash
python scripts/03_bias_correction/qm_qdm_batch_production.py --dry-run   # 51/51
python scripts/04_energy_metrics/compute_energy_metrics_batch.py --dry-run  # 51/51
python scripts/05_figure_data/build_multimodel_ensemble.py --dry-run
python scripts/05_figure_data/build_final_analysis_products.py --dry-run
python scripts/06_plotting/plot_final_figures.py --dry-run
```

Run a dry-run first to confirm the environment and inputs before any production
run.

## 8. Data availability

The raw and intermediate data are **not** distributed with this repository
(≈73 GiB total). They must be re-obtained from the sources and placed in the
directories declared in `config/project_paths.json`:

```
data/raw/          # CMIP6 downloads, ERA5 raw (.grib), ERA5 archives
data/interim/      # China-clipped CMIP6, ERA5 reference, ERA5-on-GCM-grid
data/processed/    # bias_correction/, energy_metrics/, validation/
data/boundaries/   # china_qc_boundary.shp
```

The small, regenerable **results** (figures, summary tables, final-analysis
products, audit summaries) are included — see
`docs/public_release_file_manifest.md` for the exact allow/deny list.

## 9. Main results

14 final figures:

- **Paper baseline (`results/figures/paper_baseline/`):** `Fig02`–`Fig10` —
  climate factors, PV potential (annual/period-maps/gridwise change), wind power
  density (annual/period-maps/gridwise change), and seasonal/monthly
  complementarity.
- **Optimization (`results/figures/optimization/`):** `Opt01`–`Opt04` — annual
  energy B-vs-D, PV/WPD period relative difference (D−B), national change
  (definition 2), and late-century complementarity ρ (D−B).

Key summary tables live in `results/tables/` (cross-model `combined_*.csv`) and
the final-analysis products in `results/figure_data/final_analysis/`.

## 10. Current validation status

| Check | Result |
|---|---|
| Bias-corrected NetCDF | **153/153** present |
| Energy-metric NetCDF | **272/272** present |
| Final figures | **14** (9 paper + 5 optimization) |
| Bias-correction audit | **PASS** (153/153, 51/51 manifests) |
| Energy-metrics audit | **PASS** (272/272, 0 failures) |
| Downstream dry-runs | **PASS** (all stages) |

The full audit report is a local-only document (not published; see
`docs/public_release_file_manifest.md`).

## 11. Known scientific warnings

The bias-correction audit reports **45 warnings** (all WARNING, none FAILURE),
pre-existing and unchanged:

- **42 × `tail_trigger_percent` on `tas`** — future values outside the
  historical Q02–Q98 calibration range (expected in a warming scenario).
- **3 × `physical_range` on `rsds`**.

These flag conditions for review but do not invalidate the archive.

## 12. cfgrib / ecCodes (GRIB only)

`cfgrib` is needed only to read the raw ERA5 `.grib` files; it additionally
requires the **ecCodes** C library. All NetCDF-based production is unaffected.
See `docs/environment_notes.md`.

## 13. Citation & data sources

If you use this software or its results, please cite it via the
[`CITATION.cff`](CITATION.cff) file (author: **X-X-5**). Also cite the
underlying study (the manuscript PDF is kept locally under `references/` and is
**not** included in this repository) and the data providers:

- **CMIP6** model output — obtained from the CEDA Archive (data.ceda.ac.uk).
- **ERA5** reanalysis — Copernicus Climate Change Service / ECMWF.

> TODO: insert the full bibliographic citation for the manuscript
> (`references/papers/PV&Wind_power_New.pdf`).

## 14. Limitations

- This is a **research pipeline and result archive**, not a packaged library.
- The exact figures depend on the specific model ensemble, calibration window,
  and aggregation choices; results should be interpreted as a multimodel
  assessment, **not** a strictly pixel-perfect reproduction of any single run.
- Raw data redistribution is out of scope (see §8).

## 15. License

This project is released under the **MIT License** — see [`LICENSE`](LICENSE).

Copyright (c) 2026 **X-X-5**.
