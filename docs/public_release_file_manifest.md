# Public Release File Manifest (Allow / Deny)

Date: 2026-10-05

This manifest defines, file-by-file, what is recommended to **track** (commit)
and what is **ignored** when the project is published to GitHub. It is the
reference behind the conservative `.gitignore` at the repository root.

No data file was moved, renamed, deleted, or modified to produce this manifest;
it is a read-only inventory plus the ignore rules.

---

## Privacy & portability scan (summary)

Every proposed-publish directory was scanned for personal absolute paths,
usernames, and credentials.

| Pattern | Result |
|---|---|
| Credentials (`password`, `token`, `api_key`, `secret`, `Bearer`, `Authorization`, `AKIA…`, `ghp_…`, `sk-…`) | **None** in any script, notebook, or result file |
| Username (`<username>`) | Only in `results/audits/qm_qdm_batch_logs/*_dry_run.txt` (from the xarray warning that prints the site-packages path) and in two hand-written audit docs — all on the Deny list |
| Local absolute path (`<project-root>`) | Present in many **generated** manifests/audits and in the hand-written audit docs (see Deny list) |
| Active `scripts/**` absolute paths | **None** — all paths resolved via `config/project_paths.json` |
| Notebook (`notebooks/ERA5_Data().ipynb`) | **No outputs, no absolute paths** — clean |

**Consequence:** the generated files that embed the local path are placed on the
deny list (they are all regenerable), rather than publishing machine-specific
paths. The hand-written audit docs that mention the interpreter path are also
placed on the deny list (kept local-only, not modified in place).

---

## Allow list — recommended to track

### Root

| Path | Size | Regenerable | Reason |
|---|---|---|---|
| `README.md` | — | no | entry-point documentation |
| `.gitignore` | — | no | ignore rules |
| `environment.yml` | — | no | conda environment |
| `requirements.txt` | — | no | pip reference |

### Code / config / docs / notebooks

| Path | Size | Regenerable | Reason |
|---|---|---|---|
| `scripts/**` (49 `.py`) | ~0.8 MiB | no | the full pipeline — primary artifact |
| `config/project_paths.json` | 2 KB | no | central path config (relative paths only) |
| `notebooks/ERA5_Data().ipynb` | 14 KB | no | ERA5 GRIB preprocessing (no outputs) |
| `docs/public_release_file_manifest.md` | — | no | this manifest |
| `docs/environment_notes.md` | — | no | environment guidance |

### Figures (23 PNG)

| Path | Size | Regenerable | Reason |
|---|---|---|---|
| `results/figures/paper_baseline/Fig02`–`Fig10` (9 PNG) | ~9.2 MiB | yes (`plot_final_figures.py`) | paper-baseline figures (route B) |
| `results/figures/optimized_route/Fig02`–`Fig10` (9 PNG) | ~9 MiB | yes (`plot_final_figures.py`) | optimized-route figures (route D) |
| `results/figures/method_comparison/Opt01`–`Opt04` (5 PNG) | ~2.7 MiB | yes (`plot_final_figures.py`) | B-vs-D comparison figures |

### Cross-model summary tables (6 of 7)

| Path | Size | Regenerable | Reason |
|---|---|---|---|
| `results/tables/combined_complementarity_summary.csv` | 0.33 MB | yes | complementarity summary |
| `results/tables/combined_energy_period_summary.csv` | 0.80 MB | yes | period energy summary |
| `results/tables/combined_energy_qc_summary.csv` | 0.20 MB | yes | QC summary |
| `results/tables/combined_historical_baseline_comparison.csv` | 0.12 MB | yes | baseline comparison |
| `results/tables/combined_historical_baseline_complementarity_summary.csv` | 0.04 MB | yes | baseline complementarity |
| `results/tables/combined_hub_height_sensitivity.csv` | 0.22 MB | yes | hub-height sensitivity |

### Final-analysis products (needed to plot the 23 figures)

| Path | Size | Regenerable | Reason |
|---|---|---|---|
| `results/figure_data/final_analysis/final_annual_model_timeseries.csv` | 3.0 MB | yes (`build_final_analysis_products.py`) | per-model annual series |
| `results/figure_data/final_analysis/final_annual_ensemble_timeseries.csv` | 0.53 MB | yes | ensemble annual series |
| `results/figure_data/final_analysis/final_national_change_summary.csv` | 4.5 KB | yes | national change |
| `results/figure_data/final_analysis/final_D_minus_B_summary.csv` | 1.3 KB | yes | D−B summary |
| `results/figure_data/final_analysis/final_B_D_figure_maps_1deg.nc` | 0.70 MB | yes | B/D map NetCDF |

### Audit summaries (top-level, path-clean subset)

| Path | Size | Regenerable |
|---|---|---|
| `results/audits/audit_issues.csv` | 3 KB | yes |
| `results/audits/cmip6_download_integrity_report.csv` (+ `.txt`) | 207 KB | yes |
| `results/audits/cmip6_processed_qc_report.csv` (+ `.txt`) | 34 KB | yes |
| `results/audits/cmip6_spatial_grid_report.csv` (+ `.txt`) | 54 KB | yes |
| `results/audits/cmip6_time_coverage_report.csv` (+ `.txt`) | 17 KB | yes |
| `results/audits/energy_production_audit_checks.csv` | 5 B | yes |
| `results/audits/energy_production_audit_report.txt` | 498 B | yes |
| `results/audits/energy_transition_z_gt_3_warnings.csv` | 206 KB | yes |
| `results/audits/existing_cmip6_era5_audit.csv` | 51 KB | yes |
| `results/audits/future_change_signal_all.csv` | 1.18 MB | yes |
| `results/audits/future_period_change_signal_all.csv` | 3.79 MB | yes |
| `results/audits/future_qc_all.csv` | 1.23 MB | yes |
| `results/audits/cmip6_ceda_missing_recheck.csv` | 1.1 KB | yes — CEDA source URLs (`https://data.ceda.ac.uk/…`) only; no local paths |
| `results/audits/rerun_content_hashes_{before,after,comparison}.json` | ~0.4 MB | yes (content-hash walk over the four NetCDF roots; project-relative paths, no credentials) |

---

## Deny list — ignored (via `.gitignore`)

| Path | Size | Reason | Regenerable |
|---|---|---|---|
| `data/**` | 60.7 GiB | raw/interim/processed scientific data | yes (re-obtain + rerun) |
| `archive/**` | 12.6 GiB | historical/legacy bulk | n/a |
| `references/**` | 15.8 MB | paper PDF, copyright | n/a |
| `*.grib`, `*.grib2`, `*.idx` | — | ERA5 raw + sidecars | yes |
| `results/figures/**/*.zip` | 10.9 MB | figure bundle; 23 PNGs tracked individually | yes |
| `results/tables/combined_historical_future_transition.csv` | 17.2 MB | first-version large table (excluded by release plan) | yes |
| `results/figure_data/multimodel_ensemble/**` | 7.8 MB | intermediate maps/CSVs | yes |
| `results/figure_data/final_analysis/final_analysis_manifest.json` | 2.3 KB | embeds local path | yes |
| `results/manifests/**` (14 files) | ~0.4 MB | batch/energy/figure manifests embed the local project path (8 files) or a CEDA drive path `s:/` (4 files); auto-generated | yes |
| `results/audits/qm_qdm_batch_logs/**` (102 logs) | ~0.4 MB | logs embed local path **and** username | yes |
| `results/audits/energy_production_audit/**` | ~0.6 MB | duplicate of top-level energy audit | yes |
| `results/audits/qm_qdm_production_audit/**` | ~9 MB | duplicate of top-level QM/QDM audit | yes |
| `results/audits/qm_distribution_diagnostics_17models/**` | ~2.8 MB | large diagnostics (regenerable) | yes |
| `results/audits/audit_report.txt` | 767 B | embeds local path | yes |
| `results/audits/audit_summary.json` | 597 B | embeds local path | yes |
| `results/audits/cmip6_prepare_target_periods_report.csv` | 51 KB | embeds local path | yes |
| `results/audits/csv_inventory.csv` | 82 KB | embeds local path | yes |
| `results/audits/netcdf_inventory.csv` | 55 KB | embeds local path | yes |
| `results/audits/manifest_inventory.csv` | 10.6 KB | embeds local path | yes |
| `results/audits/energy_production_audit_summary.json` | 139 KB | embeds local path | yes |
| `_reorganize.py`, `_refactor_paths.py`, `reorganization_manifest.json` | 1 MB | one-off migration artifacts, embed local path | n/a |
| `docs/post_reorganization_audit.md`, `docs/github_release_readiness_audit.md`, `docs/path_refactoring_report.md` | 25 KB | hand-written audit history — embed the local interpreter path / username; kept local-only (summarised in README §10–§11) | n/a |
| `__pycache__/`, `.ipynb_checkpoints/`, `*.log`, `*.tmp`, IDE/OS, `.env`/`.venv`/`.conda` | — | local tooling / environment | n/a |

---

## Regeneration map (result → producing script)

| Result | Produced by |
|---|---|
| 23 PNG figures | `scripts/06_plotting/plot_final_figures.py` |
| 6 combined tables | `scripts/04_energy_metrics/compute_energy_metrics_batch.py` (spatial-mean tables are intersection-area-weighted via `scripts/common/area_weights.py`; a from-archive re-summary uses `scripts/04_energy_metrics/resummarize_energy_tables.py`) |
| final-analysis products | `scripts/05_figure_data/build_final_analysis_products.py` |
| multimodel-ensemble maps/CSVs | `scripts/05_figure_data/build_multimodel_ensemble.py` |
| audit summaries | `scripts/03_bias_correction/audit_qm_qdm_production.py`, `scripts/04_energy_metrics/audit_energy_metrics_production.py` (transition-warning re-summary via `scripts/04_energy_metrics/resummarize_transition_warnings.py`), and the `01_download`/`02_data_preparation` audit scripts |
| manifests / status files | the respective batch scripts (auto-written) |

## Estimated repository footprint

- **Tracked files: ~100** (49 scripts + 1 config + 1 notebook + 2 docs + 4 root
  files + 23 PNGs + 6 tables + 5 final-analysis products + 17 audit summaries).
- **Tracked size: ~26 MiB**, with no single tracked file above 10 MiB.
- Excluded: 73 GiB of data/archive and all machine-path-tainted regenerable
  results.
