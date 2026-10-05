# Environment Notes

## Recommendation: use conda

The project was developed and validated in a single conda environment named
`pvwind` (Python 3.12, `conda-forge`). Prefer

```bash
conda env create -f environment.yml
conda activate pvwind
```

`requirements.txt` is provided as a pip reference only. It is **not** the
recommended install path because `cartopy` (and `cfgrib`) pull in native
libraries that pip resolves poorly.

## Direct dependencies

Only the following third-party packages are imported by the scripts:

| Package | Purpose | Tested version |
|---|---|---|
| numpy | arrays | 2.5.2 |
| pandas | tabular data | 3.0.5 |
| xarray | NetCDF / labelled arrays | 2026.7.0 |
| scipy | statistics (energy metrics) | 1.18.0 |
| netCDF4 | NetCDF I/O backend | 1.7.4 |
| shapely | geometry / China boundary intersection | 2.1.2 |
| cartopy | map projections | 0.26.0 |
| pyproj | geodesic areas + coordinate transforms | 3.8.0 |
| cftime | calendar/time handling | 1.6.5 |
| matplotlib | plotting | 3.11.2 |
| requests | HTTP downloads (public URLs) | — |
| cdsapi | ERA5 download via the CDS API | — |
| cfgrib | ERA5/GRIB reading (needs eccodes) | — |
| eccodes | ecCodes C library (cfgrib backend) | — |

Versions are the values observed in the working environment; they are
informative, not hard pins.

## One environment for the whole pipeline

`environment.yml` installs everything the project needs in a single `pvwind`
environment: `cdsapi` for the ERA5/CDS download, `cfgrib` + `eccodes` for
reading the raw ERA5 `.grib` files, and `netCDF4`/`xarray` for the rest.

`cfgrib` depends on the **ecCodes** C library, which is *not* a Python package
and therefore cannot be installed by pip alone; `environment.yml` declares it
as the `eccodes` conda package. Both are installed and `import cfgrib` works in
the working environment, so download, GRIB→NetCDF conversion, and the whole
NetCDF pipeline run in this one environment.

### Windows note — native DLLs on `PATH`

On Windows, ecCodes (and netCDF4) ship native DLLs under the environment's
`Library/bin` directory. When invoking the interpreter directly (without
`conda activate`), add that directory to `PATH` first:

```bash
export PATH="$CONDA_PREFIX/Library/bin:$PATH"   # Git Bash
```

Failure to do so surfaces as `RuntimeError: Cannot find the ecCodes library`
(cfgrib) or an exit-code 127 DLL-load failure (netCDF4/native dependencies).

### CDS credentials

`cdsapi` reads its CDS API key from `~/.cdsapirc` (user-level). That file is
**not** part of the repository, and no credentials are stored in any tracked
file.

## Not required

These packages are **not** imported anywhere and are not needed:
`dask`, `geopandas`, `rasterio`, `xskillscore`, `scikit-learn`.
