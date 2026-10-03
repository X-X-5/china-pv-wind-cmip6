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
| shapely | geometry / China mask | 2.1.2 |
| cartopy | map projections | 0.26.0 |
| pyproj | coordinate transforms (cartopy dep) | 3.8.0 |
| cftime | calendar/time handling | 1.6.5 |
| matplotlib | plotting | 3.11.2 |
| requests | HTTP downloads (public URLs) | — |
| cfgrib | **optional** — ERA5/GRIB reading | — |

Versions are the values observed in the working environment; they are
informative, not hard pins.

## cfgrib / ecCodes (GRIB only, optional)

`cfgrib` is used only to read the raw ERA5 `.grib` files
(`notebooks/ERA5_Data().ipynb` and the ERA5 download path). It depends on the
**ecCodes** C library, which is *not* a Python package and therefore cannot be
installed by pip alone. In the working environment `cfgrib` is present but
ecCodes was not installed, so `import cfgrib` raises
`RuntimeError: Cannot find the ecCodes library`.

To enable ERA5/GRIB preprocessing:

```bash
conda install -c conda-forge eccodes
# or use the OS package manager, e.g.:
#   apt-get install libeccodes0  (Debian/Ubuntu)
```

All NetCDF-based production (bias correction, energy metrics, ensemble,
final analysis, plotting) uses the `netCDF4` backend and is unaffected.

## Not required

These packages are **not** imported anywhere and are not needed:
`dask`, `geopandas`, `rasterio`, `xskillscore`, `scikit-learn`.
