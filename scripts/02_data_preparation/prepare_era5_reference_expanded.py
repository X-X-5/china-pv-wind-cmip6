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
from pathlib import Path

import numpy as np
import xarray as xr


# Expanded ERA5 reference (area = [56, 70, 15, 140], 1959-2014).
#
# This converts the two staged GRIB files in ``data/raw/era5_expanded/`` into a
# single 672-month NetCDF reference with the exact same variable recipe as the
# original ``prepare_era5_reference_final.py``:
#   tas     = t2m
#   sfcWind = sqrt(u10**2 + v10**2)
#   rsds    = ssrd / 86400
#
# The only difference from the original is the North bound (54 -> 56) so the
# 0.25-deg reference covers CanESM5's northernmost cell centred at 54.4162 N.
#
# Output is written to a staging NetCDF (``..._expanded.nc``) next to the GRIB
# files.  It is NOT the canonical reference until ``verify_expanded_era5.py``
# passes and the file is promoted to ``data/interim/era5_reference/``.

# Staging directory for the expanded GRIB files (see config/project_paths.json).
ERA5_DIR = get_path("data_raw_era5_expanded")

PERIOD_FILES = [
    (1959, 1993, "ERA5_1959_1993_expanded.grib"),
    (1994, 2014, "ERA5_1994_2014_expanded.grib"),
]

OUTPUT_FILE = ERA5_DIR / "ERA5_reference_195901_201412_expanded.nc"

# Expected expanded grid: lat 15..56 step 0.25 -> 165 points, lon 70..140 -> 281
EXPECTED_LAT = 165
EXPECTED_LON = 281
EXPECTED_MONTHS = 672
EXPECTED_LAT_MAX = 56.0
EXPECTED_LAT_MIN = 15.0
EXPECTED_LON_MAX = 140.0
EXPECTED_LON_MIN = 70.0

# Northernmost CanESM5 cell centre that the reference must cover.
REQUIRED_NORTH_COVERAGE = 54.4162


def drop_auxiliary_coordinates(ds):
    coordinates_to_drop = [
        "number",
        "step",
        "surface",
        "valid_time",
    ]
    for coord in coordinates_to_drop:
        if coord in ds.coords:
            ds = ds.drop_vars(coord)
    return ds


def convert_period(grib_path, start_year, end_year):
    """Convert one period GRIB to a (tas, sfcWind, rsds) Dataset on the ERA5 grid."""

    ds_main = xr.open_dataset(
        grib_path,
        engine="cfgrib",
    )

    ds_ssrd = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "shortName": "ssrd"
            }
        },
    )

    try:
        # --- spatial alignment check (time / lat / lon) ---
        if ds_main.sizes["time"] != ds_ssrd.sizes["time"]:
            raise ValueError(f"{grib_path}: time size mismatch between main and ssrd")

        if not np.array_equal(
            ds_main["latitude"].values,
            ds_ssrd["latitude"].values,
        ):
            raise ValueError(f"{grib_path}: latitude mismatch between main and ssrd")

        if not np.array_equal(
            ds_main["longitude"].values,
            ds_ssrd["longitude"].values,
        ):
            raise ValueError(f"{grib_path}: longitude mismatch between main and ssrd")

        ds_main = drop_auxiliary_coordinates(ds_main)
        ds_ssrd = drop_auxiliary_coordinates(ds_ssrd)

        # ssrd is an accumulated field; its time coordinate can differ from the
        # instantaneous/averaged fields, so standardise onto the main time axis.
        ds_ssrd = ds_ssrd.assign_coords(
            time=ds_main["time"].values
        )

        tas = ds_main["t2m"].rename("tas")
        sfcwind = np.sqrt(ds_main["u10"] ** 2 + ds_main["v10"] ** 2).rename("sfcWind")
        rsds = (ds_ssrd["ssrd"] / 86400.0).rename("rsds")

        tas.attrs = {
            "long_name": "2 metre air temperature",
            "standard_name": "air_temperature",
            "units": "K",
            "source_variable": "t2m",
        }

        sfcwind.attrs = {
            "long_name": "10 metre wind speed",
            "standard_name": "wind_speed",
            "units": "m s-1",
            "source_variables": "u10 v10",
        }

        rsds.attrs = {
            "long_name": "Surface downwelling shortwave radiation",
            "standard_name": "surface_downwelling_shortwave_flux_in_air",
            "units": "W m-2",
            "source_variable": "ssrd",
            "conversion": "ssrd divided by 86400",
        }

        era5 = xr.Dataset(
            {
                "tas": tas,
                "sfcWind": sfcwind,
                "rsds": rsds,
            }
        )

        era5 = era5.rename(
            {
                "latitude": "lat",
                "longitude": "lon",
            }
        )

        if era5["lat"].values[0] > era5["lat"].values[-1]:
            era5 = era5.sortby("lat")

        era5 = era5.sortby("lon")

        # Select the requested period and verify continuity.
        years = era5["time"].dt.year
        months = era5["time"].dt.month
        month_id = years * 12 + months - 1

        start_id = start_year * 12
        end_id = end_year * 12 + 11

        era5 = era5.sel(time=(month_id >= start_id) & (month_id <= end_id))

        expected_months = (end_year - start_year + 1) * 12
        if era5.sizes["time"] != expected_months:
            raise ValueError(
                f"{grib_path}: expected {expected_months} months, found {era5.sizes['time']}"
            )

        return era5

    finally:
        ds_main.close()
        ds_ssrd.close()


def check_combined(era5):
    # --- time: 672 continuous months 1959-01 .. 2014-12 ---
    if era5.sizes["time"] != EXPECTED_MONTHS:
        raise ValueError(
            f"Expected {EXPECTED_MONTHS} months, found {era5.sizes['time']}"
        )

    years = era5["time"].dt.year.values
    months = era5["time"].dt.month.values
    month_id = years * 12 + months - 1

    if not np.array_equal(month_id, np.arange(month_id[0], month_id[0] + EXPECTED_MONTHS)):
        raise ValueError("Monthly sequence is not continuous")

    if (int(years[0]), int(months[0])) != (1959, 1):
        raise ValueError("Unexpected first month")

    if (int(years[-1]), int(months[-1])) != (2014, 12):
        raise ValueError("Unexpected last month")

    # --- grid ---
    if era5.sizes["lat"] != EXPECTED_LAT:
        raise ValueError(
            f"Expected {EXPECTED_LAT} latitudes, found {era5.sizes['lat']}"
        )

    if era5.sizes["lon"] != EXPECTED_LON:
        raise ValueError(
            f"Expected {EXPECTED_LON} longitudes, found {era5.sizes['lon']}"
        )

    lat = era5["lat"].values
    lon = era5["lon"].values

    if not np.all(np.diff(lat) > 0):
        raise ValueError("Latitude is not strictly ascending")

    if not np.all(np.diff(lon) > 0):
        raise ValueError("Longitude is not strictly ascending")

    if abs(float(lat.min()) - EXPECTED_LAT_MIN) > 1e-6:
        raise ValueError(f"Unexpected lat min: {lat.min()}")

    if abs(float(lat.max()) - EXPECTED_LAT_MAX) > 1e-6:
        raise ValueError(f"Unexpected lat max: {lat.max()}")

    if abs(float(lon.min()) - EXPECTED_LON_MIN) > 1e-6:
        raise ValueError(f"Unexpected lon min: {lon.min()}")

    if abs(float(lon.max()) - EXPECTED_LON_MAX) > 1e-6:
        raise ValueError(f"Unexpected lon max: {lon.max()}")

    # --- variables present, no NaN anywhere (esp. the extended north strip) ---
    for variable in ["tas", "sfcWind", "rsds"]:
        if variable not in era5.data_vars:
            raise ValueError(f"Missing variable: {variable}")

        values = era5[variable].values
        nan_count = int(np.isnan(values).sum())

        if nan_count != 0:
            raise ValueError(f"{variable} contains {nan_count} NaN values")

        if not np.all(np.isfinite(values)):
            raise ValueError(f"{variable} contains non-finite values")

    # --- coverage of CanESM5's 54.4162 N cell centre ---
    if float(lat.max()) < REQUIRED_NORTH_COVERAGE:
        raise ValueError(
            f"Reference north edge {lat.max()} does not cover {REQUIRED_NORTH_COVERAGE}"
        )


def print_statistics(era5, variable):
    da = era5[variable]
    print(f"  {variable}")
    print(f"    shape: {da.shape}")
    print(f"    units: {da.attrs.get('units', '')}")
    print(f"    mean: {float(da.mean(skipna=True).values):.6f}")
    print(f"    min: {float(da.min(skipna=True).values):.6f}")
    print(f"    max: {float(da.max(skipna=True).values):.6f}")


def main():
    print("=" * 90)
    print("ERA5 EXPANDED REFERENCE PREPARATION (1959-2014, area=[56,70,15,140])")
    print("=" * 90)

    periods = []
    for start_year, end_year, filename in PERIOD_FILES:
        grib_path = ERA5_DIR / filename
        if not grib_path.exists():
            raise FileNotFoundError(f"Missing GRIB file: {grib_path}")

        print()
        print(f"Converting {filename} ({start_year}-{end_year}) ...")
        era5 = convert_period(grib_path, start_year, end_year)
        print(f"  -> {era5.sizes['time']} months, grid {era5.sizes['lat']} x {era5.sizes['lon']}")
        periods.append(era5)

    print()
    print("Combining periods along time ...")
    era5 = xr.concat(periods, dim="time")

    # Ensure ascending time (concat preserves input order; both periods are
    # ascending and 1959-1993 comes first, but sort defensively).
    era5 = era5.sortby("time")

    check_combined(era5)

    print()
    print("=" * 90)
    print("EXPANDED REFERENCE SUMMARY")
    print("=" * 90)
    print(era5)
    print()
    for variable in ["tas", "sfcWind", "rsds"]:
        print_statistics(era5, variable)
        print()
    print(f"  Time count: {era5.sizes['time']}")
    print(f"  First time: {era5.time.values[0]}")
    print(f"  Last time: {era5.time.values[-1]}")
    print(f"  Latitude: {float(era5.lat.min())} to {float(era5.lat.max())}")
    print(f"  Longitude: {float(era5.lon.min())} to {float(era5.lon.max())}")
    print(f"  Grid: {era5.sizes['lat']} x {era5.sizes['lon']}")

    encoding = {
        variable: {
            "zlib": True,
            "complevel": 4,
        }
        for variable in ["tas", "sfcWind", "rsds"]
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    era5.to_netcdf(
        OUTPUT_FILE,
        engine="netcdf4",
        encoding=encoding,
    )

    print()
    print("=" * 90)
    print("ERA5 EXPANDED PREPARATION COMPLETE")
    print("=" * 90)
    print(f"Output (staging): {OUTPUT_FILE}")
    print("NOT YET canonical — run verify_expanded_era5.py, then promote.")


if __name__ == "__main__":
    main()
