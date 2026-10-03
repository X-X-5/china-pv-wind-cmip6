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


BASE_DIR = PROJECT_ROOT

ERA5_FILE = (
    get_path("data_raw_era5_raw") / "116b0ea5b6757707487efaa87f9aaae.grib"
)

OUTPUT_FILE = (
    get_path("data_raw_era5_raw") / "ERA5_reference_199401-201412.nc"
)


def drop_auxiliary_coordinates(ds):

    coordinates_to_drop = [
        "number",
        "step",
        "surface",
        "valid_time",
    ]

    for coord in coordinates_to_drop:

        if coord in ds.coords:

            ds = ds.drop_vars(
                coord
            )

    return ds


def check_spatial_alignment(
    ds_main,
    ds_ssrd,
):

    if (
        ds_main.sizes["time"]
        != ds_ssrd.sizes["time"]
    ):

        raise ValueError(
            "Time dimension sizes do not match."
        )

    if not np.array_equal(
        ds_main["latitude"].values,
        ds_ssrd["latitude"].values,
    ):

        raise ValueError(
            "Latitude coordinates do not match."
        )

    if not np.array_equal(
        ds_main["longitude"].values,
        ds_ssrd["longitude"].values,
    ):

        raise ValueError(
            "Longitude coordinates do not match."
        )


def check_monthly_sequence(ds):

    years = ds[
        "time"
    ].dt.year.values

    months = ds[
        "time"
    ].dt.month.values

    month_ids = (
        years * 12
        + months
        - 1
    )

    if len(
        month_ids
    ) != 252:

        raise ValueError(
            f"Unexpected number of months: "
            f"{len(month_ids)}"
        )

    differences = np.diff(
        month_ids
    )

    if not np.all(
        differences == 1
    ):

        raise ValueError(
            "Monthly sequence is not continuous."
        )

    if len(
        np.unique(
            month_ids
        )
    ) != 252:

        raise ValueError(
            "Duplicate months were found."
        )

    if (
        int(years[0]) != 1994
        or int(months[0]) != 1
    ):

        raise ValueError(
            "Unexpected first month."
        )

    if (
        int(years[-1]) != 2014
        or int(months[-1]) != 12
    ):

        raise ValueError(
            "Unexpected last month."
        )


def print_statistics(
    ds,
    variable,
):

    da = ds[
        variable
    ]

    print(
        f"{variable}"
    )

    print(
        f"  shape: {da.shape}"
    )

    print(
        f"  units: "
        f"{da.attrs.get('units', '')}"
    )

    print(
        f"  mean: "
        f"{float(da.mean(skipna=True).values):.6f}"
    )

    print(
        f"  min: "
        f"{float(da.min(skipna=True).values):.6f}"
    )

    print(
        f"  max: "
        f"{float(da.max(skipna=True).values):.6f}"
    )


def main():

    print(
        "=" * 80
    )

    print(
        "ERA5 FINAL REFERENCE PREPARATION"
    )

    print(
        "=" * 80
    )

    ds_main = xr.open_dataset(
        ERA5_FILE,
        engine="cfgrib",
    )

    ds_ssrd = xr.open_dataset(
        ERA5_FILE,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "shortName": "ssrd"
            }
        },
    )

    try:

        print()
        print(
            "Input variables:"
        )

        print(
            f"  Main: "
            f"{list(ds_main.data_vars)}"
        )

        print(
            f"  SSRD: "
            f"{list(ds_ssrd.data_vars)}"
        )

        check_spatial_alignment(
            ds_main,
            ds_ssrd,
        )

        print()
        print(
            "Dimension and spatial alignment: OK"
        )

        ds_main = (
            drop_auxiliary_coordinates(
                ds_main
            )
        )

        ds_ssrd = (
            drop_auxiliary_coordinates(
                ds_ssrd
            )
        )

        ds_ssrd = ds_ssrd.assign_coords(
            time=ds_main[
                "time"
            ].values
        )

        print(
            "SSRD time coordinate standardized."
        )

        tas = (
            ds_main[
                "t2m"
            ]
            .rename(
                "tas"
            )
        )

        sfcwind = (
            np.sqrt(
                ds_main[
                    "u10"
                ] ** 2
                +
                ds_main[
                    "v10"
                ] ** 2
            )
            .rename(
                "sfcWind"
            )
        )

        rsds = (
            ds_ssrd[
                "ssrd"
            ]
            / 86400.0
        ).rename(
            "rsds"
        )

        tas.attrs = {
            "long_name": (
                "2 metre air temperature"
            ),
            "standard_name": (
                "air_temperature"
            ),
            "units": "K",
            "source_variable": "t2m",
        }

        sfcwind.attrs = {
            "long_name": (
                "10 metre wind speed"
            ),
            "standard_name": (
                "wind_speed"
            ),
            "units": "m s-1",
            "source_variables": (
                "u10 v10"
            ),
        }

        rsds.attrs = {
            "long_name": (
                "Surface downwelling "
                "shortwave radiation"
            ),
            "standard_name": (
                "surface_downwelling_"
                "shortwave_flux_in_air"
            ),
            "units": "W m-2",
            "source_variable": "ssrd",
            "conversion": (
                "ssrd divided by 86400"
            ),
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

        if (
            era5[
                "lat"
            ].values[0]
            >
            era5[
                "lat"
            ].values[-1]
        ):

            era5 = era5.sortby(
                "lat"
            )

        era5 = era5.sortby(
            "lon"
        )

        years = era5[
            "time"
        ].dt.year

        months = era5[
            "time"
        ].dt.month

        month_id = (
            years * 12
            + months
            - 1
        )

        start_id = (
            1994 * 12
        )

        end_id = (
            2014 * 12
            + 11
        )

        mask = (
            (month_id >= start_id)
            &
            (month_id <= end_id)
        )

        era5 = era5.sel(
            time=mask
        )

        check_monthly_sequence(
            era5
        )

        if (
            era5.sizes[
                "lat"
            ]
            != 157
        ):

            raise ValueError(
                f"Unexpected latitude size: "
                f"{era5.sizes['lat']}"
            )

        if (
            era5.sizes[
                "lon"
            ]
            != 281
        ):

            raise ValueError(
                f"Unexpected longitude size: "
                f"{era5.sizes['lon']}"
            )

        print()
        print(
            "=" * 80
        )

        print(
            "FINAL DATASET"
        )

        print(
            "=" * 80
        )

        print(
            era5
        )

        print()

        for variable in [
            "tas",
            "sfcWind",
            "rsds",
        ]:

            print_statistics(
                era5,
                variable,
            )

            print()

        print(
            "Coordinate summary:"
        )

        print(
            f"  Time count: "
            f"{era5.sizes['time']}"
        )

        print(
            f"  First time: "
            f"{era5.time.values[0]}"
        )

        print(
            f"  Last time: "
            f"{era5.time.values[-1]}"
        )

        print(
            f"  Latitude: "
            f"{float(era5.lat.min())} "
            f"to "
            f"{float(era5.lat.max())}"
        )

        print(
            f"  Longitude: "
            f"{float(era5.lon.min())} "
            f"to "
            f"{float(era5.lon.max())}"
        )

        print(
            f"  Grid: "
            f"{era5.sizes['lat']} x "
            f"{era5.sizes['lon']}"
        )

        print(
            f"  Latitude direction: "
            f"{'ascending' if era5.lat.values[0] < era5.lat.values[-1] else 'descending'}"
        )

        encoding = {
            "tas": {
                "zlib": True,
                "complevel": 4,
            },
            "sfcWind": {
                "zlib": True,
                "complevel": 4,
            },
            "rsds": {
                "zlib": True,
                "complevel": 4,
            },
        }

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if OUTPUT_FILE.exists():

            OUTPUT_FILE.unlink()

        era5.to_netcdf(
            OUTPUT_FILE,
            engine="netcdf4",
            encoding=encoding,
        )

        print()
        print(
            "=" * 80
        )

        print(
            "ERA5 PREPARATION COMPLETE"
        )

        print(
            "=" * 80
        )

        print(
            f"Output: "
            f"{OUTPUT_FILE}"
        )

        print(
            f"Time count: "
            f"{era5.sizes['time']}"
        )

        print(
            f"Grid: "
            f"{era5.sizes['lat']} x "
            f"{era5.sizes['lon']}"
        )

    finally:

        ds_main.close()
        ds_ssrd.close()


if __name__ == "__main__":

    main()