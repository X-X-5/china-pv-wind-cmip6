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
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = PROJECT_ROOT

ERA5_FILE = (
    get_path("data_interim_era5_reference") / "ERA5_reference_195901_201412.nc"
)

CMIP6_ROOT = (
    get_path("data_interim_cmip6_china_clipped")
)

OUTPUT_ROOT = (
    get_path("data_interim_era5_on_gcm_grid")
)

MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "AWI-CM-1-1-MR",
    "BCC-CSM2-MR",
    "CanESM5",
    "CESM2-WACCM",
    "CMCC-CM2-SR5",
    "CMCC-ESM2",
    "FGOALS-f3-L",
    "FIO-ESM-2-0",
    "IPSL-CM6A-LR",
    "KACE-1-0-G",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "TaiESM1",
]

VARIABLES = [
    "tas",
    "sfcWind",
    "rsds",
]

EXPECTED_MONTHS = 672

EXPECTED_START = (
    1959,
    1,
)

EXPECTED_END = (
    2014,
    12,
)


def get_time_coder():

    return xr.coders.CFDatetimeCoder(
        use_cftime=True
    )


def get_month_ids(ds):

    years = (
        ds["time"]
        .dt.year
        .values
        .astype(np.int64)
    )

    months = (
        ds["time"]
        .dt.month
        .values
        .astype(np.int64)
    )

    return (
        years * 12
        + months
        - 1
    )


def check_time(ds):

    if ds.sizes["time"] != EXPECTED_MONTHS:

        raise ValueError(
            f"Expected {EXPECTED_MONTHS} months, "
            f"found {ds.sizes['time']}"
        )

    start = (
        int(ds.time.dt.year.values[0]),
        int(ds.time.dt.month.values[0]),
    )

    end = (
        int(ds.time.dt.year.values[-1]),
        int(ds.time.dt.month.values[-1]),
    )

    if start != EXPECTED_START:

        raise ValueError(
            f"Unexpected start month: "
            f"{start[0]:04d}-{start[1]:02d}"
        )

    if end != EXPECTED_END:

        raise ValueError(
            f"Unexpected end month: "
            f"{end[0]:04d}-{end[1]:02d}"
        )

    month_ids = get_month_ids(ds)

    if len(np.unique(month_ids)) != EXPECTED_MONTHS:

        raise ValueError(
            "Duplicate year-month entries detected"
        )

    expected_ids = np.arange(
        month_ids[0],
        month_ids[0] + EXPECTED_MONTHS,
    )

    if not np.array_equal(
        month_ids,
        expected_ids,
    ):

        raise ValueError(
            "Year-month sequence is not continuous"
        )


def check_era5(ds):

    check_time(ds)

    for variable in VARIABLES:

        if variable not in ds.data_vars:

            raise ValueError(
                f"ERA5 variable missing: {variable}"
            )

    if "lat" not in ds.coords:

        raise ValueError(
            "ERA5 latitude coordinate missing"
        )

    if "lon" not in ds.coords:

        raise ValueError(
            "ERA5 longitude coordinate missing"
        )

    if ds["lat"].ndim != 1:

        raise ValueError(
            "ERA5 latitude is not one-dimensional"
        )

    if ds["lon"].ndim != 1:

        raise ValueError(
            "ERA5 longitude is not one-dimensional"
        )

    lat = ds["lat"].values

    lon = ds["lon"].values

    if not np.all(np.diff(lat) > 0):

        raise ValueError(
            "ERA5 latitude is not ascending"
        )

    if not np.all(np.diff(lon) > 0):

        raise ValueError(
            "ERA5 longitude is not ascending"
        )


def find_cmip6_file(
    model,
    variable,
):

    directory = (
        CMIP6_ROOT
        / model
        / "historical"
        / variable
    )

    files = sorted(
        directory.glob("*.nc")
    )

    if len(files) != 1:

        raise ValueError(
            f"{model} | {variable}: "
            f"expected 1 file, found {len(files)}"
        )

    return files[0]


def read_model_grid(
    model,
):

    time_coder = get_time_coder()

    datasets = {}

    try:

        for variable in VARIABLES:

            file_path = find_cmip6_file(
                model,
                variable,
            )

            ds = xr.open_dataset(
                file_path,
                decode_times=time_coder,
            )

            datasets[variable] = ds

            check_time(ds)

            if "lat" not in ds.coords:

                raise ValueError(
                    f"{model} | {variable}: lat missing"
                )

            if "lon" not in ds.coords:

                raise ValueError(
                    f"{model} | {variable}: lon missing"
                )

        reference_lat = (
            datasets["tas"]["lat"].values
        )

        reference_lon = (
            datasets["tas"]["lon"].values
        )

        for variable in [
            "sfcWind",
            "rsds",
        ]:

            lat = (
                datasets[variable]["lat"].values
            )

            lon = (
                datasets[variable]["lon"].values
            )

            if not np.array_equal(
                reference_lat,
                lat,
            ):

                if not np.allclose(
                    reference_lat,
                    lat,
                    rtol=0.0,
                    atol=1e-10,
                ):

                    raise ValueError(
                        f"{model}: latitude grid mismatch "
                        f"between variables"
                    )

            if not np.array_equal(
                reference_lon,
                lon,
            ):

                if not np.allclose(
                    reference_lon,
                    lon,
                    rtol=0.0,
                    atol=1e-10,
                ):

                    raise ValueError(
                        f"{model}: longitude grid mismatch "
                        f"between variables"
                    )

        return (
            reference_lat.copy(),
            reference_lon.copy(),
        )

    finally:

        for ds in datasets.values():

            ds.close()


def check_target_inside_era5(
    era5,
    target_lat,
    target_lon,
):

    era5_lat_min = float(
        era5["lat"].min().values
    )

    era5_lat_max = float(
        era5["lat"].max().values
    )

    era5_lon_min = float(
        era5["lon"].min().values
    )

    era5_lon_max = float(
        era5["lon"].max().values
    )

    target_lat_min = float(
        np.min(target_lat)
    )

    target_lat_max = float(
        np.max(target_lat)
    )

    target_lon_min = float(
        np.min(target_lon)
    )

    target_lon_max = float(
        np.max(target_lon)
    )

    if target_lat_min < era5_lat_min:

        raise ValueError(
            "Target latitude extends below ERA5 domain"
        )

    if target_lat_max > era5_lat_max:

        raise ValueError(
            "Target latitude extends above ERA5 domain"
        )

    if target_lon_min < era5_lon_min:

        raise ValueError(
            "Target longitude extends below ERA5 domain"
        )

    if target_lon_max > era5_lon_max:

        raise ValueError(
            "Target longitude extends above ERA5 domain"
        )


def interpolate_era5(
    era5,
    target_lat,
    target_lon,
):

    lat_da = xr.DataArray(
        target_lat,
        dims="lat",
        coords={
            "lat": target_lat,
        },
    )

    lon_da = xr.DataArray(
        target_lon,
        dims="lon",
        coords={
            "lon": target_lon,
        },
    )

    interpolated = era5[
        VARIABLES
    ].interp(
        lat=lat_da,
        lon=lon_da,
        method="linear",
    )

    interpolated = interpolated.transpose(
        "time",
        "lat",
        "lon",
    )

    return interpolated


def check_interpolated(
    ds,
    target_lat,
    target_lon,
):

    check_time(ds)

    if ds.sizes["lat"] != len(
        target_lat
    ):

        raise ValueError(
            "Interpolated latitude count mismatch"
        )

    if ds.sizes["lon"] != len(
        target_lon
    ):

        raise ValueError(
            "Interpolated longitude count mismatch"
        )

    if not np.allclose(
        ds["lat"].values,
        target_lat,
        rtol=0.0,
        atol=1e-10,
    ):

        raise ValueError(
            "Interpolated latitude coordinates mismatch"
        )

    if not np.allclose(
        ds["lon"].values,
        target_lon,
        rtol=0.0,
        atol=1e-10,
    ):

        raise ValueError(
            "Interpolated longitude coordinates mismatch"
        )

    for variable in VARIABLES:

        if variable not in ds.data_vars:

            raise ValueError(
                f"Interpolated variable missing: {variable}"
            )

        values = ds[
            variable
        ].values

        nan_count = int(
            np.isnan(values).sum()
        )

        if nan_count != 0:

            raise ValueError(
                f"{variable} contains "
                f"{nan_count} NaN values"
            )

        if not np.all(
            np.isfinite(values)
        ):

            raise ValueError(
                f"{variable} contains "
                f"non-finite values"
            )


def make_output_file(
    model,
):

    directory = (
        OUTPUT_ROOT
        / model
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        directory
        / f"ERA5_{model}_195901-201412.nc"
    )


def write_output(
    ds,
    output_file,
):

    encoding = {}

    for variable in VARIABLES:

        encoding[
            variable
        ] = {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }

    if output_file.exists():

        output_file.unlink()

    ds.to_netcdf(
        output_file,
        engine="netcdf4",
        encoding=encoding,
    )


def verify_saved_file(
    output_file,
    target_lat,
    target_lon,
):

    time_coder = get_time_coder()

    ds = xr.open_dataset(
        output_file,
        decode_times=time_coder,
    )

    try:

        check_interpolated(
            ds,
            target_lat,
            target_lon,
        )

    finally:

        ds.close()


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Interpolate the ERA5 reference onto each model's clipped grid."
        )
    )

    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Optional subset of model names to process (default: all models "
            "in the script's MODELS list)."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    models = (
        args.models
        if args.models
        else MODELS
    )

    print()
    print("=" * 120)
    print("ERA5 TO GCM GRID INTERPOLATION")
    print("=" * 120)

    print(
        f"ERA5 file: {ERA5_FILE}"
    )

    print(
        f"CMIP6 root: {CMIP6_ROOT}"
    )

    print(
        f"Output root: {OUTPUT_ROOT}"
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    time_coder = get_time_coder()

    era5 = xr.open_dataset(
        ERA5_FILE,
        decode_times=time_coder,
    )

    success_count = 0
    failed_count = 0
    failures = []

    try:

        check_era5(
            era5
        )

        print()
        print(
            f"ERA5 time count: "
            f"{era5.sizes['time']}"
        )

        print(
            f"ERA5 grid: "
            f"{era5.sizes['lat']} x "
            f"{era5.sizes['lon']}"
        )

        print(
            f"ERA5 latitude: "
            f"{float(era5.lat.min().values):.6f} "
            f"to "
            f"{float(era5.lat.max().values):.6f}"
        )

        print(
            f"ERA5 longitude: "
            f"{float(era5.lon.min().values):.6f} "
            f"to "
            f"{float(era5.lon.max().values):.6f}"
        )

        for index, model in enumerate(
            models,
            start=1,
        ):

            print()
            print("-" * 120)

            print(
                f"MODEL {index}/{len(models)}: "
                f"{model}"
            )

            print("-" * 120)

            interpolated = None

            try:

                (
                    target_lat,
                    target_lon,
                ) = read_model_grid(
                    model
                )

                check_target_inside_era5(
                    era5,
                    target_lat,
                    target_lon,
                )

                print(
                    f"Target grid: "
                    f"{len(target_lat)} x "
                    f"{len(target_lon)}"
                )

                print(
                    f"Target latitude: "
                    f"{target_lat.min():.6f} "
                    f"to "
                    f"{target_lat.max():.6f}"
                )

                print(
                    f"Target longitude: "
                    f"{target_lon.min():.6f} "
                    f"to "
                    f"{target_lon.max():.6f}"
                )

                interpolated = interpolate_era5(
                    era5,
                    target_lat,
                    target_lon,
                )

                interpolated.load()

                check_interpolated(
                    interpolated,
                    target_lat,
                    target_lon,
                )

                output_file = make_output_file(
                    model
                )

                write_output(
                    interpolated,
                    output_file,
                )

                verify_saved_file(
                    output_file,
                    target_lat,
                    target_lon,
                )

                for variable in VARIABLES:

                    da = interpolated[
                        variable
                    ]

                    print(
                        f"{variable}: "
                        f"mean={float(da.mean().values):.6f}, "
                        f"min={float(da.min().values):.6f}, "
                        f"max={float(da.max().values):.6f}"
                    )

                print(
                    f"Output: {output_file}"
                )

                print(
                    "Status: OK"
                )

                success_count += 1

            except Exception as exc:

                failed_count += 1

                failures.append(
                    (
                        model,
                        f"{type(exc).__name__}: {exc}",
                    )
                )

                print(
                    "Status: FAILED"
                )

                print(
                    f"Error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            finally:

                if interpolated is not None:

                    interpolated.close()

    finally:

        era5.close()

    print()
    print("=" * 120)
    print("INTERPOLATION SUMMARY")
    print("=" * 120)

    print(
        f"Models: {len(models)}"
    )

    print(
        f"Success: {success_count}"
    )

    print(
        f"Failed: {failed_count}"
    )

    if failures:

        print()
        print(
            "FAILED MODELS"
        )

        for (
            model,
            error,
        ) in failures:

            print(
                f"{model} | {error}"
            )

    else:

        print()
        print(
            "ALL ERA5 TO GCM INTERPOLATIONS PASSED"
        )


if __name__ == "__main__":
    main()