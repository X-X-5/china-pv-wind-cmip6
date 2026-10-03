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
import glob

import numpy as np
import xarray as xr


BASE_DIR = get_path("data_raw_cmip6_downloaded")

RAW_DIR = (
    BASE_DIR
    / "downloaded"
)

OUTPUT_DIR = (
    get_path("data_interim_cmip6_processed_16models")
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

EXPERIMENTS = [
    "historical",
    "ssp126",
    "ssp245",
    "ssp585",
]

VARIABLES = [
    "tas",
    "sfcWind",
    "rsds",
]

PERIODS = {
    "historical": {
        "start_year": 1959,
        "start_month": 1,
        "end_year": 2014,
        "end_month": 12,
        "expected_months": 672,
    },
    "ssp126": {
        "start_year": 2015,
        "start_month": 1,
        "end_year": 2100,
        "end_month": 12,
        "expected_months": 1032,
    },
    "ssp245": {
        "start_year": 2015,
        "start_month": 1,
        "end_year": 2100,
        "end_month": 12,
        "expected_months": 1032,
    },
    "ssp585": {
        "start_year": 2015,
        "start_month": 1,
        "end_year": 2100,
        "end_month": 12,
        "expected_months": 1032,
    },
}


def get_year_month_ids(ds):

    years = ds[
        "time"
    ].dt.year.values.astype(
        np.int64
    )

    months = ds[
        "time"
    ].dt.month.values.astype(
        np.int64
    )

    return (
        years * 12
        + months
        - 1
    )


def select_year_month_range(
    ds,
    start_year,
    start_month,
    end_year,
    end_month,
):

    years = ds[
        "time"
    ].dt.year

    months = ds[
        "time"
    ].dt.month

    start_mask = (
        (years > start_year)
        |
        (
            (years == start_year)
            &
            (months >= start_month)
        )
    )

    end_mask = (
        (years < end_year)
        |
        (
            (years == end_year)
            &
            (months <= end_month)
        )
    )

    mask = (
        start_mask
        &
        end_mask
    )

    return ds.sel(
        time=ds[
            "time"
        ][
            mask
        ]
    )


def find_raw_files(
    model,
    experiment,
    variable,
):

    pattern = str(
        RAW_DIR
        / "**"
        / f"{variable}_Amon_{model}_{experiment}_*.nc"
    )

    files = sorted(
        glob.glob(
            pattern,
            recursive=True,
        )
    )

    return [
        Path(
            file
        )
        for file in files
    ]


def open_and_merge_files(
    files,
    variable,
):

    datasets = []

    try:

        for file in files:

            ds = xr.open_dataset(
                file,
                decode_times=True,
                use_cftime=True,
            )

            if variable not in ds.data_vars:

                ds.close()

                raise ValueError(
                    f"{variable} not found in {file.name}"
                )

            datasets.append(
                ds[
                    [
                        variable
                    ]
                ]
            )

        if len(
            datasets
        ) == 1:

            combined = datasets[
                0
            ].copy()

        else:

            combined = xr.concat(
                datasets,
                dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
                join="override",
            )

        combined = combined.sortby(
            "time"
        )

        combined.load()

    finally:

        for ds in datasets:

            try:

                ds.close()

            except Exception:

                pass

    return combined


def remove_duplicate_months(
    ds,
):

    month_ids = get_year_month_ids(
        ds
    )

    _, unique_indices = np.unique(
        month_ids,
        return_index=True,
    )

    unique_indices = np.sort(
        unique_indices
    )

    if len(
        unique_indices
    ) != ds.sizes[
        "time"
    ]:

        ds = ds.isel(
            time=unique_indices
        )

    return ds


def check_time_period(
    ds,
    experiment,
):

    period = PERIODS[
        experiment
    ]

    expected_months = period[
        "expected_months"
    ]

    actual_months = ds.sizes[
        "time"
    ]

    if actual_months != expected_months:

        raise ValueError(
            f"Expected {expected_months} months, "
            f"found {actual_months}"
        )

    first_year = int(
        ds[
            "time"
        ].dt.year.values[
            0
        ]
    )

    first_month = int(
        ds[
            "time"
        ].dt.month.values[
            0
        ]
    )

    last_year = int(
        ds[
            "time"
        ].dt.year.values[
            -1
        ]
    )

    last_month = int(
        ds[
            "time"
        ].dt.month.values[
            -1
        ]
    )

    expected_start = (
        period[
            "start_year"
        ],
        period[
            "start_month"
        ],
    )

    expected_end = (
        period[
            "end_year"
        ],
        period[
            "end_month"
        ],
    )

    actual_start = (
        first_year,
        first_month,
    )

    actual_end = (
        last_year,
        last_month,
    )

    if actual_start != expected_start:

        raise ValueError(
            f"Unexpected start month: "
            f"{first_year:04d}-{first_month:02d}"
        )

    if actual_end != expected_end:

        raise ValueError(
            f"Unexpected end month: "
            f"{last_year:04d}-{last_month:02d}"
        )

    month_ids = get_year_month_ids(
        ds
    )

    if len(
        np.unique(
            month_ids
        )
    ) != expected_months:

        raise ValueError(
            "Duplicate year-month entries detected"
        )

    expected_ids = np.arange(
        month_ids[
            0
        ],
        month_ids[
            0
        ]
        + expected_months,
    )

    if not np.array_equal(
        month_ids,
        expected_ids,
    ):

        raise ValueError(
            "Year-month sequence is not continuous"
        )


def check_variable(
    ds,
    variable,
):

    if variable not in ds.data_vars:

        raise ValueError(
            f"{variable} missing from dataset"
        )

    da = ds[
        variable
    ]

    if da.sizes[
        "time"
    ] != ds.sizes[
        "time"
    ]:

        raise ValueError(
            f"{variable} time dimension mismatch"
        )

    if not np.issubdtype(
        da.dtype,
        np.number,
    ):

        raise ValueError(
            f"{variable} is not numeric"
        )


def make_output_path(
    model,
    experiment,
    variable,
):

    output_dir = (
        OUTPUT_DIR
        / model
        / experiment
        / variable
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    period = PERIODS[
        experiment
    ]

    start_label = (
        f"{period['start_year']:04d}"
        f"{period['start_month']:02d}"
    )

    end_label = (
        f"{period['end_year']:04d}"
        f"{period['end_month']:02d}"
    )

    filename = (
        f"{variable}_Amon_{model}_{experiment}_"
        f"r1i1p1f1_{start_label}-{end_label}.nc"
    )

    return (
        output_dir
        / filename
    )


def write_dataset(
    ds,
    variable,
    output_file,
):

    encoding = {
        variable: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
    }

    if output_file.exists():

        output_file.unlink()

    ds.to_netcdf(
        output_file,
        engine="netcdf4",
        encoding=encoding,
    )


def process_group(
    model,
    experiment,
    variable,
):

    files = find_raw_files(
        model,
        experiment,
        variable,
    )

    if not files:

        raise FileNotFoundError(
            f"No raw files found for "
            f"{model} | {experiment} | {variable}"
        )

    print()
    print(
        "-" * 100
    )

    print(
        f"{model} | {experiment} | {variable}"
    )

    print(
        "-" * 100
    )

    print(
        f"Raw files: {len(files)}"
    )

    ds = open_and_merge_files(
        files,
        variable,
    )

    try:

        ds = remove_duplicate_months(
            ds
        )

        period = PERIODS[
            experiment
        ]

        ds = select_year_month_range(
            ds,
            period[
                "start_year"
            ],
            period[
                "start_month"
            ],
            period[
                "end_year"
            ],
            period[
                "end_month"
            ],
        )

        check_time_period(
            ds,
            experiment,
        )

        check_variable(
            ds,
            variable,
        )

        output_file = make_output_path(
            model,
            experiment,
            variable,
        )

        write_dataset(
            ds,
            variable,
            output_file,
        )

        print(
            f"Time count: {ds.sizes['time']}"
        )

        print(
            f"Start: "
            f"{int(ds.time.dt.year.values[0]):04d}-"
            f"{int(ds.time.dt.month.values[0]):02d}"
        )

        print(
            f"End: "
            f"{int(ds.time.dt.year.values[-1]):04d}-"
            f"{int(ds.time.dt.month.values[-1]):02d}"
        )

        print(
            f"Output: {output_file}"
        )

    finally:

        ds.close()


def verify_saved_file(
    model,
    experiment,
    variable,
):

    output_file = make_output_path(
        model,
        experiment,
        variable,
    )

    ds = xr.open_dataset(
        output_file,
        decode_times=True,
        use_cftime=True,
    )

    try:

        check_time_period(
            ds,
            experiment,
        )

        check_variable(
            ds,
            variable,
        )

    finally:

        ds.close()


def main():

    print()
    print(
        "=" * 100
    )

    print(
        "CMIP6 PROCESSED_RIGHT PREPARATION"
    )

    print(
        "=" * 100
    )

    print(
        f"Raw directory: {RAW_DIR}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    total_groups = (
        len(
            MODELS
        )
        *
        len(
            EXPERIMENTS
        )
        *
        len(
            VARIABLES
        )
    )

    group_number = 0

    for model in MODELS:

        for experiment in EXPERIMENTS:

            for variable in VARIABLES:

                group_number += 1

                print()
                print(
                    "=" * 100
                )

                print(
                    f"GROUP {group_number}/{total_groups}"
                )

                print(
                    "=" * 100
                )

                try:

                    process_group(
                        model,
                        experiment,
                        variable,
                    )

                    verify_saved_file(
                        model,
                        experiment,
                        variable,
                    )

                    results.append(
                        {
                            "model": model,
                            "experiment": experiment,
                            "variable": variable,
                            "status": "OK",
                            "error": "",
                        }
                    )

                    print(
                        "Status: OK"
                    )

                except Exception as exc:

                    results.append(
                        {
                            "model": model,
                            "experiment": experiment,
                            "variable": variable,
                            "status": "FAILED",
                            "error": (
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                        }
                    )

                    print(
                        "Status: FAILED"
                    )

                    print(
                        f"Error: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

    success = [
        result
        for result in results
        if result[
            "status"
        ] == "OK"
    ]

    failed = [
        result
        for result in results
        if result[
            "status"
        ] == "FAILED"
    ]

    print()
    print(
        "=" * 100
    )

    print(
        "PROCESSING SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Groups: {len(results)}"
    )

    print(
        f"Success: {len(success)}"
    )

    print(
        f"Failed: {len(failed)}"
    )

    if failed:

        print()
        print(
            "Failed groups:"
        )

        for result in failed:

            print(
                f"  {result['model']} | "
                f"{result['experiment']} | "
                f"{result['variable']} | "
                f"{result['error']}"
            )

    else:

        print()
        print(
            "All CMIP6 groups completed successfully."
        )


if __name__ == "__main__":
    main()