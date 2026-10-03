from __future__ import annotations
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

import xarray as xr


BASE_DIR = get_path("data_raw_cmip6_downloaded")

INPUT_ROOT = (
    BASE_DIR
    / "downloaded"
    / "KACE-1-0-G"
)

OUTPUT_ROOT = (
    BASE_DIR
    / "processed"
    / "KACE-1-0-G"
)

TARGET_PERIODS = {
    "historical": (
        1994,
        1,
        2014,
        12,
    ),
    "ssp126": (
        2040,
        1,
        2100,
        12,
    ),
    "ssp245": (
        2040,
        1,
        2100,
        12,
    ),
    "ssp585": (
        2040,
        1,
        2100,
        12,
    ),
}

VARIABLES = [
    "rsds",
    "sfcWind",
    "tas",
]


def month_id(
    year,
    month,
):

    return (
        year * 12
        + month
        - 1
    )


def select_year_month(
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

    current = (
        years * 12
        + months
        - 1
    )

    start_id = month_id(
        start_year,
        start_month,
    )

    end_id = month_id(
        end_year,
        end_month,
    )

    mask = (
        (current >= start_id)
        &
        (current <= end_id)
    )

    return ds.sel(
        time=mask
    )


def build_output_path(
    experiment,
    variable,
):

    (
        start_year,
        start_month,
        end_year,
        end_month,
    ) = TARGET_PERIODS[
        experiment
    ]

    start_code = (
        f"{start_year:04d}"
        f"{start_month:02d}"
    )

    end_code = (
        f"{end_year:04d}"
        f"{end_month:02d}"
    )

    filename = (
        f"{variable}_Amon_"
        f"KACE-1-0-G_"
        f"{experiment}_"
        f"r1i1p1f1_"
        f"{start_code}-"
        f"{end_code}.nc"
    )

    return (
        OUTPUT_ROOT
        / experiment
        / variable
        / filename
    )


def process_group(
    experiment,
    variable,
):

    input_dir = (
        INPUT_ROOT
        / experiment
        / variable
    )

    files = sorted(
        input_dir.glob(
            "*.nc"
        )
    )

    if not files:

        raise FileNotFoundError(
            f"No files found: {input_dir}"
        )

    datasets = []

    try:

        for file_path in files:

            ds = xr.open_dataset(
                file_path,
                decode_times=True,
                chunks=None,
            )

            datasets.append(
                ds
            )

        if len(
            datasets
        ) == 1:

            combined = (
                datasets[0]
            )

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

        (
            start_year,
            start_month,
            end_year,
            end_month,
        ) = TARGET_PERIODS[
            experiment
        ]

        subset = select_year_month(
            combined,
            start_year,
            start_month,
            end_year,
            end_month,
        )

        expected_count = (
            252
            if experiment
            == "historical"
            else 732
        )

        actual_count = subset.sizes[
            "time"
        ]

        if actual_count != expected_count:

            raise ValueError(
                f"Unexpected time count: "
                f"{actual_count} != "
                f"{expected_count}"
            )

        output_path = build_output_path(
            experiment,
            variable,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output_path.exists():

            output_path.unlink()

        output_ds = subset[
            [
                variable
            ]
        ]

        encoding = {
            variable: {
                "zlib": True,
                "complevel": 4,
            }
        }

        output_ds.to_netcdf(
            output_path,
            engine="netcdf4",
            encoding=encoding,
        )

        print(
            f"OK | "
            f"{experiment} | "
            f"{variable} | "
            f"time={actual_count} | "
            f"{subset.time.values[0]} "
            f"to "
            f"{subset.time.values[-1]}"
        )

    finally:

        for ds in datasets:

            try:
                ds.close()
            except Exception:
                pass


def main():

    print(
        "=" * 80
    )

    print(
        "KACE-1-0-G TARGET PERIOD PREPARATION"
    )

    print(
        "=" * 80
    )

    success = 0
    failed = 0

    for experiment in TARGET_PERIODS:

        for variable in VARIABLES:

            try:

                process_group(
                    experiment,
                    variable,
                )

                success += 1

            except Exception as exc:

                failed += 1

                print(
                    f"FAILED | "
                    f"{experiment} | "
                    f"{variable} | "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

    print()
    print(
        "=" * 80
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"Success: {success}"
    )

    print(
        f"Failed: {failed}"
    )


if __name__ == "__main__":

    main()