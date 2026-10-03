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

import csv
from pathlib import Path

import xarray as xr


BASE_DIR = get_path("data_raw_cmip6_downloaded")

INPUT_ROOT = (
    BASE_DIR
    / "downloaded"
)

OUTPUT_ROOT = (
    BASE_DIR
    / "processed"
)

REPORT_CSV = (
    BASE_DIR
    / "cmip6_prepare_target_periods_report.csv"
)

TARGET_PERIODS = {
    "historical": (
        "1994-01-01",
        "2014-12-31",
    ),
    "ssp126": (
        "2040-01-01",
        "2100-12-31",
    ),
    "ssp245": (
        "2040-01-01",
        "2100-12-31",
    ),
    "ssp585": (
        "2040-01-01",
        "2100-12-31",
    ),
}


def get_groups():

    groups = {}

    for file_path in INPUT_ROOT.rglob(
        "*.nc"
    ):

        relative = file_path.relative_to(
            INPUT_ROOT
        )

        parts = relative.parts

        if len(parts) < 4:
            continue

        model = parts[0]
        experiment = parts[1]
        variable = parts[2]

        key = (
            model,
            experiment,
            variable,
        )

        groups.setdefault(
            key,
            [],
        ).append(
            file_path
        )

    for key in groups:

        groups[key] = sorted(
            groups[key]
        )

    return groups


def build_output_path(
    model,
    experiment,
    variable,
):

    start_date, end_date = (
        TARGET_PERIODS[
            experiment
        ]
    )

    start_code = (
        start_date[0:7]
        .replace("-", "")
    )

    end_code = (
        end_date[0:7]
        .replace("-", "")
    )

    filename = (
        f"{variable}_Amon_"
        f"{model}_"
        f"{experiment}_"
        f"r1i1p1f1_"
        f"{start_code}-"
        f"{end_code}.nc"
    )

    return (
        OUTPUT_ROOT
        / model
        / experiment
        / variable
        / filename
    )


def open_and_combine(
    file_paths,
):

    datasets = []

    try:

        for file_path in file_paths:

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

            combined = datasets[0]

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

        return combined, datasets

    except Exception:

        for ds in datasets:

            try:
                ds.close()
            except Exception:
                pass

        raise


def process_group(
    model,
    experiment,
    variable,
    file_paths,
):

    result = {
        "model": model,
        "experiment": experiment,
        "variable": variable,
        "input_file_count": len(
            file_paths
        ),
        "output_file": "",
        "status": "",
        "input_time_count": "",
        "output_time_count": "",
        "output_start": "",
        "output_end": "",
        "output_size_mb": "",
        "error_type": "",
        "error_message": "",
    }

    combined = None
    source_datasets = []

    try:

        combined, source_datasets = (
            open_and_combine(
                file_paths
            )
        )

        if variable not in combined.data_vars:

            raise KeyError(
                f"Variable not found: {variable}"
            )

        if "time" not in combined.coords:

            raise KeyError(
                "Time coordinate not found"
            )

        result[
            "input_time_count"
        ] = combined.sizes[
            "time"
        ]

        start_date, end_date = (
            TARGET_PERIODS[
                experiment
            ]
        )

        subset = combined.sel(
            time=slice(
                start_date,
                end_date,
            )
        )

        expected_count = (
            252
            if experiment == "historical"
            else 732
        )

        actual_count = subset.sizes[
            "time"
        ]

        result[
            "output_time_count"
        ] = actual_count

        if actual_count != expected_count:

            raise ValueError(
                f"Unexpected time count: "
                f"{actual_count} != "
                f"{expected_count}"
            )

        time_values = subset[
            "time"
        ].values

        result[
            "output_start"
        ] = str(
            time_values[0]
        )

        result[
            "output_end"
        ] = str(
            time_values[-1]
        )

        output_path = (
            build_output_path(
                model,
                experiment,
                variable,
            )
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

        result[
            "output_file"
        ] = str(
            output_path
        )

        result[
            "output_size_mb"
        ] = (
            output_path.stat().st_size
            / 1024
            / 1024
        )

        result[
            "status"
        ] = "OK"

        return result

    except Exception as exc:

        result[
            "status"
        ] = "FAILED"

        result[
            "error_type"
        ] = type(
            exc
        ).__name__

        result[
            "error_message"
        ] = str(
            exc
        )

        return result

    finally:

        for ds in source_datasets:

            try:
                ds.close()
            except Exception:
                pass


def write_report(
    rows,
):

    fieldnames = [
        "model",
        "experiment",
        "variable",
        "input_file_count",
        "output_file",
        "status",
        "input_time_count",
        "output_time_count",
        "output_start",
        "output_end",
        "output_size_mb",
        "error_type",
        "error_message",
    ]

    with REPORT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def main():

    groups = get_groups()

    sorted_groups = sorted(
        groups.items()
    )

    print(
        "=" * 80
    )

    print(
        "CMIP6 TARGET PERIOD PREPARATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Groups: "
        f"{len(sorted_groups)}"
    )

    print(
        f"Input root: "
        f"{INPUT_ROOT}"
    )

    print(
        f"Output root: "
        f"{OUTPUT_ROOT}"
    )

    print()

    rows = []

    for index, (
        key,
        file_paths,
    ) in enumerate(
        sorted_groups,
        start=1,
    ):

        (
            model,
            experiment,
            variable,
        ) = key

        print(
            f"[{index}/{len(sorted_groups)}] "
            f"{model} | "
            f"{experiment} | "
            f"{variable} | "
            f"files={len(file_paths)}"
        )

        result = process_group(
            model=model,
            experiment=experiment,
            variable=variable,
            file_paths=file_paths,
        )

        rows.append(
            result
        )

        write_report(
            rows
        )

        print(
            f"  Status: "
            f"{result['status']}"
        )

        if result[
            "status"
        ] == "OK":

            print(
                f"  Time count: "
                f"{result['output_time_count']}"
            )

            print(
                f"  Time: "
                f"{result['output_start']} "
                f"to "
                f"{result['output_end']}"
            )

            print(
                f"  Size: "
                f"{float(result['output_size_mb']):.1f} MB"
            )

        else:

            print(
                f"  Error: "
                f"{result['error_type']}: "
                f"{result['error_message']}"
            )

    ok_rows = [
        row
        for row in rows
        if row[
            "status"
        ] == "OK"
    ]

    failed_rows = [
        row
        for row in rows
        if row[
            "status"
        ] != "OK"
    ]

    print()
    print(
        "=" * 80
    )

    print(
        "PREPARATION SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"Groups processed: "
        f"{len(rows)}"
    )

    print(
        f"OK: "
        f"{len(ok_rows)}"
    )

    print(
        f"Failed: "
        f"{len(failed_rows)}"
    )

    print(
        f"Report: "
        f"{REPORT_CSV}"
    )

    if len(rows) == 192 and not failed_rows:

        print()
        print(
            "All 192 groups were prepared successfully."
        )


if __name__ == "__main__":

    main()