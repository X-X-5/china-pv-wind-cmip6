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
from collections import Counter
from pathlib import Path

import xarray as xr


BASE_DIR = get_path("data_raw_cmip6_downloaded")

PROCESSED_ROOT = (
    BASE_DIR
    / "processed"
)

OUTPUT_CSV = (
    BASE_DIR
    / "cmip6_processed_qc_report.csv"
)

SUMMARY_TXT = (
    BASE_DIR
    / "cmip6_processed_qc_summary.txt"
)

EXPECTED_MONTHS = {
    "historical": 252,
    "ssp126": 732,
    "ssp245": 732,
    "ssp585": 732,
}

EXPECTED_PERIODS = {
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


def month_id(
    year,
    month,
):

    return (
        year * 12
        + month
        - 1
    )


def month_label(
    value,
):

    year = (
        value // 12
    )

    month = (
        value % 12
        + 1
    )

    return (
        f"{year:04d}-"
        f"{month:02d}"
    )


def detect_coordinate(
    ds,
    candidates,
):

    for name in candidates:

        if name in ds.coords:
            return name

        if name in ds.dims:
            return name

    return None


def extract_year_month(
    value,
):

    if (
        hasattr(
            value,
            "year",
        )
        and
        hasattr(
            value,
            "month",
        )
    ):

        return (
            int(
                value.year
            ),
            int(
                value.month
            ),
        )

    text = str(
        value
    )

    try:

        return (
            int(
                text[0:4]
            ),
            int(
                text[5:7]
            ),
        )

    except Exception as exc:

        raise ValueError(
            f"Cannot parse time value: "
            f"{value}"
        ) from exc


def expected_sequence(
    experiment,
):

    (
        start_year,
        start_month,
        end_year,
        end_month,
    ) = EXPECTED_PERIODS[
        experiment
    ]

    start = month_id(
        start_year,
        start_month,
    )

    end = month_id(
        end_year,
        end_month,
    )

    return list(
        range(
            start,
            end + 1,
        )
    )


def parse_structure(
    file_path,
):

    relative = file_path.relative_to(
        PROCESSED_ROOT
    )

    parts = relative.parts

    if len(parts) < 4:

        return (
            "",
            "",
            "",
        )

    return (
        parts[0],
        parts[1],
        parts[2],
    )


def check_file(
    file_path,
):

    (
        model,
        experiment,
        variable,
    ) = parse_structure(
        file_path
    )

    result = {
        "model": model,
        "experiment": experiment,
        "variable": variable,
        "filename": file_path.name,
        "status": "",
        "file_size_mb": (
            file_path.stat().st_size
            / 1024
            / 1024
        ),
        "time_count": "",
        "expected_time_count": "",
        "time_start": "",
        "time_end": "",
        "missing_month_count": "",
        "duplicate_month_count": "",
        "missing_months": "",
        "duplicate_months": "",
        "lat_name": "",
        "lon_name": "",
        "shape": "",
        "calendar": "",
        "error_type": "",
        "error_message": "",
    }

    ds = None

    try:

        if file_path.stat().st_size == 0:

            result[
                "status"
            ] = "ZERO_BYTE"

            return result

        ds = xr.open_dataset(
            file_path,
            decode_times=True,
            chunks=None,
        )

        if variable not in ds.data_vars:

            result[
                "status"
            ] = "VARIABLE_MISSING"

            return result

        if "time" not in ds.coords:

            result[
                "status"
            ] = "TIME_MISSING"

            return result

        lat_name = detect_coordinate(
            ds,
            [
                "lat",
                "latitude",
                "nav_lat",
                "y",
            ],
        )

        lon_name = detect_coordinate(
            ds,
            [
                "lon",
                "longitude",
                "nav_lon",
                "x",
            ],
        )

        result[
            "lat_name"
        ] = (
            lat_name
            if lat_name
            else ""
        )

        result[
            "lon_name"
        ] = (
            lon_name
            if lon_name
            else ""
        )

        if (
            lat_name is None
            or lon_name is None
        ):

            result[
                "status"
            ] = (
                "SPATIAL_COORD_MISSING"
            )

            return result

        if experiment not in EXPECTED_MONTHS:

            result[
                "status"
            ] = (
                "UNKNOWN_EXPERIMENT"
            )

            return result

        da = ds[
            variable
        ]

        result[
            "shape"
        ] = str(
            da.shape
        )

        result[
            "calendar"
        ] = (
            ds[
                "time"
            ].encoding.get(
                "calendar",
                "",
            )
        )

        time_values = ds[
            "time"
        ].values

        result[
            "time_count"
        ] = len(
            time_values
        )

        expected_count = (
            EXPECTED_MONTHS[
                experiment
            ]
        )

        result[
            "expected_time_count"
        ] = (
            expected_count
        )

        if len(
            time_values
        ) == 0:

            result[
                "status"
            ] = (
                "NO_TIME_VALUES"
            )

            return result

        months = []

        for value in time_values:

            year, month = (
                extract_year_month(
                    value
                )
            )

            months.append(
                month_id(
                    year,
                    month,
                )
            )

        result[
            "time_start"
        ] = month_label(
            months[0]
        )

        result[
            "time_end"
        ] = month_label(
            months[-1]
        )

        counts = Counter(
            months
        )

        duplicates = sorted(
            value
            for value, count
            in counts.items()
            if count > 1
        )

        expected = (
            expected_sequence(
                experiment
            )
        )

        expected_set = set(
            expected
        )

        month_set = set(
            months
        )

        missing = sorted(
            expected_set
            - month_set
        )

        result[
            "missing_month_count"
        ] = len(
            missing
        )

        result[
            "duplicate_month_count"
        ] = len(
            duplicates
        )

        result[
            "missing_months"
        ] = ";".join(
            month_label(
                value
            )
            for value in missing
        )

        result[
            "duplicate_months"
        ] = ";".join(
            month_label(
                value
            )
            for value in duplicates
        )

        if (
            len(
                time_values
            )
            != expected_count
        ):

            result[
                "status"
            ] = (
                "TIME_COUNT_MISMATCH"
            )

        elif missing:

            result[
                "status"
            ] = (
                "MISSING_MONTHS"
            )

        elif duplicates:

            result[
                "status"
            ] = (
                "DUPLICATE_MONTHS"
            )

        elif months != expected:

            result[
                "status"
            ] = (
                "TIME_ORDER_MISMATCH"
            )

        else:

            result[
                "status"
            ] = "OK"

        return result

    except Exception as exc:

        result[
            "status"
        ] = "OPEN_FAILED"

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

        if ds is not None:

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
        "filename",
        "status",
        "file_size_mb",
        "time_count",
        "expected_time_count",
        "time_start",
        "time_end",
        "missing_month_count",
        "duplicate_month_count",
        "missing_months",
        "duplicate_months",
        "lat_name",
        "lon_name",
        "shape",
        "calendar",
        "error_type",
        "error_message",
    ]

    with OUTPUT_CSV.open(
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


def write_summary(
    rows,
):

    status_counts = Counter(
        row[
            "status"
        ]
        for row in rows
    )

    models = sorted(
        {
            row[
                "model"
            ]
            for row in rows
        }
    )

    ok_rows = [
        row
        for row in rows
        if row[
            "status"
        ] == "OK"
    ]

    bad_rows = [
        row
        for row in rows
        if row[
            "status"
        ] != "OK"
    ]

    calendars = Counter(
        row[
            "calendar"
        ]
        for row in rows
        if row[
            "calendar"
        ]
    )

    lines = []

    lines.append(
        "CMIP6 PROCESSED DATA QC SUMMARY"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"Files checked: "
        f"{len(rows)}"
    )

    lines.append(
        f"Models: "
        f"{len(models)}"
    )

    lines.append(
        f"OK files: "
        f"{len(ok_rows)}"
    )

    lines.append(
        f"Bad files: "
        f"{len(bad_rows)}"
    )

    lines.append("")

    lines.append(
        "Status counts:"
    )

    for status in sorted(
        status_counts
    ):

        lines.append(
            f"  {status}: "
            f"{status_counts[status]}"
        )

    lines.append("")

    lines.append(
        "Calendars:"
    )

    for calendar in sorted(
        calendars
    ):

        lines.append(
            f"  {calendar}: "
            f"{calendars[calendar]}"
        )

    if bad_rows:

        lines.append("")

        lines.append(
            "Problem files:"
        )

        for row in bad_rows:

            lines.append(
                f"  {row['model']} | "
                f"{row['experiment']} | "
                f"{row['variable']} | "
                f"{row['status']} | "
                f"{row['error_type']} | "
                f"{row['error_message']}"
            )

    text = "\n".join(
        lines
    )

    SUMMARY_TXT.write_text(
        text,
        encoding="utf-8",
    )

    return text


def main():

    files = sorted(
        PROCESSED_ROOT.rglob(
            "*.nc"
        )
    )

    print(
        "=" * 80
    )

    print(
        "CMIP6 PROCESSED DATA QC"
    )

    print(
        "=" * 80
    )

    print(
        f"Files found: "
        f"{len(files)}"
    )

    print()

    rows = []

    for index, file_path in enumerate(
        files,
        start=1,
    ):

        print(
            f"[{index}/{len(files)}] "
            f"{file_path.name}"
        )

        result = check_file(
            file_path
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
                f"  Time: "
                f"{result['time_start']} "
                f"to "
                f"{result['time_end']}"
            )

            print(
                f"  Count: "
                f"{result['time_count']}"
            )

            print(
                f"  Calendar: "
                f"{result['calendar']}"
            )

        else:

            print(
                f"  Error: "
                f"{result['error_type']} | "
                f"{result['error_message']}"
            )

    summary = write_summary(
        rows
    )

    print()
    print(
        summary
    )

    print()
    print(
        f"CSV: "
        f"{OUTPUT_CSV}"
    )

    print(
        f"TXT: "
        f"{SUMMARY_TXT}"
    )


if __name__ == "__main__":

    main()