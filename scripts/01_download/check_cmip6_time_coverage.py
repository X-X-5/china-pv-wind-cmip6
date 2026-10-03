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

DOWNLOAD_ROOT = (
    BASE_DIR
    / "downloaded"
)

OUTPUT_CSV = (
    BASE_DIR
    / "cmip6_time_coverage_report.csv"
)

SUMMARY_TXT = (
    BASE_DIR
    / "cmip6_time_coverage_summary.txt"
)

TARGET_PERIODS = {
    "historical": (1994, 1, 2014, 12),
    "ssp126": (2040, 1, 2100, 12),
    "ssp245": (2040, 1, 2100, 12),
    "ssp585": (2040, 1, 2100, 12),
}


def month_id(year, month):

    return year * 12 + month - 1


def month_label(month_value):

    year = month_value // 12
    month = month_value % 12 + 1

    return f"{year:04d}-{month:02d}"


def expected_months(
    start_year,
    start_month,
    end_year,
    end_month,
):

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


def extract_year_month(value):

    if hasattr(value, "year") and hasattr(
        value,
        "month",
    ):

        return (
            int(value.year),
            int(value.month),
        )

    text = str(value)

    if len(text) >= 7:

        try:

            return (
                int(text[0:4]),
                int(text[5:7]),
            )

        except ValueError:

            pass

    raise ValueError(
        f"Cannot parse time value: {value}"
    )


def get_group_paths():

    groups = {}

    for file_path in DOWNLOAD_ROOT.rglob(
        "*.nc"
    ):

        relative = file_path.relative_to(
            DOWNLOAD_ROOT
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


def read_months_from_file(
    file_path,
):

    ds = None

    try:

        ds = xr.open_dataset(
            file_path,
            decode_times=True,
            chunks=None,
        )

        if "time" not in ds.coords:

            raise KeyError(
                "time coordinate not found"
            )

        values = ds[
            "time"
        ].values

        months = []

        for value in values:

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

        return months

    finally:

        if ds is not None:

            try:
                ds.close()
            except Exception:
                pass


def check_group(
    model,
    experiment,
    variable,
    files,
):

    result = {
        "model": model,
        "experiment": experiment,
        "variable": variable,
        "file_count": len(files),
        "status": "",
        "raw_month_count": 0,
        "unique_month_count": 0,
        "target_expected_months": 0,
        "target_available_months": 0,
        "target_start": "",
        "target_end": "",
        "available_start": "",
        "available_end": "",
        "missing_month_count": 0,
        "duplicate_month_count": 0,
        "missing_months": "",
        "duplicate_months": "",
        "error_type": "",
        "error_message": "",
    }

    if experiment not in TARGET_PERIODS:

        result["status"] = (
            "UNKNOWN_EXPERIMENT"
        )

        return result

    (
        start_year,
        start_month,
        end_year,
        end_month,
    ) = TARGET_PERIODS[
        experiment
    ]

    target_months = expected_months(
        start_year,
        start_month,
        end_year,
        end_month,
    )

    target_set = set(
        target_months
    )

    result[
        "target_expected_months"
    ] = len(
        target_months
    )

    result[
        "target_start"
    ] = month_label(
        target_months[0]
    )

    result[
        "target_end"
    ] = month_label(
        target_months[-1]
    )

    all_months = []

    try:

        for file_path in files:

            months = read_months_from_file(
                file_path
            )

            all_months.extend(
                months
            )

        result[
            "raw_month_count"
        ] = len(
            all_months
        )

        if not all_months:

            result["status"] = (
                "NO_TIME_VALUES"
            )

            return result

        unique_all = sorted(
            set(
                all_months
            )
        )

        result[
            "unique_month_count"
        ] = len(
            unique_all
        )

        result[
            "available_start"
        ] = month_label(
            unique_all[0]
        )

        result[
            "available_end"
        ] = month_label(
            unique_all[-1]
        )

        target_available = [
            value
            for value in all_months
            if value in target_set
        ]

        target_unique = set(
            target_available
        )

        result[
            "target_available_months"
        ] = len(
            target_unique
        )

        missing = sorted(
            target_set
            - target_unique
        )

        counts = Counter(
            target_available
        )

        duplicates = sorted(
            month
            for month, count
            in counts.items()
            if count > 1
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

        if missing:

            result["status"] = (
                "MISSING_MONTHS"
            )

        elif duplicates:

            result["status"] = (
                "DUPLICATE_MONTHS"
            )

        elif (
            len(target_unique)
            != len(target_months)
        ):

            result["status"] = (
                "TARGET_COUNT_MISMATCH"
            )

        else:

            result["status"] = (
                "OK"
            )

        return result

    except Exception as exc:

        result["status"] = (
            "CHECK_FAILED"
        )

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


def write_report(
    rows,
):

    fieldnames = [
        "model",
        "experiment",
        "variable",
        "file_count",
        "status",
        "raw_month_count",
        "unique_month_count",
        "target_expected_months",
        "target_available_months",
        "target_start",
        "target_end",
        "available_start",
        "available_end",
        "missing_month_count",
        "duplicate_month_count",
        "missing_months",
        "duplicate_months",
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
        row["status"]
        for row in rows
    )

    ok_count = status_counts.get(
        "OK",
        0,
    )

    bad_rows = [
        row
        for row in rows
        if row["status"] != "OK"
    ]

    lines = []

    lines.append(
        "CMIP6 TIME COVERAGE SUMMARY"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"Groups checked: {len(rows)}"
    )

    lines.append(
        f"OK groups: {ok_count}"
    )

    lines.append(
        f"Bad groups: {len(bad_rows)}"
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

    if bad_rows:

        lines.append("")
        lines.append(
            "Problem groups:"
        )

        for row in bad_rows:

            lines.append(
                f"  {row['model']} | "
                f"{row['experiment']} | "
                f"{row['variable']} | "
                f"{row['status']} | "
                f"missing="
                f"{row['missing_month_count']} | "
                f"duplicates="
                f"{row['duplicate_month_count']}"
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

    groups = get_group_paths()

    print(
        "=" * 80
    )

    print(
        "CMIP6 TIME COVERAGE CHECK"
    )

    print(
        "=" * 80
    )

    print(
        f"Groups found: "
        f"{len(groups)}"
    )

    print()

    rows = []

    sorted_groups = sorted(
        groups.items()
    )

    for index, (
        key,
        files,
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
            f"files={len(files)}"
        )

        result = check_group(
            model=model,
            experiment=experiment,
            variable=variable,
            files=files,
        )

        rows.append(
            result
        )

        print(
            f"  Status: "
            f"{result['status']}"
        )

        print(
            f"  Available: "
            f"{result['available_start']} "
            f"to "
            f"{result['available_end']}"
        )

        print(
            f"  Target: "
            f"{result['target_start']} "
            f"to "
            f"{result['target_end']}"
        )

        print(
            f"  Target months: "
            f"{result['target_available_months']}/"
            f"{result['target_expected_months']}"
        )

        print(
            f"  Missing: "
            f"{result['missing_month_count']}"
        )

        print(
            f"  Duplicates: "
            f"{result['duplicate_month_count']}"
        )

        write_report(
            rows
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
        f"CSV: {OUTPUT_CSV}"
    )

    print(
        f"TXT: {SUMMARY_TXT}"
    )


if __name__ == "__main__":

    main()