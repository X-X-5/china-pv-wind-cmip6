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

DOWNLOAD_ROOT = (
    BASE_DIR
    / "downloaded"
)

OUTPUT_CSV = (
    BASE_DIR
    / "cmip6_download_integrity_report.csv"
)

SUMMARY_TXT = (
    BASE_DIR
    / "cmip6_download_integrity_summary.txt"
)

EXPECTED_PERIODS = {
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


def detect_coordinate_name(
    ds,
    candidates,
):

    for name in candidates:

        if name in ds.coords:
            return name

        if name in ds.dims:
            return name

    return None


def to_time_string(value):

    try:

        return str(
            value
        )

    except Exception:

        return ""


def check_file(
    file_path,
):

    parts = file_path.relative_to(
        DOWNLOAD_ROOT
    ).parts

    model = (
        parts[0]
        if len(parts) > 0
        else ""
    )

    experiment = (
        parts[1]
        if len(parts) > 1
        else ""
    )

    variable = (
        parts[2]
        if len(parts) > 2
        else ""
    )

    result = {
        "model": model,
        "experiment": experiment,
        "variable": variable,
        "filename": file_path.name,
        "file_size_mb": (
            file_path.stat().st_size
            / 1024
            / 1024
        ),
        "status": "",
        "variable_found": False,
        "time_found": False,
        "lat_found": False,
        "lon_found": False,
        "time_count": "",
        "time_start": "",
        "time_end": "",
        "lat_name": "",
        "lon_name": "",
        "shape": "",
        "error_type": "",
        "error_message": "",
    }

    if file_path.stat().st_size == 0:

        result["status"] = (
            "ZERO_BYTE"
        )

        return result

    ds = None

    try:

        ds = xr.open_dataset(
            file_path,
            decode_times=True,
            chunks=None,
        )

        result[
            "variable_found"
        ] = (
            variable
            in ds.data_vars
        )

        if not result[
            "variable_found"
        ]:

            result["status"] = (
                "VARIABLE_MISSING"
            )

            return result

        da = ds[
            variable
        ]

        result["shape"] = str(
            da.shape
        )

        time_name = (
            detect_coordinate_name(
                ds,
                [
                    "time",
                ],
            )
        )

        lat_name = (
            detect_coordinate_name(
                ds,
                [
                    "lat",
                    "latitude",
                    "nav_lat",
                    "y",
                ],
            )
        )

        lon_name = (
            detect_coordinate_name(
                ds,
                [
                    "lon",
                    "longitude",
                    "nav_lon",
                    "x",
                ],
            )
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

        result[
            "lat_found"
        ] = (
            lat_name is not None
        )

        result[
            "lon_found"
        ] = (
            lon_name is not None
        )

        if time_name is None:

            result["status"] = (
                "TIME_MISSING"
            )

            return result

        result[
            "time_found"
        ] = True

        time_values = ds[
            time_name
        ].values

        result[
            "time_count"
        ] = len(
            time_values
        )

        if len(
            time_values
        ) > 0:

            result[
                "time_start"
            ] = to_time_string(
                time_values[0]
            )

            result[
                "time_end"
            ] = to_time_string(
                time_values[-1]
            )

        if (
            not result[
                "lat_found"
            ]
            or
            not result[
                "lon_found"
            ]
        ):

            result["status"] = (
                "SPATIAL_COORD_MISSING"
            )

            return result

        result["status"] = "OK"

        return result

    except Exception as exc:

        result["status"] = (
            "OPEN_FAILED"
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
        "file_size_mb",
        "status",
        "variable_found",
        "time_found",
        "lat_found",
        "lon_found",
        "time_count",
        "time_start",
        "time_end",
        "lat_name",
        "lon_name",
        "shape",
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


def summarize(
    rows,
):

    status_counts = {}

    for row in rows:

        status = (
            row["status"]
        )

        status_counts[
            status
        ] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

    models = sorted(
        {
            row["model"]
            for row in rows
        }
    )

    experiments = sorted(
        {
            row["experiment"]
            for row in rows
        }
    )

    variables = sorted(
        {
            row["variable"]
            for row in rows
        }
    )

    lines = []

    lines.append(
        "CMIP6 DOWNLOAD INTEGRITY SUMMARY"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"Files checked: {len(rows)}"
    )

    lines.append(
        f"Models: {len(models)}"
    )

    lines.append(
        f"Experiments: {len(experiments)}"
    )

    lines.append(
        f"Variables: {len(variables)}"
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

    bad_rows = [
        row
        for row in rows
        if row["status"] != "OK"
    ]

    lines.append(
        f"Bad files: {len(bad_rows)}"
    )

    if bad_rows:

        lines.append("")

        for row in bad_rows:

            lines.append(
                f"{row['model']} | "
                f"{row['experiment']} | "
                f"{row['variable']} | "
                f"{row['filename']} | "
                f"{row['status']} | "
                f"{row['error_type']} | "
                f"{row['error_message']}"
            )

    summary_text = "\n".join(
        lines
    )

    SUMMARY_TXT.write_text(
        summary_text,
        encoding="utf-8",
    )

    return summary_text


def main():

    files = sorted(
        DOWNLOAD_ROOT.rglob(
            "*.nc"
        )
    )

    print(
        "=" * 80
    )

    print(
        "CMIP6 DOWNLOAD INTEGRITY CHECK"
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
                f"  Shape: "
                f"{result['shape']}"
            )

        else:

            print(
                f"  Error: "
                f"{result['error_type']} "
                f"{result['error_message']}"
            )

        write_report(
            rows
        )

    summary_text = summarize(
        rows
    )

    print()
    print(
        summary_text
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