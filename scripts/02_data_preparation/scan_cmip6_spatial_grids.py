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

import numpy as np
import xarray as xr


BASE_DIR = get_path("data_raw_cmip6_downloaded")

PROCESSED_ROOT = (
    BASE_DIR
    / "processed"
)

OUTPUT_CSV = (
    BASE_DIR
    / "cmip6_spatial_grid_report.csv"
)

SUMMARY_TXT = (
    BASE_DIR
    / "cmip6_spatial_grid_summary.txt"
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


def estimate_resolution(
    values,
):

    values = np.asarray(
        values
    )

    if values.ndim != 1:

        return None

    if values.size < 2:

        return None

    diffs = np.diff(
        values.astype(float)
    )

    finite = diffs[
        np.isfinite(
            diffs
        )
    ]

    if finite.size == 0:

        return None

    return float(
        np.median(
            np.abs(
                finite
            )
        )
    )


def is_regular_1d(
    values,
    tolerance=1e-6,
):

    values = np.asarray(
        values
    )

    if values.ndim != 1:

        return False

    if values.size < 3:

        return True

    diffs = np.diff(
        values.astype(float)
    )

    if not np.all(
        np.isfinite(
            diffs
        )
    ):

        return False

    return bool(
        np.allclose(
            diffs,
            diffs[0],
            rtol=0,
            atol=tolerance,
        )
    )


def monotonic_direction(
    values,
):

    values = np.asarray(
        values
    )

    if values.ndim != 1:

        return "not_1d"

    if values.size < 2:

        return "single"

    diffs = np.diff(
        values.astype(float)
    )

    if np.all(
        diffs > 0
    ):

        return "ascending"

    if np.all(
        diffs < 0
    ):

        return "descending"

    return "non_monotonic"


def classify_longitude(
    values,
):

    values = np.asarray(
        values
    ).astype(float)

    finite = values[
        np.isfinite(
            values
        )
    ]

    if finite.size == 0:

        return "unknown"

    lon_min = float(
        np.min(
            finite
        )
    )

    lon_max = float(
        np.max(
            finite
        )
    )

    if (
        lon_min >= 0
        and lon_max > 180
    ):

        return "0_to_360"

    if (
        lon_min < 0
        and lon_max <= 180
    ):

        return "minus180_to_180"

    return "other"


def build_grid_signature(
    lat_values,
    lon_values,
):

    lat_values = np.asarray(
        lat_values
    )

    lon_values = np.asarray(
        lon_values
    )

    return (
        lat_values.shape,
        lon_values.shape,
        float(
            np.nanmin(
                lat_values
            )
        ),
        float(
            np.nanmax(
                lat_values
            )
        ),
        float(
            np.nanmin(
                lon_values
            )
        ),
        float(
            np.nanmax(
                lon_values
            )
        ),
        estimate_resolution(
            lat_values
        ),
        estimate_resolution(
            lon_values
        ),
    )


def inspect_file(
    file_path,
):

    relative = file_path.relative_to(
        PROCESSED_ROOT
    )

    parts = relative.parts

    model = parts[0]
    experiment = parts[1]
    variable = parts[2]

    result = {
        "model": model,
        "experiment": experiment,
        "variable": variable,
        "filename": file_path.name,
        "status": "",
        "lat_name": "",
        "lon_name": "",
        "lat_ndim": "",
        "lon_ndim": "",
        "lat_size": "",
        "lon_size": "",
        "lat_min": "",
        "lat_max": "",
        "lon_min": "",
        "lon_max": "",
        "lat_resolution": "",
        "lon_resolution": "",
        "lat_direction": "",
        "lon_direction": "",
        "lat_regular": "",
        "lon_regular": "",
        "lon_convention": "",
        "grid_signature": "",
        "error_type": "",
        "error_message": "",
    }

    ds = None

    try:

        ds = xr.open_dataset(
            file_path,
            decode_times=False,
            chunks=None,
        )

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

        if (
            lat_name is None
            or lon_name is None
        ):

            result[
                "status"
            ] = "COORDINATE_MISSING"

            return result

        lat = ds[
            lat_name
        ].values

        lon = ds[
            lon_name
        ].values

        result[
            "lat_name"
        ] = lat_name

        result[
            "lon_name"
        ] = lon_name

        result[
            "lat_ndim"
        ] = lat.ndim

        result[
            "lon_ndim"
        ] = lon.ndim

        result[
            "lat_size"
        ] = lat.size

        result[
            "lon_size"
        ] = lon.size

        result[
            "lat_min"
        ] = float(
            np.nanmin(
                lat
            )
        )

        result[
            "lat_max"
        ] = float(
            np.nanmax(
                lat
            )
        )

        result[
            "lon_min"
        ] = float(
            np.nanmin(
                lon
            )
        )

        result[
            "lon_max"
        ] = float(
            np.nanmax(
                lon
            )
        )

        lat_res = estimate_resolution(
            lat
        )

        lon_res = estimate_resolution(
            lon
        )

        result[
            "lat_resolution"
        ] = (
            lat_res
            if lat_res is not None
            else ""
        )

        result[
            "lon_resolution"
        ] = (
            lon_res
            if lon_res is not None
            else ""
        )

        result[
            "lat_direction"
        ] = monotonic_direction(
            lat
        )

        result[
            "lon_direction"
        ] = monotonic_direction(
            lon
        )

        result[
            "lat_regular"
        ] = is_regular_1d(
            lat
        )

        result[
            "lon_regular"
        ] = is_regular_1d(
            lon
        )

        result[
            "lon_convention"
        ] = classify_longitude(
            lon
        )

        signature = build_grid_signature(
            lat,
            lon,
        )

        result[
            "grid_signature"
        ] = repr(
            signature
        )

        if (
            lat.ndim == 1
            and lon.ndim == 1
        ):

            result[
                "status"
            ] = "OK_1D"

        else:

            result[
                "status"
            ] = "NON_1D_GRID"

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
        "lat_name",
        "lon_name",
        "lat_ndim",
        "lon_ndim",
        "lat_size",
        "lon_size",
        "lat_min",
        "lat_max",
        "lon_min",
        "lon_max",
        "lat_resolution",
        "lon_resolution",
        "lat_direction",
        "lon_direction",
        "lat_regular",
        "lon_regular",
        "lon_convention",
        "grid_signature",
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

    models = sorted(
        {
            row["model"]
            for row in rows
        }
    )

    lines = []

    lines.append(
        "CMIP6 SPATIAL GRID SUMMARY"
    )

    lines.append(
        "=" * 100
    )

    for model in models:

        model_rows = [
            row
            for row in rows
            if row[
                "model"
            ] == model
        ]

        signatures = {
            row[
                "grid_signature"
            ]
            for row in model_rows
            if row[
                "grid_signature"
            ]
        }

        statuses = {
            row[
                "status"
            ]
            for row in model_rows
        }

        first = model_rows[
            0
        ]

        lines.append(
            f"{model}"
        )

        lines.append(
            f"  Files: "
            f"{len(model_rows)}"
        )

        lines.append(
            f"  Statuses: "
            f"{', '.join(sorted(statuses))}"
        )

        lines.append(
            f"  Coordinate names: "
            f"{first['lat_name']}, "
            f"{first['lon_name']}"
        )

        lines.append(
            f"  Dimensions: "
            f"{first['lat_size']} x "
            f"{first['lon_size']}"
        )

        lines.append(
            f"  Latitude: "
            f"{first['lat_min']} to "
            f"{first['lat_max']}"
        )

        lines.append(
            f"  Longitude: "
            f"{first['lon_min']} to "
            f"{first['lon_max']}"
        )

        lines.append(
            f"  Resolution: "
            f"{first['lat_resolution']} x "
            f"{first['lon_resolution']}"
        )

        lines.append(
            f"  Latitude direction: "
            f"{first['lat_direction']}"
        )

        lines.append(
            f"  Longitude convention: "
            f"{first['lon_convention']}"
        )

        lines.append(
            f"  Unique grid signatures: "
            f"{len(signatures)}"
        )

        lines.append("")

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
        "=" * 100
    )

    print(
        "CMIP6 SPATIAL GRID SCAN"
    )

    print(
        "=" * 100
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

        result = inspect_file(
            file_path
        )

        rows.append(
            result
        )

        write_report(
            rows
        )

        print(
            f"[{index}/{len(files)}] "
            f"{result['model']} | "
            f"{result['experiment']} | "
            f"{result['variable']}"
        )

        print(
            f"  Status: "
            f"{result['status']}"
        )

        print(
            f"  Grid: "
            f"{result['lat_size']} x "
            f"{result['lon_size']}"
        )

        print(
            f"  Resolution: "
            f"{result['lat_resolution']} x "
            f"{result['lon_resolution']}"
        )

        print(
            f"  Longitude: "
            f"{result['lon_convention']}"
        )

    summary = write_summary(
        rows
    )

    print()
    print(
        summary
    )

    print(
        f"CSV: {OUTPUT_CSV}"
    )

    print(
        f"TXT: {SUMMARY_TXT}"
    )


if __name__ == "__main__":
    main()