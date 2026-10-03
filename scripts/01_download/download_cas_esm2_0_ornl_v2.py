"""
Download CAS-ESM2-0 monthly CMIP6 data from the ORNL ESGF data node.

This version uses a verified static catalogue for all 12 files. It avoids the
intermittent ORNL ESGF search API and supports reliable HTTP range resuming.

Variables:
    tas, rsds, sfcWind

Experiments:
    historical, ssp126, ssp245, ssp585

Member, table and grid:
    r1i1p1f1, Amon, gn

No external Python packages are required.
"""

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

import argparse
import csv
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ============================================================
# USER CONFIGURATION
# ============================================================

OUTPUT_DIR = get_path("data_raw_cmip6_downloaded")

MODEL = "CAS-ESM2-0"
VARIANT = "r1i1p1f1"
TABLE_ID = "Amon"
GRID_LABEL = "gn"
DATA_NODE = "esgf-node.ornl.gov"

VARIABLES = ("tas", "rsds", "sfcWind")
EXPERIMENTS = ("historical", "ssp126", "ssp245", "ssp585")

DOWNLOAD_TIMEOUT = 300
RETRIES = 10
RETRY_SLEEP = 10
CHUNK_SIZE = 4 * 1024 * 1024

MANIFEST_CSV = OUTPUT_DIR / "cas_esm2_0_ornl_manifest_v2.csv"

# Scenario files have fixed byte sizes in the verified ORNL catalogue.
# These values allow incomplete files to be detected even if a HEAD request
# temporarily fails. Historical sizes are obtained from the server at runtime.
SCENARIO_FILE_SIZES = {
    "tas": 135_313_996,
    "rsds": 135_313_728,
    "sfcWind": 135_313_992,
}


# ============================================================
# STATIC FILE CATALOGUE
# ============================================================

@dataclass
class FileRecord:
    experiment: str
    variable: str
    filename: str
    version: str
    download_url: str
    local_path: str
    remote_size: int | None = None
    status: str = "pending"
    message: str = ""


def build_record(experiment: str, variable: str) -> FileRecord:
    if experiment == "historical":
        activity = "CMIP"
        version = "20201227"
        period = "185001-201412"
    else:
        activity = "ScenarioMIP"
        version = "20201228"
        period = "201501-210012"

    filename = (
        f"{variable}_{TABLE_ID}_{MODEL}_{experiment}_"
        f"{VARIANT}_{GRID_LABEL}_{period}.nc"
    )

    download_url = (
        f"https://{DATA_NODE}/thredds/fileServer/css03_data/CMIP6/"
        f"{activity}/CAS/{MODEL}/{experiment}/{VARIANT}/{TABLE_ID}/"
        f"{variable}/{GRID_LABEL}/v{version}/{filename}"
    )

    local_path = (
        OUTPUT_DIR
        / MODEL
        / experiment
        / VARIANT
        / TABLE_ID
        / variable
        / GRID_LABEL
        / filename
    )

    return FileRecord(
        experiment=experiment,
        variable=variable,
        filename=filename,
        version=version,
        download_url=download_url,
        local_path=str(local_path),
        remote_size=(
            None
            if experiment == "historical"
            else SCENARIO_FILE_SIZES[variable]
        ),
    )


def build_catalogue() -> list[FileRecord]:
    return [
        build_record(experiment, variable)
        for experiment in EXPERIMENTS
        for variable in VARIABLES
    ]


# ============================================================
# HTTP HELPERS
# ============================================================

def build_headers() -> dict[str, str]:
    return {
        "User-Agent": "CAS-ESM2-0-ORNL-Downloader/2.0",
        "Accept": "application/netcdf,application/octet-stream,*/*",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }


def parse_content_range(value: str | None) -> int | None:
    if not value:
        return None

    match = re.match(r"bytes\s+\d+-\d+/(\d+|\*)", value)

    if not match or match.group(1) == "*":
        return None

    return int(match.group(1))


def query_remote_size(url: str) -> int | None:
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            headers = build_headers()
            request = Request(url, headers=headers, method="HEAD")

            with urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                content_length = response.headers.get("Content-Length")

                if content_length and content_length.isdigit():
                    return int(content_length)

                return None

        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc

            if attempt < 3:
                time.sleep(3)

    if last_error is not None:
        print(f"  Remote size query unavailable: {last_error}")

    return None


# ============================================================
# FILE VALIDATION AND PARTIAL-FILE HANDLING
# ============================================================

def has_netcdf_signature(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 8:
        return False

    with path.open("rb") as file_object:
        header = file_object.read(8)

    return header.startswith(b"CDF") or header == b"\x89HDF\r\n\x1a\n"


def validate_complete_file(
    path: Path,
    remote_size: int | None,
) -> tuple[bool, str]:
    if not path.exists():
        return False, "File does not exist."

    local_size = path.stat().st_size

    if local_size <= 0:
        return False, "File is empty."

    if remote_size is not None and local_size != remote_size:
        return (
            False,
            f"Size mismatch: local={local_size}, remote={remote_size}",
        )

    if not has_netcdf_signature(path):
        return False, "NetCDF signature is invalid."

    return True, f"Validation passed ({local_size} bytes)."


def keep_largest_partial(destination: Path, part_path: Path) -> None:
    if not destination.exists():
        return

    destination_size = destination.stat().st_size
    part_size = part_path.stat().st_size if part_path.exists() else -1

    if destination_size > part_size:
        destination.replace(part_path)
        print(
            f"  Reusing incomplete file as partial data: "
            f"{destination_size} bytes"
        )
    else:
        destination.unlink()
        print(
            f"  Keeping larger partial file: {part_size} bytes"
        )


# ============================================================
# DOWNLOADER
# ============================================================

def download_file(record: FileRecord) -> None:
    destination = Path(record.local_path)
    part_path = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    queried_size = query_remote_size(record.download_url)

    if queried_size is not None:
        record.remote_size = queried_size

    if record.remote_size is not None:
        print(
            f"  Remote size: "
            f"{record.remote_size / 1024**2:.2f} MB "
            f"({record.remote_size} bytes)"
        )
    else:
        print("  Remote size: unavailable")

    valid, message = validate_complete_file(
        destination,
        record.remote_size,
    )

    if valid:
        record.status = "existing"
        record.message = message
        print(f"  SKIP: {destination.name}")
        return

    if destination.exists():
        print(f"  Existing file is incomplete: {message}")
        keep_largest_partial(destination, part_path)

    last_error: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        try:
            existing_size = (
                part_path.stat().st_size if part_path.exists() else 0
            )
            headers = build_headers()

            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"
                print(
                    f"  Resume attempt {attempt}/{RETRIES} from "
                    f"{existing_size} bytes"
                )
            else:
                print(f"  Download attempt {attempt}/{RETRIES}")

            request = Request(record.download_url, headers=headers)

            with urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                status_code = getattr(response, "status", response.getcode())
                response_length_text = response.headers.get("Content-Length")
                response_length = (
                    int(response_length_text)
                    if response_length_text and response_length_text.isdigit()
                    else None
                )
                range_total = parse_content_range(
                    response.headers.get("Content-Range")
                )

                if existing_size > 0 and status_code == 206:
                    mode = "ab"
                    expected_total = range_total or record.remote_size
                else:
                    mode = "wb"
                    existing_size = 0
                    expected_total = record.remote_size or response_length

                with part_path.open(mode) as file_object:
                    while True:
                        chunk = response.read(CHUNK_SIZE)

                        if not chunk:
                            break

                        file_object.write(chunk)

            current_size = part_path.stat().st_size

            if expected_total is not None and current_size != expected_total:
                raise RuntimeError(
                    f"Incomplete transfer: local={current_size}, "
                    f"remote={expected_total}"
                )

            if not has_netcdf_signature(part_path):
                raise RuntimeError("Downloaded file has an invalid NetCDF signature.")

            part_path.replace(destination)

            valid, message = validate_complete_file(
                destination,
                expected_total,
            )

            if not valid:
                keep_largest_partial(destination, part_path)
                raise RuntimeError(message)

            record.remote_size = expected_total
            record.status = "downloaded"
            record.message = message
            print(f"  DONE: {destination.name}")
            return

        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            partial_size = (
                part_path.stat().st_size if part_path.exists() else 0
            )
            print(
                f"  Download failed ({attempt}/{RETRIES}): {exc}; "
                f"partial={partial_size} bytes"
            )

            if attempt < RETRIES:
                time.sleep(RETRY_SLEEP)

    record.status = "failed"
    record.message = str(last_error)
    raise RuntimeError(
        f"Failed after {RETRIES} attempts: {last_error}"
    ) from last_error


# ============================================================
# MANIFEST AND MAIN PROGRAM
# ============================================================

def write_manifest(records: list[FileRecord]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(FileRecord.__dataclass_fields__)

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8-sig") as file_object:
        writer = csv.DictWriter(file_object, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the 12 CAS-ESM2-0 CMIP6 files from verified ORNL URLs."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the static catalogue without downloading files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    records = build_catalogue()

    print("=" * 100)
    print("CAS-ESM2-0 ORNL ESGF DOWNLOADER V2")
    print("=" * 100)
    print(f"Data node   : {DATA_NODE}")
    print(f"Model       : {MODEL}")
    print(f"Variant     : {VARIANT}")
    print(f"Table       : {TABLE_ID}")
    print(f"Grid        : {GRID_LABEL}")
    print(f"Output root : {OUTPUT_DIR}")
    print(f"Dry run     : {args.dry_run}")

    for index, record in enumerate(records, start=1):
        print()
        print("-" * 100)
        print(
            f"[{index}/{len(records)}] "
            f"{MODEL} {record.experiment} {record.variable}"
        )
        print("-" * 100)
        print(f"  FILE: {record.filename}")
        print(f"  URL : {record.download_url}")

        if args.dry_run:
            record.status = "dry_run"
            record.message = "Static catalogue entry available."
            continue

        try:
            download_file(record)
        except Exception as exc:
            record.status = "failed"
            record.message = str(exc)
            print(f"  ERROR: {exc}")

    write_manifest(records)

    downloaded = sum(record.status == "downloaded" for record in records)
    existing = sum(record.status == "existing" for record in records)
    dry_run = sum(record.status == "dry_run" for record in records)
    failed = sum(record.status == "failed" for record in records)

    print()
    print("=" * 100)
    print("DOWNLOAD SUMMARY")
    print("=" * 100)
    print(f"Expected files  : {len(records)}")
    print(f"Downloaded files: {downloaded}")
    print(f"Existing files  : {existing}")
    print(f"Dry-run files   : {dry_run}")
    print(f"Failed files    : {failed}")
    print(f"Manifest        : {MANIFEST_CSV}")

    if failed:
        print()
        print("FAILED FILES")

        for record in records:
            if record.status == "failed":
                print(
                    f"  {record.experiment:10s} "
                    f"{record.variable:8s} {record.message}"
                )

        return 1

    print()
    print("ALL 12 CAS-ESM2-0 FILES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
