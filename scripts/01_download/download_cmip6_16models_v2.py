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
import re
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


BASE_DIR = get_path("data_raw_cmip6_downloaded")

MANIFEST_CSV = (
    BASE_DIR
    / "cmip6_ceda_clean_manifest_16models.csv"
)

DOWNLOAD_ROOT = (
    BASE_DIR
    / "downloaded"
)

FAILED_CSV = (
    BASE_DIR
    / "cmip6_download_failed_v2.csv"
)

DOWNLOAD_HOST = "https://dap.ceda.ac.uk"

MAX_WORKERS = 4

REQUEST_TIMEOUT = 120

MAX_FILE_RETRIES = 100

RETRY_DELAY = 5

CHUNK_SIZE = 1024 * 1024

FILE_PRINT_INTERVAL = 10

GLOBAL_PRINT_INTERVAL = 5


print_lock = threading.Lock()
progress_lock = threading.Lock()


@dataclass
class ProgressItem:
    label: str
    downloaded: int = 0
    total: int | None = None
    session_downloaded: int = 0
    start_time: float = 0.0
    active: bool = False


progress_items = {}

global_stats = {
    "total_files": 0,
    "completed_files": 0,
    "failed_files": 0,
    "skipped_files": 0,
    "downloaded_this_run": 0,
    "start_time": time.time(),
}


def safe_print(*args, **kwargs):

    with print_lock:
        print(
            *args,
            **kwargs,
            flush=True,
        )


def data_to_dap_url(data_url):

    parsed = urlparse(
        data_url
    )

    return (
        f"{DOWNLOAD_HOST}"
        f"{parsed.path}"
    )


def split_matching_files(value):

    if not value:
        return []

    return [
        item.strip()
        for item in value.split(";")
        if item.strip()
    ]


def build_file_url(
    version_url,
    filename,
):

    data_url = (
        f"{version_url.rstrip('/')}/"
        f"{filename}"
    )

    return data_to_dap_url(
        data_url
    )


def build_output_path(
    row,
    filename,
):

    return (
        DOWNLOAD_ROOT
        / row["model"]
        / row["experiment"]
        / row["variable"]
        / filename
    )


def read_manifest():

    with MANIFEST_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


def build_jobs(rows):

    jobs = []

    for row in rows:

        filenames = split_matching_files(
            row["matching_files"]
        )

        for filename in filenames:

            jobs.append(
                {
                    "model": row["model"],
                    "experiment": row["experiment"],
                    "variable": row["variable"],
                    "variant": row["variant"],
                    "grid": row["grid"],
                    "version": row["version"],
                    "filename": filename,
                    "url": build_file_url(
                        row["url"],
                        filename,
                    ),
                    "output_path": build_output_path(
                        row,
                        filename,
                    ),
                }
            )

    return jobs


def make_label(job):

    return (
        f"{job['model']} | "
        f"{job['experiment']} | "
        f"{job['variable']}"
    )


def update_progress(
    job_id,
    downloaded=None,
    total=None,
    session_add=0,
    active=None,
):

    with progress_lock:

        item = progress_items[
            job_id
        ]

        if downloaded is not None:
            item.downloaded = downloaded

        if total is not None:
            item.total = total

        if session_add:
            item.session_downloaded += (
                session_add
            )

            global_stats[
                "downloaded_this_run"
            ] += session_add

        if active is not None:
            item.active = active


def mark_completed():

    with progress_lock:

        global_stats[
            "completed_files"
        ] += 1


def mark_failed():

    with progress_lock:

        global_stats[
            "failed_files"
        ] += 1


def mark_skipped():

    with progress_lock:

        global_stats[
            "skipped_files"
        ] += 1


def format_size(size):

    if size is None:
        return "?"

    mb = (
        size
        / 1024
        / 1024
    )

    if mb < 1024:
        return f"{mb:.1f} MB"

    gb = (
        mb
        / 1024
    )

    return f"{gb:.2f} GB"


def global_reporter(stop_event):

    while not stop_event.wait(
        GLOBAL_PRINT_INTERVAL
    ):

        with progress_lock:

            total_files = (
                global_stats[
                    "total_files"
                ]
            )

            completed = (
                global_stats[
                    "completed_files"
                ]
            )

            failed = (
                global_stats[
                    "failed_files"
                ]
            )

            skipped = (
                global_stats[
                    "skipped_files"
                ]
            )

            downloaded = (
                global_stats[
                    "downloaded_this_run"
                ]
            )

            elapsed = (
                time.time()
                - global_stats[
                    "start_time"
                ]
            )

            active_items = [
                item
                for item
                in progress_items.values()
                if item.active
            ]

        speed = (
            downloaded
            / elapsed
            if elapsed > 0
            else 0
        )

        finished = (
            completed
            + failed
            + skipped
        )

        safe_print()
        safe_print(
            "=" * 80
        )

        safe_print(
            f"GLOBAL | "
            f"finished={finished}/{total_files} | "
            f"completed={completed} | "
            f"skipped={skipped} | "
            f"failed={failed} | "
            f"run_download={format_size(downloaded)} | "
            f"avg_speed={speed / 1024 / 1024:.2f} MB/s"
        )

        for item in active_items:

            if item.total:

                percent = (
                    item.downloaded
                    / item.total
                    * 100
                )

                safe_print(
                    f"ACTIVE | "
                    f"{item.label} | "
                    f"{percent:6.2f}% | "
                    f"{format_size(item.downloaded)} / "
                    f"{format_size(item.total)}"
                )

            else:

                safe_print(
                    f"ACTIVE | "
                    f"{item.label} | "
                    f"{format_size(item.downloaded)}"
                )

        safe_print(
            "=" * 80
        )


def parse_total_size(
    response,
    existing_size,
):

    content_range = (
        response.headers.get(
            "Content-Range"
        )
    )

    if content_range:

        match = re.search(
            r"bytes\s+\d+-\d+/(\d+|\*)",
            content_range,
        )

        if (
            match
            and match.group(1) != "*"
        ):

            return int(
                match.group(1)
            )

    content_length = (
        response.headers.get(
            "Content-Length"
        )
    )

    if content_length:

        content_length = int(
            content_length
        )

        status_code = getattr(
            response,
            "status",
            None,
        )

        if status_code == 206:

            return (
                existing_size
                + content_length
            )

        return content_length

    return None


def download_file(
    job_id,
    job,
):

    url = job["url"]

    output_path = job[
        "output_path"
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    label = make_label(
        job
    )

    if (
        output_path.exists()
        and output_path.stat().st_size > 0
    ):

        safe_print(
            f"SKIP | "
            f"{label} | "
            f"{job['filename']}"
        )

        mark_skipped()

        return {
            "success": True,
            "skipped": True,
            **job,
        }

    temp_path = output_path.with_suffix(
        output_path.suffix
        + ".part"
    )

    with progress_lock:

        progress_items[
            job_id
        ] = ProgressItem(
            label=label,
            start_time=time.time(),
            active=True,
        )

    for attempt in range(
        1,
        MAX_FILE_RETRIES + 1,
    ):

        existing_size = 0

        if temp_path.exists():

            existing_size = (
                temp_path.stat().st_size
            )

        update_progress(
            job_id,
            downloaded=existing_size,
            active=True,
        )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64)"
            ),
            "Connection": "close",
        }

        if existing_size > 0:

            headers["Range"] = (
                f"bytes={existing_size}-"
            )

        try:

            if existing_size > 0:

                safe_print(
                    f"RESUME | "
                    f"{label} | "
                    f"attempt={attempt}/{MAX_FILE_RETRIES} | "
                    f"from={format_size(existing_size)} | "
                    f"{job['filename']}"
                )

            else:

                safe_print(
                    f"START | "
                    f"{label} | "
                    f"attempt={attempt}/{MAX_FILE_RETRIES} | "
                    f"{job['filename']}"
                )

            request = Request(
                url,
                headers=headers,
            )

            with urlopen(
                request,
                timeout=REQUEST_TIMEOUT,
            ) as response:

                status_code = getattr(
                    response,
                    "status",
                    None,
                )

                if (
                    existing_size > 0
                    and status_code != 206
                ):

                    safe_print(
                        f"RANGE_REJECTED | "
                        f"{label} | "
                        f"status={status_code}"
                    )

                    response.close()

                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

                    time.sleep(
                        1
                    )

                    continue

                total_size = parse_total_size(
                    response,
                    existing_size,
                )

                if total_size is not None:

                    update_progress(
                        job_id,
                        total=total_size,
                    )

                mode = (
                    "ab"
                    if existing_size > 0
                    else "wb"
                )

                downloaded = (
                    existing_size
                )

                session_downloaded = 0

                session_start = (
                    time.time()
                )

                last_file_print = (
                    session_start
                )

                with temp_path.open(
                    mode
                ) as file:

                    while True:

                        chunk = response.read(
                            CHUNK_SIZE
                        )

                        if not chunk:
                            break

                        file.write(
                            chunk
                        )

                        size = len(
                            chunk
                        )

                        downloaded += size

                        session_downloaded += (
                            size
                        )

                        update_progress(
                            job_id,
                            downloaded=downloaded,
                            session_add=size,
                        )

                        current_time = (
                            time.time()
                        )

                        if (
                            current_time
                            - last_file_print
                            >= FILE_PRINT_INTERVAL
                        ):

                            elapsed = (
                                current_time
                                - session_start
                            )

                            speed = (
                                session_downloaded
                                / elapsed
                                if elapsed > 0
                                else 0
                            )

                            if total_size:

                                percent = (
                                    downloaded
                                    / total_size
                                    * 100
                                )

                                remaining = (
                                    total_size
                                    - downloaded
                                )

                                eta_seconds = (
                                    remaining
                                    / speed
                                    if speed > 0
                                    else 0
                                )

                                safe_print(
                                    f"PROGRESS | "
                                    f"{label} | "
                                    f"{percent:6.2f}% | "
                                    f"{format_size(downloaded)} / "
                                    f"{format_size(total_size)} | "
                                    f"{speed / 1024 / 1024:.2f} MB/s | "
                                    f"ETA={eta_seconds / 60:.1f} min"
                                )

                            else:

                                safe_print(
                                    f"PROGRESS | "
                                    f"{label} | "
                                    f"{format_size(downloaded)} | "
                                    f"{speed / 1024 / 1024:.2f} MB/s"
                                )

                            last_file_print = (
                                current_time
                            )

            final_part_size = (
                temp_path.stat().st_size
            )

            if total_size is not None:

                if (
                    final_part_size
                    < total_size
                ):

                    safe_print(
                        f"CONNECTION_ENDED | "
                        f"{label} | "
                        f"{format_size(final_part_size)} / "
                        f"{format_size(total_size)}"
                    )

                    if (
                        attempt
                        < MAX_FILE_RETRIES
                    ):

                        time.sleep(
                            RETRY_DELAY
                        )

                    continue

                if (
                    final_part_size
                    > total_size
                ):

                    safe_print(
                        f"SIZE_ERROR | "
                        f"{label} | "
                        f"{final_part_size} > "
                        f"{total_size}"
                    )

                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

                    time.sleep(
                        RETRY_DELAY
                    )

                    continue

            temp_path.replace(
                output_path
            )

            update_progress(
                job_id,
                downloaded=(
                    output_path
                    .stat()
                    .st_size
                ),
                active=False,
            )

            mark_completed()

            safe_print(
                f"DONE | "
                f"{label} | "
                f"{format_size(output_path.stat().st_size)} | "
                f"{job['filename']}"
            )

            return {
                "success": True,
                "skipped": False,
                **job,
            }

        except HTTPError as exc:

            if exc.code == 404:

                safe_print(
                    f"HTTP_404 | "
                    f"{label} | "
                    f"{url}"
                )

                break

            if exc.code == 416:

                safe_print(
                    f"HTTP_416 | "
                    f"{label} | "
                    f"partial={format_size(existing_size)}"
                )

                try:
                    temp_path.unlink()
                except OSError:
                    pass

            else:

                safe_print(
                    f"HTTP_ERROR | "
                    f"{label} | "
                    f"code={exc.code} | "
                    f"attempt={attempt}"
                )

        except (
            IncompleteRead,
            RemoteDisconnected,
            ssl.SSLError,
            URLError,
            TimeoutError,
            ConnectionError,
        ) as exc:

            partial_size = 0

            if temp_path.exists():

                partial_size = (
                    temp_path
                    .stat()
                    .st_size
                )

            safe_print(
                f"INTERRUPTED | "
                f"{label} | "
                f"attempt={attempt}/{MAX_FILE_RETRIES} | "
                f"partial={format_size(partial_size)} | "
                f"{type(exc).__name__}: {exc}"
            )

        except Exception as exc:

            partial_size = 0

            if temp_path.exists():

                partial_size = (
                    temp_path
                    .stat()
                    .st_size
                )

            safe_print(
                f"ERROR | "
                f"{label} | "
                f"attempt={attempt}/{MAX_FILE_RETRIES} | "
                f"partial={format_size(partial_size)} | "
                f"{type(exc).__name__}: {exc}"
            )

        if (
            attempt
            < MAX_FILE_RETRIES
        ):

            time.sleep(
                RETRY_DELAY
            )

    update_progress(
        job_id,
        active=False,
    )

    mark_failed()

    safe_print(
        f"FAILED | "
        f"{label} | "
        f"{job['filename']}"
    )

    return {
        "success": False,
        "skipped": False,
        **job,
    }


def write_failed_csv(
    failed_results,
):

    if not failed_results:

        if FAILED_CSV.exists():

            try:
                FAILED_CSV.unlink()
            except OSError:
                pass

        return

    fieldnames = [
        "model",
        "experiment",
        "variable",
        "variant",
        "grid",
        "version",
        "filename",
        "url",
    ]

    with FAILED_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for item in failed_results:

            writer.writerow(
                {
                    key: item[key]
                    for key in fieldnames
                }
            )


def main():

    rows = read_manifest()

    jobs = build_jobs(
        rows
    )

    with progress_lock:

        global_stats[
            "total_files"
        ] = len(
            jobs
        )

        global_stats[
            "start_time"
        ] = time.time()

    safe_print(
        "=" * 80
    )

    safe_print(
        "CMIP6 16-MODEL PARALLEL DOWNLOADER V2"
    )

    safe_print(
        "=" * 80
    )

    safe_print(
        f"Manifest rows: "
        f"{len(rows)}"
    )

    safe_print(
        f"Files: "
        f"{len(jobs)}"
    )

    safe_print(
        f"Workers: "
        f"{MAX_WORKERS}"
    )

    safe_print(
        f"Retries per file: "
        f"{MAX_FILE_RETRIES}"
    )

    safe_print(
        f"Download root: "
        f"{DOWNLOAD_ROOT}"
    )

    safe_print(
        "=" * 80
    )

    stop_event = (
        threading.Event()
    )

    reporter_thread = (
        threading.Thread(
            target=global_reporter,
            args=(
                stop_event,
            ),
            daemon=True,
        )
    )

    reporter_thread.start()

    failed_results = []

    try:

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            future_map = {}

            for job_id, job in enumerate(
                jobs,
                start=1,
            ):

                future = executor.submit(
                    download_file,
                    job_id,
                    job,
                )

                future_map[
                    future
                ] = job_id

            for future in as_completed(
                future_map
            ):

                result = future.result()

                if not result[
                    "success"
                ]:

                    failed_results.append(
                        result
                    )

                    write_failed_csv(
                        failed_results
                    )

    except KeyboardInterrupt:

        safe_print()
        safe_print(
            "Keyboard interrupt received."
        )

        safe_print(
            "Partial files are preserved."
        )

        safe_print(
            "Run the script again to resume."
        )

    finally:

        stop_event.set()

        reporter_thread.join(
            timeout=2
        )

    write_failed_csv(
        failed_results
    )

    with progress_lock:

        total_files = (
            global_stats[
                "total_files"
            ]
        )

        completed = (
            global_stats[
                "completed_files"
            ]
        )

        skipped = (
            global_stats[
                "skipped_files"
            ]
        )

        failed = (
            global_stats[
                "failed_files"
            ]
        )

        downloaded = (
            global_stats[
                "downloaded_this_run"
            ]
        )

        elapsed = (
            time.time()
            - global_stats[
                "start_time"
            ]
        )

    safe_print()
    safe_print(
        "=" * 80
    )

    safe_print(
        "DOWNLOAD SESSION FINISHED"
    )

    safe_print(
        "=" * 80
    )

    safe_print(
        f"Total files: "
        f"{total_files}"
    )

    safe_print(
        f"Completed: "
        f"{completed}"
    )

    safe_print(
        f"Skipped: "
        f"{skipped}"
    )

    safe_print(
        f"Failed: "
        f"{failed}"
    )

    safe_print(
        f"Downloaded this run: "
        f"{format_size(downloaded)}"
    )

    safe_print(
        f"Elapsed: "
        f"{elapsed / 3600:.2f} hours"
    )

    if elapsed > 0:

        safe_print(
            f"Average aggregate speed: "
            f"{downloaded / elapsed / 1024 / 1024:.2f} MB/s"
        )

    if failed_results:

        safe_print(
            f"Failed list: "
            f"{FAILED_CSV}"
        )


if __name__ == "__main__":

    main()