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
from html.parser import HTMLParser
import re
import time

import requests


DISCOVERY_HOST = "https://data.ceda.ac.uk"

DOWNLOAD_HOST = "https://dap.ceda.ac.uk"

CEDA_ROOT = "/badc/cmip6/data/CMIP6"

OUTPUT_ROOT = get_path("data_raw_cmip6_downloaded")

TABLE_ID = "Amon"

VARIANT = "r1i1p1f1"

GRID = "gn"

VARIABLES = [
    "tas",
    "sfcWind",
    "rsds",
]

TARGETS = {
    "AWI-CM-1-1-MR": {
        "historical": {
            "activity": "CMIP",
            "institution": "AWI",
            "start": 195901,
            "end": 199312,
        },
        "ssp126": {
            "activity": "ScenarioMIP",
            "institution": "AWI",
            "start": 201501,
            "end": 203912,
        },
        "ssp245": {
            "activity": "ScenarioMIP",
            "institution": "AWI",
            "start": 201501,
            "end": 203912,
        },
        "ssp585": {
            "activity": "ScenarioMIP",
            "institution": "AWI",
            "start": 201501,
            "end": 203912,
        },
    },
    "MPI-ESM1-2-HR": {
        "historical": {
            "activity": "CMIP",
            "institution": "MPI-M",
            "start": 195901,
            "end": 198912,
        },
        "ssp126": {
            "activity": "ScenarioMIP",
            "institution": "DKRZ",
            "start": 201501,
            "end": 203912,
        },
        "ssp245": {
            "activity": "ScenarioMIP",
            "institution": "DKRZ",
            "start": 201501,
            "end": 203912,
        },
        "ssp585": {
            "activity": "ScenarioMIP",
            "institution": "DKRZ",
            "start": 201501,
            "end": 203912,
        },
    },
    "MPI-ESM1-2-LR": {
        "historical": {
            "activity": "CMIP",
            "institution": "MPI-M",
            "start": 195901,
            "end": 198912,
        },
        "ssp126": {
            "activity": "ScenarioMIP",
            "institution": "MPI-M",
            "start": 201501,
            "end": 203412,
        },
        "ssp245": {
            "activity": "ScenarioMIP",
            "institution": "MPI-M",
            "start": 201501,
            "end": 203412,
        },
        "ssp585": {
            "activity": "ScenarioMIP",
            "institution": "MPI-M",
            "start": 201501,
            "end": 203412,
        },
    },
}

REQUEST_TIMEOUT = 120

MAX_RETRIES = 5

CHUNK_SIZE = 1024 * 1024

RETRY_DELAY = 5


class LinkParser(HTMLParser):

    def __init__(self):

        super().__init__()

        self.links = []

    def handle_starttag(
        self,
        tag,
        attrs,
    ):

        if tag != "a":

            return

        for key, value in attrs:

            if key == "href":

                self.links.append(
                    value
                )


def get_links(
    session,
    url,
):

    response = session.get(
        url,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    parser = LinkParser()

    parser.feed(
        response.text
    )

    return parser.links


def normalize_directory_name(
    href,
):

    if not href:

        return None

    href = href.split(
        "?"
    )[
        0
    ]

    href = href.split(
        "#"
    )[
        0
    ]

    href = href.rstrip(
        "/"
    )

    if not href:

        return None

    name = href.split(
        "/"
    )[
        -1
    ]

    return name


def discover_latest_version(
    session,
    model,
    experiment,
    variable,
    config,
):

    activity = config[
        "activity"
    ]

    institution = config[
        "institution"
    ]

    relative_path = (
        f"{CEDA_ROOT}/"
        f"{activity}/"
        f"{institution}/"
        f"{model}/"
        f"{experiment}/"
        f"{VARIANT}/"
        f"{TABLE_ID}/"
        f"{variable}/"
        f"{GRID}/"
    )

    url = (
        DISCOVERY_HOST
        +
        relative_path
    )

    links = get_links(
        session,
        url,
    )

    versions = []

    for href in links:

        name = normalize_directory_name(
            href
        )

        if name is None:

            continue

        if re.fullmatch(
            r"v\d{8}",
            name,
        ):

            versions.append(
                name
            )

    if not versions:

        raise RuntimeError(
            f"No version directory found: {url}"
        )

    versions = sorted(
        set(
            versions
        )
    )

    version = versions[
        -1
    ]

    return (
        version,
        relative_path
        +
        version
        +
        "/",
    )


def parse_file_period(
    filename,
):

    match = re.search(
        r"_(\d{6})-(\d{6})\.nc$",
        filename,
    )

    if match is None:

        return None

    start = int(
        match.group(
            1
        )
    )

    end = int(
        match.group(
            2
        )
    )

    return (
        start,
        end,
    )


def periods_overlap(
    file_start,
    file_end,
    target_start,
    target_end,
):

    return (
        file_end >= target_start
        and
        file_start <= target_end
    )


def discover_target_files(
    session,
    model,
    experiment,
    variable,
    config,
):

    version, version_path = discover_latest_version(
        session,
        model,
        experiment,
        variable,
        config,
    )

    directory_url = (
        DISCOVERY_HOST
        +
        version_path
    )

    links = get_links(
        session,
        directory_url,
    )

    target_start = config[
        "start"
    ]

    target_end = config[
        "end"
    ]

    prefix = (
        f"{variable}_"
        f"{TABLE_ID}_"
        f"{model}_"
        f"{experiment}_"
        f"{VARIANT}_"
        f"{GRID}_"
    )

    files = []

    for href in links:

        filename = normalize_directory_name(
            href
        )

        if filename is None:

            continue

        if not filename.startswith(
            prefix
        ):

            continue

        if not filename.endswith(
            ".nc"
        ):

            continue

        period = parse_file_period(
            filename
        )

        if period is None:

            continue

        file_start, file_end = period

        if periods_overlap(
            file_start,
            file_end,
            target_start,
            target_end,
        ):

            files.append(
                {
                    "filename": filename,
                    "file_start": file_start,
                    "file_end": file_end,
                    "version": version,
                    "version_path": version_path,
                }
            )

    files.sort(
        key=lambda item: (
            item[
                "file_start"
            ],
            item[
                "file_end"
            ],
        )
    )

    if not files:

        raise RuntimeError(
            f"No matching files found for "
            f"{model} | {experiment} | {variable} | "
            f"{target_start}-{target_end}"
        )

    return files


def make_local_path(
    model,
    experiment,
    variable,
    filename,
):

    directory = (
        OUTPUT_ROOT
        / model
        / experiment
        / VARIANT
        / TABLE_ID
        / variable
        / GRID
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        directory
        /
        filename
    )


def download_file(
    session,
    url,
    local_path,
):

    if (
        local_path.exists()
        and
        local_path.stat().st_size > 0
    ):

        print(
            f"SKIP existing: {local_path.name}"
        )

        return "SKIPPED"

    part_path = Path(
        str(
            local_path
        )
        +
        ".part"
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            existing_size = 0

            headers = {}

            mode = "wb"

            if part_path.exists():

                existing_size = (
                    part_path.stat().st_size
                )

                if existing_size > 0:

                    headers[
                        "Range"
                    ] = (
                        f"bytes={existing_size}-"
                    )

                    mode = "ab"

            with session.get(
                url,
                headers=headers,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            ) as response:

                if (
                    existing_size > 0
                    and
                    response.status_code == 200
                ):

                    existing_size = 0

                    mode = "wb"

                elif (
                    existing_size > 0
                    and
                    response.status_code != 206
                ):

                    response.raise_for_status()

                else:

                    response.raise_for_status()

                with open(
                    part_path,
                    mode,
                ) as file_handle:

                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):

                        if chunk:

                            file_handle.write(
                                chunk
                            )

            part_path.replace(
                local_path
            )

            print(
                f"DONE: {local_path.name}"
            )

            return "DOWNLOADED"

        except Exception as exc:

            print(
                f"Attempt {attempt}/{MAX_RETRIES} failed: "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt == MAX_RETRIES:

                raise

            time.sleep(
                RETRY_DELAY
                *
                attempt
            )

    return "FAILED"


def verify_discovered_coverage(
    files,
    target_start,
    target_end,
):

    if not files:

        raise ValueError(
            "No files were discovered"
        )

    ordered = sorted(
        files,
        key=lambda item: item[
            "file_start"
        ],
    )

    first_start = ordered[
        0
    ][
        "file_start"
    ]

    last_end = max(
        item[
            "file_end"
        ]
        for item in ordered
    )

    if first_start > target_start:

        raise ValueError(
            f"Discovered data start at {first_start}, "
            f"but target starts at {target_start}"
        )

    if last_end < target_end:

        raise ValueError(
            f"Discovered data end at {last_end}, "
            f"but target ends at {target_end}"
        )


def main():

    print()
    print(
        "=" * 110
    )

    print(
        "CMIP6 MISSING DATA DOWNLOADER"
    )

    print(
        "=" * 110
    )

    print(
        f"Output root: {OUTPUT_ROOT}"
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "CMIP6-missing-data-downloader"
            )
        }
    )

    discovered_count = 0
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    failed_items = []

    try:

        for model, experiments in TARGETS.items():

            print()
            print(
                "#" * 110
            )

            print(
                model
            )

            print(
                "#" * 110
            )

            for experiment, config in experiments.items():

                for variable in VARIABLES:

                    print()
                    print(
                        "-" * 110
                    )

                    print(
                        f"{model} | "
                        f"{experiment} | "
                        f"{variable}"
                    )

                    print(
                        f"Target: "
                        f"{config['start']} "
                        f"to "
                        f"{config['end']}"
                    )

                    print(
                        "-" * 110
                    )

                    try:

                        files = discover_target_files(
                            session,
                            model,
                            experiment,
                            variable,
                            config,
                        )

                        verify_discovered_coverage(
                            files,
                            config[
                                "start"
                            ],
                            config[
                                "end"
                            ],
                        )

                        print(
                            f"Version: "
                            f"{files[0]['version']}"
                        )

                        print(
                            f"Matching files: "
                            f"{len(files)}"
                        )

                        for item in files:

                            print(
                                f"  "
                                f"{item['filename']} "
                                f"["
                                f"{item['file_start']}-"
                                f"{item['file_end']}"
                                f"]"
                            )

                        discovered_count += len(
                            files
                        )

                        for item in files:

                            filename = item[
                                "filename"
                            ]

                            download_url = (
                                DOWNLOAD_HOST
                                +
                                item[
                                    "version_path"
                                ]
                                +
                                filename
                            )

                            local_path = make_local_path(
                                model,
                                experiment,
                                variable,
                                filename,
                            )

                            print()
                            print(
                                f"Downloading: {filename}"
                            )

                            status = download_file(
                                session,
                                download_url,
                                local_path,
                            )

                            if status == "DOWNLOADED":

                                downloaded_count += 1

                            elif status == "SKIPPED":

                                skipped_count += 1

                    except Exception as exc:

                        failed_count += 1

                        failed_items.append(
                            (
                                model,
                                experiment,
                                variable,
                                f"{type(exc).__name__}: {exc}",
                            )
                        )

                        print(
                            f"FAILED: "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )

    finally:

        session.close()

    print()
    print(
        "=" * 110
    )

    print(
        "DOWNLOAD SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        f"Discovered files: {discovered_count}"
    )

    print(
        f"Downloaded files: {downloaded_count}"
    )

    print(
        f"Skipped files: {skipped_count}"
    )

    print(
        f"Failed groups: {failed_count}"
    )

    if failed_items:

        print()
        print(
            "FAILED GROUPS"
        )

        for (
            model,
            experiment,
            variable,
            error,
        ) in failed_items:

            print(
                f"{model} | "
                f"{experiment} | "
                f"{variable} | "
                f"{error}"
            )

    else:

        print()
        print(
            "ALL MISSING CMIP6 DOWNLOADS COMPLETED"
        )


if __name__ == "__main__":
    main()