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
import time
from html.parser import HTMLParser
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen


BASE_URL = "https://data.ceda.ac.uk/badc/cmip6/data/CMIP6"

OUTPUT_DIR = get_path("data_raw_cmip6_downloaded")

OUTPUT_CSV = OUTPUT_DIR / "cmip6_ceda_missing_recheck.csv"

REQUEST_TIMEOUT = 40
MAX_RETRIES = 5
RETRY_DELAY = 3

TABLE_ID = "Amon"
VARIABLE = "sfcWind"
VARIANT = "r1i1p1f1"


TARGETS = [
    {
        "model": "CAS-ESM2-0",
        "institution": "CAS",
        "activity": "CMIP",
        "experiment": "historical",
        "start_year": 1994,
        "end_year": 2014,
    },
    {
        "model": "CAS-ESM2-0",
        "institution": "CAS",
        "activity": "ScenarioMIP",
        "experiment": "ssp126",
        "start_year": 2040,
        "end_year": 2100,
    },
    {
        "model": "CAS-ESM2-0",
        "institution": "CAS",
        "activity": "ScenarioMIP",
        "experiment": "ssp245",
        "start_year": 2040,
        "end_year": 2100,
    },
    {
        "model": "CAS-ESM2-0",
        "institution": "CAS",
        "activity": "ScenarioMIP",
        "experiment": "ssp585",
        "start_year": 2040,
        "end_year": 2100,
    },
    {
        "model": "TaiESM1",
        "institution": "AS-RCEC",
        "activity": "ScenarioMIP",
        "experiment": "ssp126",
        "start_year": 2040,
        "end_year": 2100,
    },
]


class LinkParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):

        if tag.lower() != "a":
            return

        for key, value in attrs:

            if key.lower() == "href" and value:
                self.hrefs.append(value)
                break


def normalize_url(url):

    parsed = urlparse(url)

    path = parsed.path

    if not path.endswith("/"):
        path += "/"

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            "",
            "",
            "",
        )
    )


def get_name(url):

    return urlparse(
        url
    ).path.rstrip("/").split("/")[-1]


def request_page(url):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140.0 Safari/537.36"
        ),
        "Connection": "close",
    }

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            request = Request(
                url,
                headers=headers,
            )

            with urlopen(
                request,
                timeout=REQUEST_TIMEOUT,
            ) as response:

                content = response.read()

                return content.decode(
                    "utf-8",
                    errors="replace",
                )

        except HTTPError as exc:

            if exc.code == 404:

                print(
                    f"404: {url}"
                )

                return None

            if 500 <= exc.code <= 599:

                print(
                    f"Server error {exc.code} "
                    f"({attempt}/{MAX_RETRIES}): "
                    f"{url}"
                )

                if attempt < MAX_RETRIES:
                    time.sleep(
                        RETRY_DELAY
                    )

                continue

            print(
                f"HTTP error {exc.code}: "
                f"{url}"
            )

            return None

        except IncompleteRead as exc:

            print(
                f"Incomplete response "
                f"({attempt}/{MAX_RETRIES}): "
                f"{url}"
            )

            print(
                f"Received: "
                f"{len(exc.partial)} bytes"
            )

            print(
                f"Missing: "
                f"{exc.expected} bytes"
            )

        except RemoteDisconnected as exc:

            print(
                f"Remote disconnected "
                f"({attempt}/{MAX_RETRIES}): "
                f"{url}"
            )

            print(
                f"Reason: {exc}"
            )

        except ssl.SSLError as exc:

            print(
                f"SSL error "
                f"({attempt}/{MAX_RETRIES}): "
                f"{url}"
            )

            print(
                f"Reason: {exc}"
            )

        except (
            URLError,
            TimeoutError,
            ConnectionError,
        ) as exc:

            print(
                f"Network error "
                f"({attempt}/{MAX_RETRIES}): "
                f"{url}"
            )

            print(
                f"Reason: {exc}"
            )

        except Exception as exc:

            print(
                f"Unexpected error "
                f"({attempt}/{MAX_RETRIES}): "
                f"{url}"
            )

            print(
                f"Type: "
                f"{type(exc).__name__}"
            )

            print(
                f"Reason: {exc}"
            )

        if attempt < MAX_RETRIES:

            time.sleep(
                RETRY_DELAY
            )

    print(
        f"Failed after "
        f"{MAX_RETRIES} attempts: "
        f"{url}"
    )

    return None


def list_directory(url):

    url = normalize_url(
        url
    )

    html = request_page(
        url
    )

    if html is None:
        return None

    parser = LinkParser()
    parser.feed(
        html
    )

    base = urlparse(
        url
    )

    base_path = (
        base.path
    )

    directories = set()
    files = set()

    for href in parser.hrefs:

        if href.startswith("#"):
            continue

        candidate = urljoin(
            url,
            href,
        )

        parsed = urlparse(
            candidate
        )

        if (
            parsed.netloc.lower()
            != "data.ceda.ac.uk"
        ):
            continue

        if not parsed.path.startswith(
            base_path
        ):
            continue

        relative = parsed.path[
            len(base_path):
        ].strip("/")

        if not relative:
            continue

        if "/" in relative:
            continue

        clean = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                "",
                "",
                "",
            )
        )

        if relative.endswith(
            ".nc"
        ):

            files.add(
                clean
            )

        else:

            directories.add(
                normalize_url(
                    clean
                )
            )

    return {
        "directories": sorted(
            directories
        ),
        "files": sorted(
            files
        ),
    }


def extract_file_years(
    filename,
):

    match = re.search(
        r"_(\d{4})(?:\d{2})?"
        r"-(\d{4})(?:\d{2})?"
        r"\.nc$",
        filename,
    )

    if match is None:
        return None

    return (
        int(
            match.group(1)
        ),
        int(
            match.group(2)
        ),
    )


def overlaps_period(
    filename,
    target_start,
    target_end,
):

    years = extract_file_years(
        filename
    )

    if years is None:
        return False

    file_start, file_end = years

    return (
        file_end >= target_start
        and
        file_start <= target_end
    )


def find_real_versions(
    grid_url,
):

    listing = list_directory(
        grid_url
    )

    if listing is None:
        return []

    versions = []

    for directory_url in (
        listing["directories"]
    ):

        version = get_name(
            directory_url
        )

        if re.fullmatch(
            r"v\d{8}",
            version,
        ):

            versions.append(
                directory_url
            )

    return sorted(
        versions,
        key=get_name,
    )


def recheck_target(
    target,
):

    model = (
        target["model"]
    )

    institution = (
        target["institution"]
    )

    activity = (
        target["activity"]
    )

    experiment = (
        target["experiment"]
    )

    target_start = (
        target["start_year"]
    )

    target_end = (
        target["end_year"]
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"{model} | "
        f"{experiment} | "
        f"{VARIABLE}"
    )

    print(
        "=" * 80
    )

    variant_url = (
        f"{BASE_URL}/"
        f"{activity}/"
        f"{institution}/"
        f"{model}/"
        f"{experiment}/"
        f"{VARIANT}/"
    )

    print(
        f"Variant URL: "
        f"{variant_url}"
    )

    variant_listing = (
        list_directory(
            variant_url
        )
    )

    if variant_listing is None:

        return [{
            "model": model,
            "institution": institution,
            "activity": activity,
            "experiment": experiment,
            "variant": VARIANT,
            "table_id": TABLE_ID,
            "variable": VARIABLE,
            "grid": "",
            "version": "",
            "status": "VARIANT_NOT_ACCESSIBLE",
            "file_count": 0,
            "matching_file_count": 0,
            "matching_files": "",
            "url": variant_url,
        }]

    table_names = {
        get_name(url): url
        for url
        in variant_listing[
            "directories"
        ]
    }

    print(
        "Variant children:"
    )

    for name in sorted(
        table_names
    ):

        print(
            f"  {name}"
        )

    if TABLE_ID not in table_names:

        return [{
            "model": model,
            "institution": institution,
            "activity": activity,
            "experiment": experiment,
            "variant": VARIANT,
            "table_id": TABLE_ID,
            "variable": VARIABLE,
            "grid": "",
            "version": "",
            "status": "AMON_NOT_FOUND",
            "file_count": 0,
            "matching_file_count": 0,
            "matching_files": "",
            "url": variant_url,
        }]

    amon_url = (
        table_names[
            TABLE_ID
        ]
    )

    amon_listing = (
        list_directory(
            amon_url
        )
    )

    if amon_listing is None:

        return [{
            "model": model,
            "institution": institution,
            "activity": activity,
            "experiment": experiment,
            "variant": VARIANT,
            "table_id": TABLE_ID,
            "variable": VARIABLE,
            "grid": "",
            "version": "",
            "status": "AMON_NOT_ACCESSIBLE",
            "file_count": 0,
            "matching_file_count": 0,
            "matching_files": "",
            "url": amon_url,
        }]

    variable_map = {
        get_name(url): url
        for url
        in amon_listing[
            "directories"
        ]
    }

    print(
        "Amon variables:"
    )

    for name in sorted(
        variable_map
    ):

        print(
            f"  {name}"
        )

    if VARIABLE not in variable_map:

        return [{
            "model": model,
            "institution": institution,
            "activity": activity,
            "experiment": experiment,
            "variant": VARIANT,
            "table_id": TABLE_ID,
            "variable": VARIABLE,
            "grid": "",
            "version": "",
            "status": "VARIABLE_CONFIRMED_MISSING",
            "file_count": 0,
            "matching_file_count": 0,
            "matching_files": "",
            "url": amon_url,
        }]

    variable_url = (
        variable_map[
            VARIABLE
        ]
    )

    variable_listing = (
        list_directory(
            variable_url
        )
    )

    if variable_listing is None:

        return [{
            "model": model,
            "institution": institution,
            "activity": activity,
            "experiment": experiment,
            "variant": VARIANT,
            "table_id": TABLE_ID,
            "variable": VARIABLE,
            "grid": "",
            "version": "",
            "status": "VARIABLE_NOT_ACCESSIBLE",
            "file_count": 0,
            "matching_file_count": 0,
            "matching_files": "",
            "url": variable_url,
        }]

    grid_urls = (
        variable_listing[
            "directories"
        ]
    )

    print(
        "Grids:"
    )

    for grid_url in grid_urls:

        print(
            f"  {get_name(grid_url)}"
        )

    rows = []

    if not grid_urls:

        rows.append(
            {
                "model": model,
                "institution": institution,
                "activity": activity,
                "experiment": experiment,
                "variant": VARIANT,
                "table_id": TABLE_ID,
                "variable": VARIABLE,
                "grid": "",
                "version": "",
                "status": "NO_GRID_FOUND",
                "file_count": 0,
                "matching_file_count": 0,
                "matching_files": "",
                "url": variable_url,
            }
        )

        return rows

    for grid_url in grid_urls:

        grid = get_name(
            grid_url
        )

        versions = (
            find_real_versions(
                grid_url
            )
        )

        print(
            f"Grid {grid}: "
            f"{len(versions)} real versions"
        )

        if not versions:

            rows.append(
                {
                    "model": model,
                    "institution": institution,
                    "activity": activity,
                    "experiment": experiment,
                    "variant": VARIANT,
                    "table_id": TABLE_ID,
                    "variable": VARIABLE,
                    "grid": grid,
                    "version": "",
                    "status": "NO_REAL_VERSION",
                    "file_count": 0,
                    "matching_file_count": 0,
                    "matching_files": "",
                    "url": grid_url,
                }
            )

            continue

        latest_version_url = (
            versions[-1]
        )

        version = get_name(
            latest_version_url
        )

        print(
            f"Selected version: "
            f"{version}"
        )

        version_listing = (
            list_directory(
                latest_version_url
            )
        )

        if version_listing is None:

            rows.append(
                {
                    "model": model,
                    "institution": institution,
                    "activity": activity,
                    "experiment": experiment,
                    "variant": VARIANT,
                    "table_id": TABLE_ID,
                    "variable": VARIABLE,
                    "grid": grid,
                    "version": version,
                    "status": "VERSION_NOT_ACCESSIBLE",
                    "file_count": 0,
                    "matching_file_count": 0,
                    "matching_files": "",
                    "url": latest_version_url,
                }
            )

            continue

        nc_files = (
            version_listing[
                "files"
            ]
        )

        matching_files = []

        for file_url in nc_files:

            filename = get_name(
                file_url
            )

            if overlaps_period(
                filename,
                target_start,
                target_end,
            ):

                matching_files.append(
                    filename
                )

        print(
            f"NetCDF files: "
            f"{len(nc_files)}"
        )

        print(
            f"Matching files: "
            f"{len(matching_files)}"
        )

        for filename in (
            matching_files
        ):

            print(
                f"  {filename}"
            )

        status = "FOUND"

        if not matching_files:

            status = (
                "NO_TARGET_FILES"
            )

        rows.append(
            {
                "model": model,
                "institution": institution,
                "activity": activity,
                "experiment": experiment,
                "variant": VARIANT,
                "table_id": TABLE_ID,
                "variable": VARIABLE,
                "grid": grid,
                "version": version,
                "status": status,
                "file_count": len(
                    nc_files
                ),
                "matching_file_count": len(
                    matching_files
                ),
                "matching_files": ";".join(
                    matching_files
                ),
                "url": latest_version_url,
            }
        )

    return rows


def write_results(
    rows,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "institution",
        "activity",
        "experiment",
        "variant",
        "table_id",
        "variable",
        "grid",
        "version",
        "status",
        "file_count",
        "matching_file_count",
        "matching_files",
        "url",
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


def main():

    print(
        "=" * 80
    )

    print(
        "CMIP6 CEDA MISSING DATA RECHECK"
    )

    print(
        "=" * 80
    )

    print(
        f"Targets: "
        f"{len(TARGETS)}"
    )

    print(
        "No NetCDF files will be downloaded."
    )

    all_rows = []

    for index, target in enumerate(
        TARGETS,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(TARGETS)}]"
        )

        rows = recheck_target(
            target
        )

        all_rows.extend(
            rows
        )

        write_results(
            all_rows
        )

    print()
    print(
        "=" * 80
    )

    print(
        "RECHECK FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"CSV: "
        f"{OUTPUT_CSV}"
    )

    print()

    for row in all_rows:

        print(
            f"{row['model']} | "
            f"{row['experiment']} | "
            f"{row['variable']} | "
            f"{row['grid']} | "
            f"{row['version']} | "
            f"{row['status']} | "
            f"matching="
            f"{row['matching_file_count']}"
        )


if __name__ == "__main__":

    main()