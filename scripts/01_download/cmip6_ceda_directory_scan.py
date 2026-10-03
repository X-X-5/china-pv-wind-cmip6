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
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from http.client import IncompleteRead, RemoteDisconnected
import ssl
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen


BASE_URL = "https://data.ceda.ac.uk/badc/cmip6/data/CMIP6"

OUTPUT_DIR = get_path("data_raw_cmip6_downloaded")

CSV_FILE = OUTPUT_DIR / "cmip6_ceda_directory_scan_v2.csv"
TXT_FILE = OUTPUT_DIR / "cmip6_ceda_directory_scan_v2.txt"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2

TABLE_ID = "Amon"

VARIABLES = [
    "rsds",
    "sfcWind",
    "tas",
]

EXPERIMENTS = {
    "historical": {
        "activity": "CMIP",
        "start_year": 1994,
        "end_year": 2014,
    },
    "ssp126": {
        "activity": "ScenarioMIP",
        "start_year": 2040,
        "end_year": 2100,
    },
    "ssp245": {
        "activity": "ScenarioMIP",
        "start_year": 2040,
        "end_year": 2100,
    },
    "ssp585": {
        "activity": "ScenarioMIP",
        "start_year": 2040,
        "end_year": 2100,
    },
}

MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "AWI-CM-1-1-MR",
    "BCC-CSM2-MR",
    "CanESM5",
    "CAS-ESM2-0",
    "CESM2-WACCM",
    "CMCC-CM2-SR5",
    "CMCC-ESM2",
    "FGOALS-f3-L",
    "FIO-ESM-2-0",
    "IPSL-CM6A-LR",
    "KACE-1-0-G",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "TaiESM1",
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

    return urlparse(url).path.rstrip("/").split("/")[-1]


def request_page(url):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140.0 Safari/537.36"
        )
    }

    for attempt in range(1, MAX_RETRIES + 1):

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
                    f"Server error "
                    f"{exc.code} "
                    f"({attempt}/{MAX_RETRIES}): "
                    f"{url}"
                )

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

                continue

            print(
                f"HTTP error "
                f"{exc.code}: "
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

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        except RemoteDisconnected as exc:

            print(
                f"Remote connection closed "
                f"({attempt}/{MAX_RETRIES}): "
                f"{url}"
            )

            print(
                f"Reason: {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        except ssl.SSLError as exc:

            print(
                f"SSL error "
                f"({attempt}/{MAX_RETRIES}): "
                f"{url}"
            )

            print(
                f"Reason: {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

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

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        except Exception as exc:

            print(
                f"Unexpected network error "
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
                time.sleep(RETRY_DELAY)

    print(
        f"Failed after "
        f"{MAX_RETRIES} attempts: "
        f"{url}"
    )

    return None

def list_directory(url):

    url = normalize_url(url)

    html = request_page(url)

    if html is None:
        return None

    parser = LinkParser()
    parser.feed(html)

    base = urlparse(url)
    base_path = base.path

    directories = set()
    files = set()

    for href in parser.hrefs:

        if href.startswith("#"):
            continue

        candidate = urljoin(
            url,
            href,
        )

        parsed = urlparse(candidate)

        if parsed.netloc.lower() != "data.ceda.ac.uk":
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

        if not parsed.path.startswith(base_path):
            continue

        relative = parsed.path[
            len(base_path):
        ].strip("/")

        if not relative:
            continue

        if "/" in relative:
            continue

        if relative.endswith(".nc"):

            files.add(clean)

        else:

            directories.add(
                normalize_url(clean)
            )

    return {
        "directories": sorted(directories),
        "files": sorted(files),
    }


def discover_model_locations(activity):

    activity_url = (
        f"{BASE_URL}/{activity}/"
    )

    print()
    print(
        f"Discovering institutions for {activity}"
    )

    listing = list_directory(
        activity_url
    )

    if listing is None:

        raise RuntimeError(
            f"Cannot access: {activity_url}"
        )

    locations = {
        model: []
        for model in MODELS
    }

    institution_urls = (
        listing["directories"]
    )

    for index, institution_url in enumerate(
        institution_urls,
        start=1,
    ):

        institution = get_name(
            institution_url
        )

        print(
            f"  [{index}/{len(institution_urls)}] "
            f"{institution}"
        )

        institution_listing = (
            list_directory(
                institution_url
            )
        )

        if institution_listing is None:
            continue

        for model_url in (
            institution_listing["directories"]
        ):

            model = get_name(
                model_url
            )

            if model in locations:

                locations[model].append(
                    {
                        "institution": institution,
                        "model_url": model_url,
                    }
                )

                print(
                    f"    Found: {model}"
                )

    return locations


def get_variant_set(
    model_locations,
    experiment,
):

    variants = set()
    variant_sources = {}

    for location in model_locations:

        institution = (
            location["institution"]
        )

        model_url = (
            location["model_url"]
        )

        experiment_url = normalize_url(
            urljoin(
                model_url,
                f"{experiment}/",
            )
        )

        listing = list_directory(
            experiment_url
        )

        if listing is None:
            continue

        for variant_url in (
            listing["directories"]
        ):

            variant = get_name(
                variant_url
            )

            if not re.fullmatch(
                r"r\d+i\d+p\d+f\d+",
                variant,
            ):
                continue

            variants.add(
                variant
            )

            variant_sources.setdefault(
                variant,
                []
            ).append(
                {
                    "institution": institution,
                    "variant_url": variant_url,
                }
            )

    return variants, variant_sources


def choose_common_variants(
    experiment_data,
):

    sets = []

    for experiment in EXPERIMENTS:

        variants = (
            experiment_data[
                experiment
            ]["variants"]
        )

        if not variants:

            return []

        sets.append(
            variants
        )

    common = set.intersection(
        *sets
    )

    def variant_key(value):

        match = re.match(
            r"r(\d+)i(\d+)p(\d+)f(\d+)",
            value,
        )

        if match is None:
            return (
                999999,
                value,
            )

        return tuple(
            int(x)
            for x in match.groups()
        )

    return sorted(
        common,
        key=variant_key,
    )


def get_variable_grids(
    variant_url,
    variable,
):

    table_url = normalize_url(
        urljoin(
            variant_url,
            f"{TABLE_ID}/",
        )
    )

    table_listing = list_directory(
        table_url
    )

    if table_listing is None:
        return []

    variable_map = {
        get_name(url): url
        for url in table_listing[
            "directories"
        ]
    }

    if variable not in variable_map:
        return []

    variable_url = (
        variable_map[variable]
    )

    variable_listing = list_directory(
        variable_url
    )

    if variable_listing is None:
        return []

    return variable_listing[
        "directories"
    ]


def get_real_versions(
    grid_url,
):

    listing = list_directory(
        grid_url
    )

    if listing is None:
        return []

    versions = []

    for url in listing[
        "directories"
    ]:

        name = get_name(
            url
        )

        if re.fullmatch(
            r"v\d{8}",
            name,
        ):

            versions.append(
                url
            )

    versions.sort(
        key=get_name
    )

    return versions


def get_latest_real_version(
    grid_url,
):

    versions = get_real_versions(
        grid_url
    )

    if not versions:
        return None

    return versions[-1]


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
        int(match.group(1)),
        int(match.group(2)),
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


def inspect_dataset_path(
    model,
    experiment,
    activity,
    variant,
    institution,
    variant_url,
    variable,
    target_start,
    target_end,
):

    rows = []

    grid_urls = get_variable_grids(
        variant_url,
        variable,
    )

    if not grid_urls:

        rows.append(
            {
                "model": model,
                "activity": activity,
                "institution": institution,
                "experiment": experiment,
                "variant": variant,
                "table_id": TABLE_ID,
                "variable": variable,
                "grid": "",
                "version": "",
                "status": "VARIABLE_NOT_FOUND",
                "url": "",
                "file_count": 0,
                "matching_file_count": 0,
                "matching_files": "",
            }
        )

        return rows

    for grid_url in grid_urls:

        grid = get_name(
            grid_url
        )

        version_url = (
            get_latest_real_version(
                grid_url
            )
        )

        if version_url is None:

            rows.append(
                {
                    "model": model,
                    "activity": activity,
                    "institution": institution,
                    "experiment": experiment,
                    "variant": variant,
                    "table_id": TABLE_ID,
                    "variable": variable,
                    "grid": grid,
                    "version": "",
                    "status": "NO_REAL_VERSION",
                    "url": grid_url,
                    "file_count": 0,
                    "matching_file_count": 0,
                    "matching_files": "",
                }
            )

            continue

        version = get_name(
            version_url
        )

        version_listing = (
            list_directory(
                version_url
            )
        )

        if version_listing is None:

            rows.append(
                {
                    "model": model,
                    "activity": activity,
                    "institution": institution,
                    "experiment": experiment,
                    "variant": variant,
                    "table_id": TABLE_ID,
                    "variable": variable,
                    "grid": grid,
                    "version": version,
                    "status": "VERSION_NOT_ACCESSIBLE",
                    "url": version_url,
                    "file_count": 0,
                    "matching_file_count": 0,
                    "matching_files": "",
                }
            )

            continue

        nc_files = (
            version_listing["files"]
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

        status = "FOUND"

        if not matching_files:
            status = "NO_TARGET_FILES"

        rows.append(
            {
                "model": model,
                "activity": activity,
                "institution": institution,
                "experiment": experiment,
                "variant": variant,
                "table_id": TABLE_ID,
                "variable": variable,
                "grid": grid,
                "version": version,
                "status": status,
                "url": version_url,
                "file_count": len(
                    nc_files
                ),
                "matching_file_count": len(
                    matching_files
                ),
                "matching_files": ";".join(
                    matching_files
                ),
            }
        )

    return rows


def write_csv(
    rows,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "activity",
        "institution",
        "experiment",
        "variant",
        "table_id",
        "variable",
        "grid",
        "version",
        "status",
        "url",
        "file_count",
        "matching_file_count",
        "matching_files",
    ]

    with CSV_FILE.open(
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
    summary_data,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TXT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "CMIP6 CEDA Directory Scan V2\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        for model in MODELS:

            item = summary_data.get(
                model,
                {}
            )

            file.write(
                f"{model}\n"
            )

            file.write(
                "-" * 80 + "\n"
            )

            for experiment in EXPERIMENTS:

                variants = item.get(
                    "experiment_variants",
                    {}
                ).get(
                    experiment,
                    []
                )

                file.write(
                    f"{experiment}: "
                    f"{', '.join(variants) if variants else 'NONE'}\n"
                )

            common = item.get(
                "common_variants",
                []
            )

            file.write(
                "common_variants: "
                f"{', '.join(common) if common else 'NONE'}\n"
            )

            file.write("\n")


def main():

    print(
        "=" * 80
    )

    print(
        "CMIP6 CEDA DIRECTORY SCANNER V2"
    )

    print(
        "=" * 80
    )

    print(
        "No NetCDF files will be downloaded."
    )

    print()

    cmip_locations = (
        discover_model_locations(
            "CMIP"
        )
    )

    scenario_locations = (
        discover_model_locations(
            "ScenarioMIP"
        )
    )

    all_rows = []
    summary_data = {}

    for model_index, model in enumerate(
        MODELS,
        start=1,
    ):

        print()
        print(
            "=" * 80
        )

        print(
            f"[{model_index}/{len(MODELS)}] "
            f"{model}"
        )

        print(
            "=" * 80
        )

        experiment_data = {}

        for experiment, config in (
            EXPERIMENTS.items()
        ):

            activity = (
                config["activity"]
            )

            if activity == "CMIP":

                locations = (
                    cmip_locations.get(
                        model,
                        [],
                    )
                )

            else:

                locations = (
                    scenario_locations.get(
                        model,
                        [],
                    )
                )

            variants, sources = (
                get_variant_set(
                    locations,
                    experiment,
                )
            )

            experiment_data[
                experiment
            ] = {
                "variants": variants,
                "sources": sources,
            }

            print(
                f"{experiment}: "
                f"{len(variants)} variants"
            )

            if variants:

                print(
                    "  "
                    + ", ".join(
                        sorted(
                            variants
                        )
                    )
                )

        common_variants = (
            choose_common_variants(
                experiment_data
            )
        )

        print()

        print(
            "Common variants:"
        )

        if common_variants:

            print(
                "  "
                + ", ".join(
                    common_variants
                )
            )

        else:

            print(
                "  NONE"
            )

        summary_data[
            model
        ] = {
            "experiment_variants": {
                experiment: sorted(
                    experiment_data[
                        experiment
                    ]["variants"]
                )
                for experiment
                in EXPERIMENTS
            },
            "common_variants": (
                common_variants
            ),
        }

        if not common_variants:

            write_csv(
                all_rows
            )

            write_summary(
                summary_data
            )

            continue

        preferred_variant = (
            common_variants[0]
        )

        print()

        print(
            f"Selected variant: "
            f"{preferred_variant}"
        )

        for experiment, config in (
            EXPERIMENTS.items()
        ):

            activity = (
                config["activity"]
            )

            sources = (
                experiment_data[
                    experiment
                ]["sources"].get(
                    preferred_variant,
                    [],
                )
            )

            if not sources:

                continue

            print()
            print(
                f"  {experiment}"
            )

            for source in sources:

                institution = (
                    source[
                        "institution"
                    ]
                )

                variant_url = (
                    source[
                        "variant_url"
                    ]
                )

                print(
                    f"    Institution: "
                    f"{institution}"
                )

                for variable in VARIABLES:

                    print(
                        f"      Variable: "
                        f"{variable}"
                    )

                    rows = (
                        inspect_dataset_path(
                            model=model,
                            experiment=experiment,
                            activity=activity,
                            variant=preferred_variant,
                            institution=institution,
                            variant_url=variant_url,
                            variable=variable,
                            target_start=(
                                config[
                                    "start_year"
                                ]
                            ),
                            target_end=(
                                config[
                                    "end_year"
                                ]
                            ),
                        )
                    )

                    all_rows.extend(
                        rows
                    )

        write_csv(
            all_rows
        )

        write_summary(
            summary_data
        )

        print()
        print(
            f"Progress saved: "
            f"{CSV_FILE}"
        )

    print()
    print(
        "=" * 80
    )

    print(
        "SCAN FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"CSV: {CSV_FILE}"
    )

    print(
        f"TXT: {TXT_FILE}"
    )

    print(
        f"Rows: {len(all_rows)}"
    )


if __name__ == "__main__":

    main()