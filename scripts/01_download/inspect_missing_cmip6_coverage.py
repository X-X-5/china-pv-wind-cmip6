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
import glob

import xarray as xr


BASE_DIR = get_path("data_raw_cmip6_downloaded")

RAW_DIR = (
    BASE_DIR
    / "downloaded"
)

MODELS = [
    "AWI-CM-1-1-MR",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
]

EXPERIMENTS = [
    "historical",
    "ssp126",
    "ssp245",
    "ssp585",
]

VARIABLES = [
    "tas",
    "sfcWind",
    "rsds",
]


def find_files(
    model,
    experiment,
    variable,
):

    pattern = str(
        RAW_DIR
        / "**"
        / f"{variable}_Amon_{model}_{experiment}_*.nc"
    )

    return sorted(
        glob.glob(
            pattern,
            recursive=True,
        )
    )


def inspect_file(
    file_path,
):

    time_coder = xr.coders.CFDatetimeCoder(
        use_cftime=True
    )

    ds = xr.open_dataset(
        file_path,
        decode_times=time_coder,
    )

    try:

        count = ds.sizes[
            "time"
        ]

        first_year = int(
            ds.time.dt.year.values[
                0
            ]
        )

        first_month = int(
            ds.time.dt.month.values[
                0
            ]
        )

        last_year = int(
            ds.time.dt.year.values[
                -1
            ]
        )

        last_month = int(
            ds.time.dt.month.values[
                -1
            ]
        )

        calendar = (
            ds.time.encoding.get(
                "calendar",
                ds.time.attrs.get(
                    "calendar",
                    "unknown",
                ),
            )
        )

        return {
            "count": count,
            "start": (
                f"{first_year:04d}-"
                f"{first_month:02d}"
            ),
            "end": (
                f"{last_year:04d}-"
                f"{last_month:02d}"
            ),
            "calendar": calendar,
        }

    finally:

        ds.close()


def main():

    print()
    print(
        "=" * 120
    )

    print(
        "CMIP6 RAW COVERAGE INSPECTION"
    )

    print(
        "=" * 120
    )

    print(
        f"Raw directory: {RAW_DIR}"
    )

    for model in MODELS:

        print()
        print(
            "#" * 120
        )

        print(
            model
        )

        print(
            "#" * 120
        )

        for experiment in EXPERIMENTS:

            print()
            print(
                f"[{experiment}]"
            )

            for variable in VARIABLES:

                files = find_files(
                    model,
                    experiment,
                    variable,
                )

                print()
                print(
                    f"  {variable}"
                )

                print(
                    f"    files: {len(files)}"
                )

                if not files:

                    print(
                        "    NO FILES"
                    )

                    continue

                total_months = 0

                earliest = None
                latest = None

                for file_path in files:

                    info = inspect_file(
                        file_path
                    )

                    total_months += info[
                        "count"
                    ]

                    if (
                        earliest is None
                        or info[
                            "start"
                        ] < earliest
                    ):

                        earliest = info[
                            "start"
                        ]

                    if (
                        latest is None
                        or info[
                            "end"
                        ] > latest
                    ):

                        latest = info[
                            "end"
                        ]

                    print(
                        f"    {Path(file_path).name}"
                    )

                    print(
                        f"      months: "
                        f"{info['count']}"
                    )

                    print(
                        f"      period: "
                        f"{info['start']} "
                        f"to "
                        f"{info['end']}"
                    )

                    print(
                        f"      calendar: "
                        f"{info['calendar']}"
                    )

                print(
                    f"    combined raw count: "
                    f"{total_months}"
                )

                print(
                    f"    raw coverage: "
                    f"{earliest} "
                    f"to "
                    f"{latest}"
                )

    print()
    print(
        "=" * 120
    )

    print(
        "INSPECTION COMPLETE"
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()