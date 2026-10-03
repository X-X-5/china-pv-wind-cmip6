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

import xarray as xr


ROOT = get_path("data_interim_cmip6_processed_16models")

MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "AWI-CM-1-1-MR",
    "BCC-CSM2-MR",
    "CanESM5",
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


def find_file(model):

    directory = (
        ROOT
        / model
        / "historical"
        / "tas"
    )

    files = sorted(
        directory.glob("*.nc")
    )

    if len(files) != 1:

        raise ValueError(
            f"{model}: expected 1 file, found {len(files)}"
        )

    return files[0]


def main():

    print()
    print("=" * 120)
    print("CMIP6 PROCESSED_RIGHT SPATIAL DOMAIN CHECK")
    print("=" * 120)

    for model in MODELS:

        file_path = find_file(
            model
        )

        ds = xr.open_dataset(
            file_path
        )

        try:

            if "lat" not in ds.coords:
                raise ValueError(
                    f"{model}: lat coordinate missing"
                )

            if "lon" not in ds.coords:
                raise ValueError(
                    f"{model}: lon coordinate missing"
                )

            lat = ds["lat"]
            lon = ds["lon"]

            print()
            print(model)

            print(
                f"  lat count: {lat.size}"
            )

            print(
                f"  lat range: "
                f"{float(lat.min().values):.6f} "
                f"to "
                f"{float(lat.max().values):.6f}"
            )

            print(
                f"  lon count: {lon.size}"
            )

            print(
                f"  lon range: "
                f"{float(lon.min().values):.6f} "
                f"to "
                f"{float(lon.max().values):.6f}"
            )

            print(
                f"  time count: {ds.sizes['time']}"
            )

        finally:

            ds.close()

    print()
    print("=" * 120)
    print("SPATIAL DOMAIN CHECK COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()