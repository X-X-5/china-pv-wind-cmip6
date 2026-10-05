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
import cdsapi


# Canonical ERA5 reference download (1959-2014), area = [56, 70, 15, 140].
#
# The North bound is 56 deg N (not 54) so the 0.25-deg reference covers
# CanESM5's northernmost cell centre at 54.4162 N.  West / South / East stay
# 70 / 15 / 140 (the minimal China-intersect crop never exceeds 73.125-135.0 E).
#
# Two periods are downloaded to bound the CDS request size:
#   1959-1993 (35 years) and 1994-2014 (21 years).
#
# Outputs are staged in ``data/raw/era5_expanded/`` as GRIB, then converted to
# the canonical 672-month NetCDF reference by
# ``scripts/02_data_preparation/prepare_era5_reference_expanded.py``
# (which writes ``data/interim/era5_reference/ERA5_reference_195901_201412.nc``).

DATASET = "reanalysis-era5-single-levels-monthly-means"

OUTPUT_DIR = get_path("data_raw_era5_expanded")

PERIODS = [
    ("1959_1993", 1959, 1993),
    ("1994_2014", 1994, 2014),
]

VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "surface_solar_radiation_downwards",
]

# area order: North, West, South, East
AREA = [56, 70, 15, 140]


def build_request(start_year: int, end_year: int) -> dict:
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": VARIABLES,
        "year": [str(year) for year in range(start_year, end_year + 1)],
        "month": [f"{month:02d}" for month in range(1, 13)],
        "time": ["00:00"],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": AREA,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for label, start_year, end_year in PERIODS:
        output_file = OUTPUT_DIR / f"ERA5_{label}_expanded.grib"
        print()
        print("=" * 100)
        print(f"DOWNLOADING {label} ({start_year}-{end_year})  area={AREA}")
        print("=" * 100)
        request = build_request(start_year, end_year)
        result = client.retrieve(DATASET, request)
        result.download(str(output_file))
        print(f"Wrote {output_file}")

    print()
    print("ERA5 EXPANDED DOWNLOAD COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Next: scripts/02_data_preparation/prepare_era5_reference_expanded.py")


if __name__ == "__main__":
    main()
