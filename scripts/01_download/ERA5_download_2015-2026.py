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

import cdsapi


OUTPUT_DIR = get_path("data_raw_era5_2015_2025")

OUTPUT_FILE = OUTPUT_DIR / "ERA5_monthly_201501_202512.grib"

# 文件夹不存在时自动创建
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


dataset = "reanalysis-era5-single-levels-monthly-means"

request = {
    "product_type": [
        "monthly_averaged_reanalysis"
    ],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "surface_solar_radiation_downwards",
    ],
    "year": [
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
    ],
    "month": [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
    ],
    "time": [
        "00:00"
    ],
    "data_format": "grib",
    "download_format": "unarchived",

    # 顺序：北、西、南、东
    "area": [
        54,
        70,
        15,
        140,
    ],
}


client = cdsapi.Client()

result = client.retrieve(
    dataset,
    request,
)

result.download(
    str(OUTPUT_FILE)
)

print("ERA5 DOWNLOAD COMPLETED")
print(f"Output file: {OUTPUT_FILE}")
