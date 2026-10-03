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
"""Create a China-only Natural Earth shapefile for QC masking.

This script only creates boundary files. It does not read CMIP6, ERA5,
QM, or QDM outputs.
"""

from pathlib import Path
import shutil
import sys

import cartopy.io.shapereader as shpreader
import shapefile


OUTPUT_DIR = get_path("data_boundaries")
OUTPUT_SHP = OUTPUT_DIR / "china_qc_boundary.shp"


def is_china(attributes):
    # A broad SOVEREIGNT == "China" test also selects the separate
    # Hong Kong and Macao records in Natural Earth.  Select the single
    # country polygon explicitly so this QC mask has exactly one record.
    admin = str(attributes.get("ADMIN", "")).strip().lower()
    adm0_a3 = str(attributes.get("ADM0_A3", "")).strip().upper()
    return admin == "china" and adm0_a3 == "CHN"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_shp = Path(
        shpreader.natural_earth(
            resolution="10m",
            category="cultural",
            name="admin_0_countries",
        )
    )

    reader = shapefile.Reader(str(source_shp), encoding="utf-8")
    field_names = [field[0] for field in reader.fields[1:]]

    selected = []
    for shape_record in reader.iterShapeRecords():
        attributes = dict(zip(field_names, shape_record.record))
        if is_china(attributes):
            selected.append((shape_record.shape, attributes))

    if len(selected) != 1:
        raise RuntimeError(
            "Expected exactly one China record in Natural Earth; "
            f"found {len(selected)}."
        )

    writer = shapefile.Writer(
        str(OUTPUT_SHP),
        shapeType=reader.shapeType,
        encoding="utf-8",
    )
    writer.field("NAME", "C", size=80)
    writer.field("ADM0_A3", "C", size=10)
    writer.field("SOURCE", "C", size=40)

    for geometry, attributes in selected:
        writer.shape(geometry)
        writer.record(
            NAME=attributes.get("ADMIN", "China"),
            ADM0_A3=attributes.get("ADM0_A3", "CHN"),
            SOURCE="Natural Earth 10m",
        )

    writer.close()
    reader.close()

    source_prj = source_shp.with_suffix(".prj")
    if source_prj.exists():
        shutil.copyfile(source_prj, OUTPUT_SHP.with_suffix(".prj"))
    else:
        OUTPUT_SHP.with_suffix(".prj").write_text(
            'GEOGCS["WGS 84",DATUM["WGS_1984",'
            'SPHEROID["WGS 84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]',
            encoding="ascii",
        )

    OUTPUT_SHP.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")

    check = shapefile.Reader(str(OUTPUT_SHP), encoding="utf-8")
    print("=" * 80)
    print("CHINA QC SHAPEFILE CREATED")
    print("=" * 80)
    print(f"Source  : {source_shp}")
    print(f"Output  : {OUTPUT_SHP}")
    print(f"Records : {len(check)}")
    print(f"BBox    : {check.bbox}")
    print("Files   : .shp, .shx, .dbf, .prj, .cpg")
    check.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
