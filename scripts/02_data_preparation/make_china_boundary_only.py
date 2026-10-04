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
"""Create a China analysis-domain Natural Earth shapefile for QC masking.

This script only creates boundary files. It does not read CMIP6, ERA5,
QM, or QDM outputs.
"""

from pathlib import Path
import shutil
import sys

import cartopy.io.shapereader as shpreader
import shapefile
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union


OUTPUT_DIR = get_path("data_boundaries")
OUTPUT_SHP = OUTPUT_DIR / "china_qc_boundary.shp"

# China land area within the study domain is stored by Natural Earth as
# separate admin-0 records (CHN, TWN, and the HKG/MAC enclaves). They are
# unioned here solely to construct the quantitative analysis mask.
REQUIRED_ADM0_A3 = ("CHN", "TWN", "HKG", "MAC")


def is_target(attributes):
    return str(attributes.get("ADM0_A3", "")).strip().upper() in REQUIRED_ADM0_A3


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
        if is_target(attributes):
            selected.append((shape_record.shape, attributes))

    found = sorted(
        str(attributes.get("ADM0_A3", "")).strip().upper()
        for _, attributes in selected
    )
    if found != sorted(REQUIRED_ADM0_A3):
        raise RuntimeError(
            "Expected exactly one record for each required Natural Earth "
            f"admin-0 code {sorted(REQUIRED_ADM0_A3)}; found {found}."
        )

    # Union all selected geometries into a single boundary, preserving every
    # Polygon/MultiPolygon part (mainland, Hainan, islands, Taiwan and its
    # offshore islands, and the Hong Kong / Macao enclaves).
    merged = unary_union([shapely_shape(shape) for shape, _ in selected])

    writer = shapefile.Writer(
        str(OUTPUT_SHP),
        shapeType=shapefile.POLYGON,
        encoding="utf-8",
    )
    writer.field("NAME", "C", size=80)
    writer.field("ADM0_A3", "C", size=20)
    writer.field("SOURCE", "C", size=40)
    writer.field("PURPOSE", "C", size=120)

    writer.shape(merged)
    writer.record(
        NAME="China analysis domain",
        ADM0_A3="CHN+TWN+HKG+MAC",
        SOURCE="Natural Earth 10m",
        PURPOSE="quantitative analysis mask; not an official standard map",
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
    print("CHINA ANALYSIS BOUNDARY CREATED")
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
