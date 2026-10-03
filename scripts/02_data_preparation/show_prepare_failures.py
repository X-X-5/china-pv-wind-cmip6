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
import pandas as pd


REPORT_FILE = (
    get_path("data_raw_cmip6_downloaded") / "cmip6_prepare_target_periods_report.csv"
)


df = pd.read_csv(
    REPORT_FILE
)

failed = df[
    df["status"] != "OK"
].copy()

columns = [
    "model",
    "experiment",
    "variable",
    "input_file_count",
    "input_time_count",
    "output_time_count",
    "error_type",
    "error_message",
]

print(
    "=" * 100
)

print(
    f"Failed groups: {len(failed)}"
)

print(
    "=" * 100
)

print(
    failed[
        columns
    ].to_string(
        index=False
    )
)

print()

print(
    "Failure count by model:"
)

print(
    failed[
        "model"
    ].value_counts()
)

print()

print(
    "Failure count by error type:"
)

print(
    failed[
        "error_type"
    ].value_counts()
)