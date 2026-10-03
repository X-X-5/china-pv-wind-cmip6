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
import pandas as pd


BASE_DIR = get_path("data_raw_cmip6_downloaded")

MAIN_CSV = BASE_DIR / "cmip6_ceda_directory_scan_v2.csv"
RECHECK_CSV = BASE_DIR / "cmip6_ceda_missing_recheck.csv"

OUTPUT_CSV = BASE_DIR / "cmip6_ceda_clean_manifest_16models.csv"

EXCLUDED_MODELS = {
    "CAS-ESM2-0",
}


def main():

    main_df = pd.read_csv(
        MAIN_CSV
    )

    recheck_df = pd.read_csv(
        RECHECK_CSV
    )

    print(
        f"Main rows: {len(main_df)}"
    )

    print(
        f"Recheck rows: {len(recheck_df)}"
    )

    main_df = main_df[
        ~main_df["model"].isin(
            EXCLUDED_MODELS
        )
    ].copy()

    recheck_df = recheck_df[
        ~recheck_df["model"].isin(
            EXCLUDED_MODELS
        )
    ].copy()

    replacement_keys = (
        recheck_df[
            [
                "model",
                "experiment",
                "variable",
            ]
        ]
        .drop_duplicates()
    )

    merged = main_df.merge(
        replacement_keys.assign(
            replace=True
        ),
        on=[
            "model",
            "experiment",
            "variable",
        ],
        how="left",
    )

    main_df = merged[
        merged["replace"].isna()
    ].drop(
        columns=[
            "replace"
        ]
    )

    common_columns = [
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

    main_df = main_df[
        common_columns
    ]

    recheck_df = recheck_df[
        common_columns
    ]

    combined = pd.concat(
        [
            main_df,
            recheck_df,
        ],
        ignore_index=True,
    )

    combined = combined[
        combined["status"] == "FOUND"
    ].copy()

    combined = combined[
        combined[
            "matching_file_count"
        ] > 0
    ].copy()

    combined = combined.sort_values(
        by=[
            "model",
            "experiment",
            "variable",
        ]
    )

    combined.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    models = sorted(
        combined[
            "model"
        ].unique()
    )

    print()
    print(
        "=" * 80
    )

    print(
        "CLEAN MANIFEST CREATED"
    )

    print(
        "=" * 80
    )

    print(
        f"Models: {len(models)}"
    )

    print(
        f"Rows: {len(combined)}"
    )

    print(
        f"Output: {OUTPUT_CSV}"
    )

    print()

    for model in models:

        count = len(
            combined[
                combined["model"]
                == model
            ]
        )

        print(
            f"{model}: {count} rows"
        )

    expected_rows = (
        16
        * 4
        * 3
    )

    print()
    print(
        f"Expected rows: "
        f"{expected_rows}"
    )

    print(
        f"Actual rows: "
        f"{len(combined)}"
    )

    if len(combined) != expected_rows:

        print()
        print(
            "WARNING: "
            "Manifest is incomplete."
        )

    else:

        print()
        print(
            "Manifest structure is complete."
        )


if __name__ == "__main__":

    main()