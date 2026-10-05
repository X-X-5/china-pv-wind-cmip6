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
import numpy as np
import xarray as xr


# Verifies the staged expanded ERA5 reference (165x281, lat 15-56) against the
# current canonical reference (157x281, lat 15-54) on the overlap region, and
# confirms the extended north strip (54.25-56) carries no NaN.  This is the
# gate before promoting the expanded file to the canonical path.

EXPANDED_FILE = (
    PROJECT_ROOT / "data" / "raw" / "era5_expanded" / "ERA5_reference_195901_201412_expanded.nc"
)

CANONICAL_FILE = (
    get_path("data_interim_era5_reference") / "ERA5_reference_195901_201412.nc"
)

VARIABLES = ["tas", "sfcWind", "rsds"]

# Old canonical domain (lat 15..54).  Overlap is the full old domain.
OLD_LAT_MIN = 15.0
OLD_LAT_MAX = 54.0

# Extended north strip added by the expanded reference (lat > 54.0).
EXTENDED_LAT_MIN = 54.0


def main():
    print("=" * 90)
    print("VERIFY EXPANDED ERA5 REFERENCE")
    print("=" * 90)

    new = xr.open_dataset(EXPANDED_FILE)
    old = xr.open_dataset(CANONICAL_FILE)

    try:
        # --- 1. 672 continuous months ---
        if new.sizes["time"] != 672:
            raise ValueError(f"Expanded has {new.sizes['time']} months, expected 672")

        years = new["time"].dt.year.values
        months = new["time"].dt.month.values
        month_id = years * 12 + months - 1
        if not np.array_equal(month_id, np.arange(month_id[0], month_id[0] + 672)):
            raise ValueError("Expanded time sequence is not continuous")

        if (int(years[0]), int(months[0])) != (1959, 1):
            raise ValueError("Expanded does not start 1959-01")

        if (int(years[-1]), int(months[-1])) != (2014, 12):
            raise ValueError("Expanded does not end 2014-12")

        print()
        print(f"[1] Time continuity: OK  ({new.sizes['time']} months, "
              f"{new.time.values[0]} .. {new.time.values[-1]})")

        # --- 2. grid + no NaN in extended north strip ---
        print()
        print("[2] Extended region NaN check:")
        extended = new.sel(lat=slice(EXTENDED_LAT_MIN, None))
        print(f"    extended lat range: {float(extended.lat.min()):.4f} .. "
              f"{float(extended.lat.max()):.4f} "
              f"({extended.sizes['lat']} points)")
        for variable in VARIABLES:
            nan_count = int(np.isnan(extended[variable].values).sum())
            print(f"    {variable}: NaN count in extended strip = {nan_count}")
            if nan_count != 0:
                raise ValueError(f"{variable} has NaN in the extended north strip")

        print(f"    north edge covers CanESM5 54.4162 N? "
              f"{'YES' if float(new.lat.max()) >= 54.4162 else 'NO'} "
              f"(lat_max={float(new.lat.max()):.4f})")

        # --- 3. overlap region per-variable max_abs_diff ---
        print()
        print("[3] Overlap comparison (expanded vs canonical, lat 15-54):")

        overlap_new = new.sel(lat=slice(OLD_LAT_MIN, OLD_LAT_MAX))

        if not np.allclose(
            overlap_new["lat"].values,
            old["lat"].values,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("Overlap latitude coordinates do not match")

        if not np.allclose(
            overlap_new["lon"].values,
            old["lon"].values,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("Overlap longitude coordinates do not match")

        if overlap_new.sizes["time"] != old.sizes["time"]:
            raise ValueError(
                f"Overlap time sizes differ: {overlap_new.sizes['time']} vs {old.sizes['time']}"
            )

        all_ok = True
        for variable in VARIABLES:
            diff = np.abs(
                overlap_new[variable].values.astype(np.float64)
                - old[variable].values.astype(np.float64)
            )
            max_diff = float(np.nanmax(diff)) if diff.size else np.nan
            n_differ = int(np.count_nonzero(diff > 1e-12))
            print(f"    {variable}: max_abs_diff = {max_diff:.6e}  "
                  f"(cells differing >1e-12: {n_differ})")
            # Tolerance: 0 or pure float-write error only (allow ~1e-5 for f32 write).
            if max_diff > 1e-5:
                all_ok = False

        print()
        if all_ok:
            print("RESULT: PASS — overlap values match to float-write precision; "
                  "extended strip has no NaN.")
        else:
            print("RESULT: FAIL — overlap differences exceed float-write tolerance.")
            raise SystemExit(1)

    finally:
        new.close()
        old.close()


if __name__ == "__main__":
    main()
