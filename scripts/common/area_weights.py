#!/usr/bin/env python
r"""Shared intersection-area weighting for China-region spatial statistics.

The centre-point mask (``geometry.covers(Point(lon, lat))``) and the
cos(latitude) weight are both retired as *formal* national statistics.  A grid
cell is now attributed to the China analysis region by the *geodesic area* of
its intersection with the China boundary polygon, so a national spatial mean is

.. math::

    \bar{x} = \frac{\sum_i x_i \, A_i}{\sum_i A_i},

where :math:`A_i` is the ellipsoidal area (WGS84, km²) of the overlap
between grid cell :math:`i` and the China boundary, and the sums run over the
cells with finite :math:`x_i` and :math:`A_i > 0`.

Only the *area* changes; the underlying per-cell energy values are untouched.

Public functions
----------------
* :func:`coordinate_edges_from_centers` -- cell edges from centre midpoints.
* :func:`geodesic_gridcell_areas`     -- full-cell ellipsoidal areas (km²).
* :func:`china_intersection_weights`  -- per-cell China-overlap areas/fractions.
* :func:`area_weighted_mean`          -- area-weighted mean over (lat, lon).
* :func:`area_weighted_median`        -- area-weighted median over (lat, lon).

Weight arrays are cached for identical grids (and an unchanged shapefile), so
the 17-model pipeline pays the geodesic cost once per unique native grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from pyproj import Geod
except ImportError as error:  # pragma: no cover - environment-specific
    raise RuntimeError(
        "pyproj is required for intersection-area weighting. "
        "Install it in the pvwind environment."
    ) from error

try:
    from shapely.geometry import box as _shapely_box
    from shapely.ops import unary_union as _unary_union
except ImportError as error:  # pragma: no cover - environment-specific
    raise RuntimeError(
        "shapely is required for intersection-area weighting. "
        "Install it in the pvwind environment."
    ) from error

__all__ = [
    "coordinate_edges_from_centers",
    "geodesic_gridcell_areas",
    "china_intersection_weights",
    "area_weighted_mean",
    "area_weighted_median",
]

# Intersections below this many km^2 (== 1 m^2) are treated as "no overlap".
# Exact touches (a corner on the polygon edge) yield zero geodesic area and are
# excluded naturally; this floor only removes genuine numerical dust.
_AREA_FLOOR_KM2 = 1.0e-6

# A cell is "fully inside" when its overlap reaches the full cell area to this
# relative tolerance (geodesic area is only accurate to ~1e-9 relative).
_FULLY_INSIDE_REL_TOL = 1.0e-6

# Cache: (lat_bytes, lon_bytes, shapefile_signature) -> weights result dict.
_WEIGHTS_CACHE: dict[tuple[bytes, bytes, tuple[Any, ...]], dict[str, Any]] = {}


def coordinate_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Derive ``n + 1`` cell edges from ``n`` centre coordinates.

    Edges are placed at the midpoints of neighbouring centres; the first and
    last intervals are reflected about the first/last centre.  Both ascending
    and descending 1-D coordinates are supported, but the coordinate must be
    strictly monotonic and finite.
    """
    centers = np.asarray(centers, dtype=np.float64).ravel()
    if centers.size < 2:
        raise ValueError("At least two cell centres are required to derive edges")
    if not np.all(np.isfinite(centers)):
        raise ValueError("Coordinate centres must be finite")
    differences = np.diff(centers)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise ValueError(
            "Coordinate centres must be strictly monotonic (ascending or "
            "descending); equal or non-monotonic values are not supported"
        )
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first_edge = centers[0] - 0.5 * (centers[1] - centers[0])
    last_edge = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate(([first_edge], midpoints, [last_edge]))


def _geometry_area_km2(geometry, geod: Geod) -> float:
    """Ellipsoidal area (km^2) of any Polygon/MultiPolygon, holes included."""
    area_m2, _ = geod.geometry_area_perimeter(geometry)
    return float(abs(area_m2)) / 1.0e6


def geodesic_gridcell_areas(
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
    geod: Geod | None = None,
) -> np.ndarray:
    """Return ``(nlat, nlon)`` full-cell ellipsoidal areas in km².

    Areas use :class:`pyproj.Geod` (WGS84) on the lon/lat rectangles implied by
    the centre coordinates, so non-uniform Gaussian latitude grids are handled
    exactly (each row has its own meridional extent).
    """
    lat_centers = np.asarray(lat_centers, dtype=np.float64)
    lon_centers = np.asarray(lon_centers, dtype=np.float64)
    lat_edges = coordinate_edges_from_centers(lat_centers)
    lon_edges = coordinate_edges_from_centers(lon_centers)
    geod = geod if geod is not None else Geod(ellps="WGS84")
    nlat, nlon = lat_centers.size, lon_centers.size
    areas = np.zeros((nlat, nlon), dtype=np.float64)
    for i in range(nlat):
        lat_lo, lat_hi = sorted((lat_edges[i], lat_edges[i + 1]))
        for j in range(nlon):
            lon_lo, lon_hi = sorted((lon_edges[j], lon_edges[j + 1]))
            areas[i, j] = _geometry_area_km2(
                _shapely_box(lon_lo, lat_lo, lon_hi, lat_hi), geod
            )
    return areas


def _shapefile_signature(shapefile: Path) -> tuple[Any, ...]:
    stat = shapefile.stat()
    return (str(shapefile.resolve()), stat.st_mtime_ns, stat.st_size)


def _load_china_geometry(shapefile: Path):
    """Union all polygon features of a WGS84 shapefile into one geometry."""
    if not shapefile.is_file():
        raise FileNotFoundError(f"China shapefile does not exist: {shapefile}")
    for suffix in (".shx", ".dbf", ".prj"):
        if not shapefile.with_suffix(suffix).is_file():
            raise FileNotFoundError(
                f"China shapefile sidecar is missing: {shapefile.with_suffix(suffix)}"
            )
    try:
        import cartopy.io.shapereader as shpreader
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "cartopy is required to read the China boundary shapefile."
        ) from error
    reader = shpreader.Reader(str(shapefile))
    geometries = list(reader.geometries())
    close = getattr(reader, "close", None)
    if close is not None:
        close()
    if not geometries:
        raise ValueError(f"No geometry found in {shapefile}")
    return _unary_union(geometries)


def china_intersection_weights(
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
    shapefile: str | Path,
    geod: Geod | None = None,
) -> dict[str, Any]:
    """Per-cell China-overlap areas, fractions, and selection mask.

    Parameters
    ----------
    lat_centers, lon_centers:
        1-D centre coordinates of a rectilinear grid (ascending or descending).
    shapefile:
        WGS84 China boundary polygon (any Polygon/MultiPolygon, holes included).

    Returns
    -------
    dict with keys:
        ``gridcell_area_km2``            (nlat, nlon) full-cell areas
        ``china_intersection_area_km2``  (nlat, nlon) overlap areas (0 if none)
        ``china_area_fraction``          (nlat, nlon) overlap / full cell in [0,1]
        ``china_intersects``             (nlat, nlon) bool selection mask
        ``china_boundary_area_km2``      float total boundary area
        ``counts``                       dict of cell-count tallies
    """
    lat_centers = np.asarray(lat_centers, dtype=np.float64)
    lon_centers = np.asarray(lon_centers, dtype=np.float64)
    shapefile = Path(shapefile)
    key = (
        lat_centers.tobytes(),
        lon_centers.tobytes(),
        _shapefile_signature(shapefile),
    )
    cached = _WEIGHTS_CACHE.get(key)
    if cached is not None:
        return cached

    lat_edges = coordinate_edges_from_centers(lat_centers)
    lon_edges = coordinate_edges_from_centers(lon_centers)
    geod = geod if geod is not None else Geod(ellps="WGS84")
    china = _load_china_geometry(shapefile)

    nlat, nlon = lat_centers.size, lon_centers.size
    gridcell = np.zeros((nlat, nlon), dtype=np.float64)
    intersection = np.zeros((nlat, nlon), dtype=np.float64)
    for i in range(nlat):
        lat_lo, lat_hi = sorted((lat_edges[i], lat_edges[i + 1]))
        for j in range(nlon):
            lon_lo, lon_hi = sorted((lon_edges[j], lon_edges[j + 1]))
            cell = _shapely_box(lon_lo, lat_lo, lon_hi, lat_hi)
            gridcell[i, j] = _geometry_area_km2(cell, geod)
            overlap = cell.intersection(china)
            if not overlap.is_empty:
                intersection[i, j] = _geometry_area_km2(overlap, geod)

    intersection = np.where(
        intersection < _AREA_FLOOR_KM2, 0.0, intersection
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(
            gridcell > 0.0, intersection / gridcell, 0.0
        )
    fraction = np.clip(fraction, 0.0, 1.0)
    intersects = intersection > 0.0
    fully_inside = fraction >= 1.0 - _FULLY_INSIDE_REL_TOL
    partial = intersects & ~fully_inside
    outside = ~intersects

    result: dict[str, Any] = {
        "gridcell_area_km2": gridcell,
        "china_intersection_area_km2": intersection,
        "china_area_fraction": fraction,
        "china_intersects": intersects,
        "china_boundary_area_km2": _geometry_area_km2(china, geod),
        "counts": {
            "total": int(lat_centers.size * lon_centers.size),
            "intersects": int(intersects.sum()),
            "fully_inside": int(fully_inside.sum()),
            "partial": int(partial.sum()),
            "outside": int(outside.sum()),
        },
    }
    _WEIGHTS_CACHE[key] = result
    return result


def area_weighted_mean(values: np.ndarray, weights: np.ndarray):
    """Area-weighted mean over the trailing ``(lat, lon)`` axes.

    ``values`` may be 2-D ``(lat, lon)`` (returns a float) or any leading shape
    ``(..., lat, lon)`` (returns the same leading shape).  ``weights`` is a
    2-D ``(lat, lon)`` area array (e.g. ``china_intersection_area_km2``).
    NaN cells contribute neither to the numerator nor the denominator.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2:] != weights.shape:
        raise ValueError(
            f"values must end in (lat, lon) matching weights; "
            f"got values shape {values.shape} and weights shape {weights.shape}"
        )
    finite = np.isfinite(values)
    numerator = np.where(finite, values * weights, 0.0).sum(axis=(-2, -1))
    denominator = np.where(finite, weights, 0.0).sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.divide(
            numerator,
            denominator,
            out=np.full(np.shape(numerator), np.nan, dtype=np.float64),
            where=denominator > 0.0,
        )
    if values.ndim == 2:
        return float(result)
    return result


def area_weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Area-weighted median over a 2-D ``(lat, lon)`` field."""
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if values.shape != weights.shape:
        raise ValueError("values and weights must share the same shape")
    finite = np.isfinite(values) & (weights > 0.0)
    ordered_values = values[finite]
    ordered_weights = weights[finite]
    if ordered_values.size == 0:
        return float("nan")
    order = np.argsort(ordered_values)
    ordered_values = ordered_values[order]
    ordered_weights = ordered_weights[order]
    cumulative = np.cumsum(ordered_weights)
    total = cumulative[-1]
    if total <= 0.0:
        return float("nan")
    index = int(np.searchsorted(cumulative, 0.5 * total))
    return float(ordered_values[index])


def _run_self_test(shapefile: str | Path | None = None) -> int:
    """Validate the module invariants; returns the number of failed checks."""
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    # --- coordinate_edges_from_centers -----------------------------------
    ascending = np.array([15.0, 16.0, 17.0])
    edges = coordinate_edges_from_centers(ascending)
    check(edges.size == 4, "ascending edges length != n+1")
    check(np.allclose(edges, [14.5, 15.5, 16.5, 17.5]), "ascending edges wrong")
    descending = np.array([17.0, 16.0, 15.0])
    d_edges = coordinate_edges_from_centers(descending)
    check(np.allclose(np.sort(d_edges), np.sort(edges)), "descending edges wrong")
    for bad in (np.array([15.0, 15.0, 16.0]), np.array([15.0, 17.0, 16.0])):
        try:
            coordinate_edges_from_centers(bad)
            check(False, "non-monotonic centres did not raise")
        except ValueError:
            pass

    # --- geodesic_gridcell_areas: symmetric grid, uniform area -----------
    lat = np.arange(15.0, 55.0, 1.0)
    lon = np.arange(73.0, 136.0, 1.0)
    areas = geodesic_gridcell_areas(lat, lon)
    check(areas.shape == (40, 63), f"area shape {areas.shape} != (40, 63)")
    check(np.all(np.isfinite(areas)) and np.all(areas > 0.0), "non-positive area")
    # A 1 deg x 1 deg cell near 15N must be > 0 and < a pole cell near 54N.
    check(areas[0, 0] > areas[-1, -1], "cell area should decrease with latitude")
    # Symmetric in longitude for a uniform lon grid.
    check(
        np.allclose(areas[:, 0], areas[:, -1], rtol=1e-12),
        "uniform-lon grid not longitude-symmetric",
    )

    # --- area_weighted_mean ----------------------------------------------
    values = np.array([[1.0, 2.0, np.nan], [4.0, 6.0, np.nan]])
    weights = np.array([[1.0, 1.0, 5.0], [2.0, 2.0, 100.0]])
    # NaN cells (0,2) and (1,2) are excluded entirely; denominator = 1+1+2+2 = 6.
    expected = (1 * 1 + 2 * 1 + 4 * 2 + 6 * 2) / 6.0
    check(
        np.isclose(area_weighted_mean(values, weights), expected),
        "area_weighted_mean NaN handling wrong",
    )
    series = np.stack([values, 2.0 * values])
    means = area_weighted_mean(series, weights)
    check(means.shape == (2,), "area_weighted_mean leading shape wrong")
    check(
        np.isclose(means[1], 2.0 * means[0]),
        "area_weighted_mean time axis wrong",
    )

    # --- china_intersection_weights (requires the real shapefile) --------
    if shapefile is not None and Path(shapefile).is_file():
        result = china_intersection_weights(lat, lon, shapefile)
        frac = result["china_area_fraction"]
        inter = result["china_intersection_area_km2"]
        cell = result["gridcell_area_km2"]
        check(np.all(frac >= 0.0) and np.all(frac <= 1.0), "fraction outside [0,1]")
        check(
            np.all(inter <= cell + 1e-6),
            "intersection area exceeds gridcell area",
        )
        total_intersection = float(inter.sum())
        boundary = float(result["china_boundary_area_km2"])
        # The grid fully covers the China boundary, so the overlap areas must
        # tile the boundary area exactly (to floating-point tolerance).
        check(
            np.isclose(total_intersection, boundary, rtol=1e-6, atol=10.0),
            f"intersection total {total_intersection:.3f} != boundary "
            f"{boundary:.3f} km^2",
        )
        # Reference value for the 1-degree 73-135E / 15-54N grid.
        check(
            np.isclose(total_intersection, 9.4125e6, rtol=2e-4),
            f"intersection total {total_intersection:.3f} km^2 not ~9.4125M",
        )
        counts = result["counts"]
        check(counts["intersects"] > counts["fully_inside"], "counts inconsistent")
        check(
            counts["intersects"] == counts["fully_inside"] + counts["partial"],
            "counts do not add up",
        )
        check(counts["total"] == 40 * 63, "counts total wrong")
        # Cache hit returns the identical object.
        again = china_intersection_weights(lat, lon, shapefile)
        check(again is result, "weights cache did not hit")

    if failures:
        print("AREA WEIGHTS SELF-TEST FAILED:")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("AREA WEIGHTS SELF-TEST PASSED")
    if shapefile is not None and Path(shapefile).is_file():
        result = china_intersection_weights(lat, lon, shapefile)
        counts = result["counts"]
        print(
            "1-degree grid: "
            f"fully_inside={counts['fully_inside']}, "
            f"partial={counts['partial']}, "
            f"intersects={counts['intersects']}, "
            f"outside={counts['outside']}; "
            f"intersection total={float(result['china_intersection_area_km2'].sum()):.3f} km^2; "
            f"boundary={float(result['china_boundary_area_km2']):.3f} km^2"
        )
    return 0


if __name__ == "__main__":
    import sys as _sys

    _shapefile = None
    if len(_sys.argv) > 1:
        _shapefile = _sys.argv[1]
    else:
        try:
            from project_paths import get_path as _get_path

            _shapefile = _get_path("china_shapefile")
        except Exception:
            _shapefile = None
    _sys.exit(_run_self_test(_shapefile))
