"""Feature extraction on synthetic DEMs."""

import numpy as np

from geogen.gis.features import extract_features
from geogen.gis.mpc import TILE_SIZE_M, Tile
from geogen.gis.regions import NZ_REGIONS


def _make_tile(dem, s2=None):
    return Tile(
        region=NZ_REGIONS[0],
        center_lonlat=(170.0, -43.5),
        bbox_nztm=(0.0, 0.0, TILE_SIZE_M, TILE_SIZE_M),
        dem=dem,
        s2_rgbnir=s2,
    )


def test_flat_dem_has_zero_relief_and_low_slope():
    dem = np.full((64, 64), 100.0, dtype=np.float32)
    f = extract_features(_make_tile(dem))
    assert f.relief_m == 0.0
    assert f.slope_mean_deg < 0.1
    assert f.lineament_strength <= 1.0


def test_inclined_plane_has_consistent_slope_and_orientation():
    """A planar tilt to the east should give a non-trivial slope and a
    near-N-S striking dominant orientation (perpendicular to the gradient)."""
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    dem = xx * 30.0  # 30 m rise per pixel = pure east-dipping plane
    f = extract_features(_make_tile(dem))
    assert f.slope_mean_deg > 30.0  # steep
    assert f.lineament_strength > 0.95  # near-perfectly aligned


def test_ridge_field_recovers_strike_orientation():
    """Parallel ridges striking ~ NE-SW should give a strong oriented signal."""
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    # Ridges with strike at 45 degrees from grid-north
    dem = 100.0 * np.sin((xx + yy) * 0.2)
    f = extract_features(_make_tile(dem))
    assert f.lineament_strength > 0.5
    # Ridges along x+y=const have strike at 45 deg from grid-N
    # (allow some tolerance because of discretization)
    az = f.lineament_azimuth_deg
    assert min(abs(az - 45.0), abs(az - 135.0)) < 15.0


def test_ndvi_computed_when_s2_present():
    dem = np.zeros((32, 32), dtype=np.float32)
    s2 = np.zeros((4, 32, 32), dtype=np.float32)
    s2[0] = 0.1   # red
    s2[3] = 0.6   # nir => high NDVI
    f = extract_features(_make_tile(dem, s2))
    assert f.ndvi_mean is not None
    assert f.ndvi_mean > 0.5
