"""
Per-tile structural-geology features extracted from DEM + Sentinel-2.

These features summarize the surface expression of the underlying tectonic
fabric (relief, slope statistics, ridge/valley lineament fabric, vegetation
cover proxy) in a small fixed-size feature dict that downstream
conditioning code can map to GeoGen Markov priors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from geogen.gis.mpc import Tile


@dataclass
class TileFeatures:
    relief_m: float          # max - min elevation
    mean_elev_m: float
    slope_mean_deg: float
    slope_std_deg: float
    roughness: float         # std of laplacian, normalized
    lineament_strength: float  # [0, 1], anisotropy of gradient orientation
    lineament_azimuth_deg: float  # [0, 180), dominant strike (NZTM north = 0)
    ndvi_mean: Optional[float] = None
    ndvi_std: Optional[float] = None
    extras: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        d = {
            "relief_m": self.relief_m,
            "mean_elev_m": self.mean_elev_m,
            "slope_mean_deg": self.slope_mean_deg,
            "slope_std_deg": self.slope_std_deg,
            "roughness": self.roughness,
            "lineament_strength": self.lineament_strength,
            "lineament_azimuth_deg": self.lineament_azimuth_deg,
        }
        if self.ndvi_mean is not None:
            d["ndvi_mean"] = self.ndvi_mean
            d["ndvi_std"] = self.ndvi_std
        d.update(self.extras)
        return d


def _slope_deg(dem: np.ndarray, pixel_m: float) -> np.ndarray:
    """Per-pixel slope magnitude in degrees."""
    gy, gx = np.gradient(dem.astype(np.float32), pixel_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def _roughness(dem: np.ndarray) -> float:
    """Std of the discrete laplacian, normalized by relief."""
    lap = (
        -4.0 * dem
        + np.roll(dem, 1, 0)
        + np.roll(dem, -1, 0)
        + np.roll(dem, 1, 1)
        + np.roll(dem, -1, 1)
    )
    rng = float(dem.max() - dem.min())
    return float(lap.std() / rng) if rng > 1e-6 else 0.0


def _gradient_orientation_stats(dem: np.ndarray, pixel_m: float):
    """Strength + azimuth of the dominant gradient orientation.

    Uses the doubled-angle structure tensor: ridges and valleys (which differ
    by 180 deg) reinforce instead of cancel, so the result captures the
    *strike* of linear topographic features (faults/folds/foliation).
    Returns (strength in [0, 1], azimuth in [0, 180) deg from grid-north).
    """
    gy, gx = np.gradient(dem.astype(np.float32), pixel_m)
    mag = np.hypot(gx, gy)
    if mag.max() <= 1e-9:
        return 0.0, 0.0
    mask = mag > np.percentile(mag, 60)  # focus on lineament-bearing pixels
    if not mask.any():
        # Degenerate case: gradient magnitude nearly uniform. Use everything.
        mask = np.ones_like(mag, dtype=bool)
    # Direction perpendicular to gradient = strike of the linear feature
    strike = np.arctan2(gx, gy)  # angle from +y (grid-north), [-pi, pi]
    # Double the angle so opposite directions sum coherently
    c = np.cos(2.0 * strike[mask]).mean()
    s = np.sin(2.0 * strike[mask]).mean()
    strength = float(np.hypot(c, s))  # 0=isotropic, 1=perfectly aligned
    azimuth = (np.degrees(0.5 * np.arctan2(s, c)) + 180.0) % 180.0
    return strength, float(azimuth)


def _ndvi_stats(s2_rgbnir: np.ndarray):
    red = s2_rgbnir[0]
    nir = s2_rgbnir[3]
    denom = nir + red
    ndvi = np.where(denom > 1e-4, (nir - red) / denom, 0.0)
    return float(ndvi.mean()), float(ndvi.std())


def extract_features(tile: Tile) -> TileFeatures:
    dem = tile.dem
    slope = _slope_deg(dem, tile.pixel_size_m)
    strength, azimuth = _gradient_orientation_stats(dem, tile.pixel_size_m)

    ndvi_mean = ndvi_std = None
    if tile.s2_rgbnir is not None:
        ndvi_mean, ndvi_std = _ndvi_stats(tile.s2_rgbnir)

    return TileFeatures(
        relief_m=float(dem.max() - dem.min()),
        mean_elev_m=float(dem.mean()),
        slope_mean_deg=float(slope.mean()),
        slope_std_deg=float(slope.std()),
        roughness=_roughness(dem),
        lineament_strength=strength,
        lineament_azimuth_deg=azimuth,
        ndvi_mean=ndvi_mean,
        ndvi_std=ndvi_std,
    )
