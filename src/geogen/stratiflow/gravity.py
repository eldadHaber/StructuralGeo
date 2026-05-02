"""Forward gravity for the stratiflow synthetic dataset.

Reuses the FFT-per-layer point-mass forward operator from
:mod:`geogen.gis.gravity` (the spec's "alternative" to harmonica.prism_gravity).
Adds station noise + a low-frequency regional trend per spec.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import numpy as np

from geogen.gis.gravity import forward_gz
from geogen.stratiflow.config import GenerationConfig


def forward_gravity_grid(
    density_contrast_zyx: np.ndarray,
    dem_yx: np.ndarray,
    cfg: GenerationConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Compute the surface gravity grid + obs_z grid.

    Returns ``(gravity_mgal, obs_z_m, regional_trend_meta)``.
    """
    # Observation plane is constant at max(DEM) + obs_height (FFT-friendly).
    z_top_m = 0.0
    obs_z_const = float(dem_yx.max()) + cfg.obs_height_m
    gz = forward_gz(
        density_contrast=density_contrast_zyx,
        dx_m=cfg.dx_m, dy_m=cfg.dy_m, dz_m=cfg.dz_m,
        obs_z_const_m=obs_z_const, z_top_m=z_top_m,
    )

    # Add a low-frequency regional trend (linear or quadratic over the domain)
    ny, nx = gz.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    # Normalize coords to [-1, 1] for clean trend coefficients
    u = (xx - (nx - 1) / 2) / max(1.0, (nx - 1) / 2)
    v = (yy - (ny - 1) / 2) / max(1.0, (ny - 1) / 2)
    a = float(rng.uniform(-1.0, 1.0))
    b = float(rng.uniform(-1.0, 1.0))
    c = float(rng.uniform(-0.5, 0.5))
    norm = np.sqrt(a * a + b * b + c * c) or 1.0
    a, b, c = a / norm, b / norm, c / norm
    raw_trend = a * u + b * v + c * (u * u + v * v - 1.0)
    trend = (cfg.gravity_regional_trend_amplitude_mgal * raw_trend).astype(np.float32)
    gz_with_trend = gz + trend

    # Station noise
    noise = rng.normal(0.0, cfg.gravity_station_noise_mgal, gz.shape).astype(np.float32)
    gz_noisy = gz_with_trend + noise

    obs_z = (dem_yx + cfg.obs_height_m).astype(np.float32)

    meta = {
        "obs_plane_z_m": float(obs_z_const),
        "regional_trend": {
            "kind": "linear+quadratic_isotropic",
            "amplitude_mgal": float(cfg.gravity_regional_trend_amplitude_mgal),
            "coefficients": {"a_u": a, "b_v": b, "c_radial": c},
        },
        "station_noise_mgal_sigma": float(cfg.gravity_station_noise_mgal),
    }
    return gz_noisy.astype(np.float32), obs_z, meta


def write_gravity_stations_csv(
    grid_mgal: np.ndarray,
    dem_yx: np.ndarray,
    cfg: GenerationConfig,
    rng: np.random.Generator,
    out_path: Path,
) -> Path:
    """Write a perturbed station-form CSV. Stations are jittered ±20 m
    horizontal and ±5 m vertical relative to grid centers, with extra
    0.05 mGal noise on top of the grid noise."""
    ny, nx = grid_mgal.shape
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x_m", "y_m", "z_m", "gravity_mgal"])
        for j in range(ny):
            for i in range(nx):
                x_c = (i + 0.5) * cfg.dx_m + float(rng.uniform(-20.0, 20.0))
                y_c = (j + 0.5) * cfg.dy_m + float(rng.uniform(-20.0, 20.0))
                z_c = float(dem_yx[j, i] + cfg.obs_height_m + rng.uniform(-5.0, 5.0))
                g = float(grid_mgal[j, i] + rng.normal(0.0, 0.05))
                w.writerow([
                    f"GV-{j * nx + i + 1:05d}",
                    round(x_c, 2), round(y_c, 2), round(z_c, 2),
                    round(g, 4),
                ])
    return out_path
