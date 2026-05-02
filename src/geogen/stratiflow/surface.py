"""Surface DEM extraction + surface contact raster + GeoTIFF writer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from geogen.stratiflow.lithology import AIR_CODE


def extract_dem_from_volume(
    lithology_zyx: np.ndarray,
    dz_m: float,
    z_top_m: float = 0.0,
) -> np.ndarray:
    """Top-of-column elevation (meters) per (y, x) pixel.

    For each column, find the shallowest non-air voxel (smallest z_index)
    and convert to elevation. Pixels that are all air return ``z_top_m``
    (treated as flat top).
    """
    nz, ny, nx = lithology_zyx.shape
    air_mask = lithology_zyx == AIR_CODE
    rock_mask = ~air_mask
    # For each (y, x) column, top non-air index = argmax along z.
    has_rock = rock_mask.any(axis=0)
    top_idx = np.argmax(rock_mask, axis=0).astype(np.int32)
    # Voxel top face elevation = z_top_m - top_idx * dz_m
    dem = z_top_m - top_idx.astype(np.float32) * dz_m
    dem[~has_rock] = z_top_m
    return dem.astype(np.float32)


def apply_topographic_perturbation(
    lithology_zyx: np.ndarray,
    dz_m: float,
    amplitude_m: float,
    rng: np.random.Generator,
    z_top_m: float = 0.0,
    n_harmonics: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stamp non-trivial relief onto a (possibly already air-topped) geomodel.

    Finds the current top-of-rock per column, then trims an additional N
    rock voxels off the top to imprint a relief pattern with peak-to-peak
    amplitude ~= 2 * amplitude_m. Returns ``(new_lithology_zyx, new_dem_yx)``.
    """
    nz, ny, nx = lithology_zyx.shape

    # Current top-of-rock index per column (z=0 at top).
    rock_mask = lithology_zyx != AIR_CODE
    has_rock = rock_mask.any(axis=0)
    cur_top_idx = np.argmax(rock_mask, axis=0).astype(np.int32)

    # Build a low-frequency 2D relief field in [-amplitude_m, +amplitude_m].
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    domain_x = float(nx)
    domain_y = float(ny)
    relief = np.zeros((ny, nx), dtype=np.float32)
    for _ in range(n_harmonics):
        wlx = rng.uniform(0.4, 0.9) * domain_x
        wly = rng.uniform(0.4, 0.9) * domain_y
        ph_x = rng.uniform(0.0, 2.0 * np.pi)
        ph_y = rng.uniform(0.0, 2.0 * np.pi)
        relief += (
            np.sin(2 * np.pi * xx / wlx + ph_x)
            * np.sin(2 * np.pi * yy / wly + ph_y)
        )
    rng_max = float(np.max(np.abs(relief))) + 1e-9
    relief = (relief / rng_max) * amplitude_m

    # Per-column extra rock voxels to TRIM off the top, in [0, 2*amp/dz].
    extra_air = np.maximum(
        0, np.round((relief.max() - relief) / dz_m)
    ).astype(np.int32)

    new_litho = lithology_zyx.copy()
    for j in range(ny):
        for i in range(nx):
            if not has_rock[j, i]:
                continue
            top = int(cur_top_idx[j, i])
            new_top = min(nz, top + int(extra_air[j, i]))
            if new_top > top:
                new_litho[top:new_top, j, i] = AIR_CODE

    dem = extract_dem_from_volume(new_litho, dz_m=dz_m, z_top_m=z_top_m)
    return new_litho, dem


def surface_contact_raster(lithology_zyx: np.ndarray) -> np.ndarray:
    """[ny, nx] int32 of the topmost non-air lithology code at each pixel.

    Pixels that are all-air get ``-1``.
    """
    rock_mask = lithology_zyx != AIR_CODE
    has_rock = rock_mask.any(axis=0)
    top_idx = np.argmax(rock_mask, axis=0).astype(np.int32)
    ny, nx = lithology_zyx.shape[1:]
    out = np.full((ny, nx), -1, dtype=np.int32)
    j, i = np.indices((ny, nx))
    out[has_rock] = lithology_zyx[top_idx[has_rock], j[has_rock], i[has_rock]].astype(np.int32)
    return out


def degrade_surface_map(
    contact: np.ndarray,
    unmapped_fraction: float,
    error_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply unmapped-pixel masking + adjacent-unit confusion to simulate
    real-world surface mapping. No-op if both fractions are 0."""
    if unmapped_fraction <= 0.0 and error_fraction <= 0.0:
        return contact.copy()
    out = contact.copy()
    ny, nx = contact.shape
    n = ny * nx
    flat = out.reshape(-1)

    if unmapped_fraction > 0.0:
        n_unmapped = int(round(unmapped_fraction * n))
        idx = rng.choice(n, size=n_unmapped, replace=False)
        flat[idx] = -1

    if error_fraction > 0.0:
        n_err = int(round(error_fraction * n))
        idx = rng.choice(n, size=n_err, replace=False)
        codes_present = [c for c in np.unique(out) if c >= 0]
        if len(codes_present) >= 2:
            for k in idx:
                cur = flat[k]
                if cur < 0:
                    continue
                # swap to a neighbor code
                others = [c for c in codes_present if c != cur]
                flat[k] = rng.choice(others)
    return out


def write_dem_geotiff(
    dem_yx: np.ndarray,
    out_path: Path,
    dx_m: float,
    dy_m: float,
    origin_xy: Tuple[float, float] = (0.0, 0.0),
    crs_epsg: int = 32760,  # synthetic local UTM-like CRS
) -> Optional[Path]:
    """Save a DEM as GeoTIFF. Returns the path on success or None if rasterio
    is unavailable (don't make rasterio a hard requirement for offline use)."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(origin_xy[0], origin_xy[1] + dy_m * dem_yx.shape[0], dx_m, dy_m)
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=dem_yx.shape[0], width=dem_yx.shape[1],
        count=1, dtype="float32",
        crs=f"EPSG:{crs_epsg}", transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(dem_yx.astype(np.float32), 1)
    return out_path
