"""
Build and serialize a sample tile that conforms to GEOSCIENCE_IO_SPEC.md.

A "sample tile" bundles, on a regular [nz, ny, nx] grid:

  * ``surface`` -- topographic elevation from satellite DEM, [ny, nx], meters
  * ``obs_z``   -- observation elevation, [ny, nx], meters (DEM + airborne offset)
  * ``target``  -- subsurface density *contrast*, [nz, ny, nx], kg/m^3
  * ``lithology`` -- categorical GeoGen rock codes, [nz, ny, nx]
  * ``gravity`` -- forward-modeled gz on the obs plane, [ny, nx], mGal
  * ``latitude`` / ``longitude`` -- per-pixel WGS84 coordinates, [ny, nx]
  * sidecar JSON -- schema_version, terrain_id, CRS, mesh, region, etc.

Coordinate convention follows the spec exactly: ``z_index = 0`` is the
shallowest (top) voxel, increasing z is downward. GeoGen natively produces
``[nx, ny, nz]`` with z=0 at the bottom, so this module performs the
necessary transpose/flip.

Subsurface stitching: the GeoGen 3.84 km vertical column is anchored such
that its top voxel's upper face equals the maximum DEM elevation in the
tile. Per-column, voxels above the local DEM elevation are masked as air
(lithology -1, density 0). Internal GeoGen air voxels below the local DEM
are forward-filled from above so the rock column is continuous.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from geogen.gis.density import lithology_to_density
from geogen.gis.gravity import forward_gz
from geogen.gis.mpc import NZTM_EPSG, TILE_SIZE_M, Tile, WGS84_EPSG

SCHEMA_VERSION = "geogen-sample-tile-v1"
DEFAULT_BACKGROUND_DENSITY_KG_M3 = 2670.0
DEFAULT_OBS_HEIGHT_M = 50.0
AIR_LITHOLOGY = -1
BASEMENT_LITHOLOGY = 0


@dataclass
class SampleTile:
    """In-memory representation of a spec-compliant sample tile."""

    surface: np.ndarray         # [ny, nx]  m
    obs_z: np.ndarray           # [ny, nx]  m
    target: np.ndarray          # [nz, ny, nx]  kg/m^3 contrast
    lithology: np.ndarray       # [nz, ny, nx]  int codes
    gravity: np.ndarray         # [ny, nx]  mGal (gz)
    latitude: np.ndarray        # [ny, nx]  degrees
    longitude: np.ndarray       # [ny, nx]  degrees
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Convenience
    @property
    def nx(self) -> int: return self.surface.shape[1]
    @property
    def ny(self) -> int: return self.surface.shape[0]
    @property
    def nz(self) -> int: return self.target.shape[0]

    def to_torch(self) -> Dict[str, Any]:
        """Return spec-shaped torch tensors (loads torch lazily)."""
        import torch
        return {
            "surface":  torch.from_numpy(self.surface).float(),
            "obs_z":    torch.from_numpy(self.obs_z).float(),
            "target":   torch.from_numpy(self.target).float(),
            "gravity":  torch.from_numpy(self.gravity).float(),
            "schema_version": SCHEMA_VERSION,
            "terrain_id": self.metadata.get("terrain_id", "sample-tile"),
        }


# ---------------------------------------------------------------------------
# Subsurface stitching: GeoGen [nx, ny, nz] -> spec [nz, ny, nx]


def _forward_fill_air_from_top(litho_xyz: np.ndarray) -> np.ndarray:
    """Replace internal air voxels with the topmost non-air rock per column.

    GeoGen marks voxels above its internal topography as -1. When we re-mask
    by a real DEM, we want to use rock-only columns; the topmost remaining
    rock layer is extrapolated upward to fill the GeoGen-air region.
    """
    out = litho_xyz.copy()
    nx, ny, nz = out.shape
    # Walk from top (z=nz-1) downward to find topmost non-air per column.
    for ix in range(nx):
        for iy in range(ny):
            col = out[ix, iy]
            non_air = np.where(col != AIR_LITHOLOGY)[0]
            if non_air.size == 0:
                col[:] = BASEMENT_LITHOLOGY
                continue
            top_rock_idx = non_air.max()
            top_rock_val = col[top_rock_idx]
            col[top_rock_idx + 1 :] = top_rock_val
    return out


def _xyz_zfromtop_to_spec(litho_xyz_filled: np.ndarray) -> np.ndarray:
    """[nx, ny, nz] (z=0 bottom) -> [nz, ny, nx] (z=0 top), per spec."""
    return np.transpose(litho_xyz_filled, (2, 1, 0))[::-1, :, :].copy()


def _mask_above_dem(
    litho_zyx: np.ndarray,
    surface_yx: np.ndarray,
    z_top_m: float,
    dz_m: float,
) -> np.ndarray:
    """Set voxels above the local DEM to AIR_LITHOLOGY."""
    nz, ny, nx = litho_zyx.shape
    # Voxel-center elevation for spec z_index k:
    z_centers = z_top_m - (np.arange(nz) + 0.5) * dz_m   # shape (nz,)
    # broadcast: (nz, 1, 1) > (1, ny, nx)
    above = z_centers[:, None, None] > surface_yx[None, :, :]
    out = litho_zyx.copy()
    out[above] = AIR_LITHOLOGY
    return out


# ---------------------------------------------------------------------------
# Geographic coordinates


def _per_pixel_lonlat(bbox_nztm, ny: int, nx: int):
    """Compute per-pixel (lon, lat) on a regular NZTM grid.

    Returns ``(lat[ny, nx], lon[ny, nx])`` in degrees (WGS84).
    """
    try:
        import pyproj
    except ImportError as e:
        raise ImportError(
            "pyproj is required for per-pixel lat/lon. "
            "Install with: pip install -e .[gis]"
        ) from e

    xmin, ymin, xmax, ymax = bbox_nztm
    # Voxel-center NZTM coordinates. Note: image y increases downward in raster
    # convention, but NZTM y increases northward; choose a consistent layout
    # (row 0 = north edge, row ny-1 = south edge).
    xs = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
    ys = ymax - (np.arange(ny) + 0.5) * (ymax - ymin) / ny
    XX, YY = np.meshgrid(xs, ys)
    transformer = pyproj.Transformer.from_crs(NZTM_EPSG, WGS84_EPSG, always_xy=True)
    lon, lat = transformer.transform(XX.ravel(), YY.ravel())
    return (
        np.asarray(lat, dtype=np.float64).reshape(ny, nx),
        np.asarray(lon, dtype=np.float64).reshape(ny, nx),
    )


# ---------------------------------------------------------------------------
# Top-level builder


def build_sample_tile(
    tile: Tile,
    lithology_xyz: np.ndarray,
    *,
    dz_m: float = 30.0,
    background_density_kg_m3: float = DEFAULT_BACKGROUND_DENSITY_KG_M3,
    observation_height_m: float = DEFAULT_OBS_HEIGHT_M,
    terrain_id: Optional[str] = None,
    density_seed: Optional[int] = None,
    skip_geographic: bool = False,
) -> SampleTile:
    """Build a spec-compliant sample tile from an MPC tile + GeoGen lithology.

    Parameters
    ----------
    tile : Tile
        Output of geogen.gis.mpc -- must contain ``dem`` of shape (ny, nx).
    lithology_xyz : np.ndarray
        GeoGen lithology grid in native [nx, ny, nz] order (z=0 at bottom),
        e.g. from ``model.get_data_grid()``.
    dz_m : float
        Vertical voxel size in meters. Default 30 to match horizontal.
    background_density_kg_m3 : float
        Reference density used for converting absolute -> contrast.
    observation_height_m : float
        Constant offset above ``surface`` for ``obs_z``. Also used as the
        observation plane elevation for the gravity forward (set at
        ``max(surface) + observation_height_m``).
    terrain_id : str, optional
        Identifier for sidecar metadata.
    density_seed : int, optional
        Seed for per-voxel density jitter (lithology -> kg/m^3).
    skip_geographic : bool
        If True, skip the lat/lon computation (offline/no-pyproj). Outputs
        ``latitude`` and ``longitude`` arrays of NaN.
    """
    # 1. Surface = real satellite DEM
    surface = np.asarray(tile.dem, dtype=np.float32)
    if surface.ndim != 2:
        raise ValueError(f"tile.dem must be 2D [ny, nx], got {surface.shape}")
    ny, nx = surface.shape
    obs_z = surface + np.float32(observation_height_m)

    # 2. Subsurface: forward-fill GeoGen internal air, transpose to spec layout
    if lithology_xyz.shape[2] == 0:
        raise ValueError("lithology_xyz has zero z-extent")
    litho_filled = _forward_fill_air_from_top(lithology_xyz.astype(np.int8))
    litho_zyx = _xyz_zfromtop_to_spec(litho_filled)
    nz = litho_zyx.shape[0]

    # 3. Anchor spec z=0 to max(DEM). Voxels above local DEM -> air.
    z_top_m = float(surface.max())
    litho_zyx = _mask_above_dem(litho_zyx, surface, z_top_m=z_top_m, dz_m=dz_m)

    # 4. Convert lithology to absolute density, then to contrast.
    abs_density = lithology_to_density(litho_zyx, seed=density_seed, jitter=True)
    # Air voxels: absolute density 0 (already zeroed by lithology_to_density)
    contrast = abs_density - np.float32(background_density_kg_m3)
    contrast[litho_zyx == AIR_LITHOLOGY] = -np.float32(background_density_kg_m3)
    target = contrast.astype(np.float32)

    # 5. Forward gravity on a constant observation plane at max(surface) + obs_h
    obs_z_const = z_top_m + observation_height_m
    pixel_m = float(tile.pixel_size_m)
    gravity = forward_gz(
        density_contrast=target,
        dx_m=pixel_m,
        dy_m=pixel_m,
        dz_m=dz_m,
        obs_z_const_m=obs_z_const,
        z_top_m=z_top_m,
    )

    # 6. Geographic coords
    if skip_geographic:
        lat = np.full((ny, nx), np.nan, dtype=np.float64)
        lon = np.full((ny, nx), np.nan, dtype=np.float64)
    else:
        lat, lon = _per_pixel_lonlat(tile.bbox_nztm, ny, nx)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "terrain_id": terrain_id or f"sample-{tile.region.name}",
        "region_name": tile.region.name,
        "region_description": tile.region.description,
        "center_lonlat": list(tile.center_lonlat),
        "bbox_nztm": list(tile.bbox_nztm),
        "crs_horizontal": f"EPSG:{NZTM_EPSG}",
        "crs_geographic": f"EPSG:{WGS84_EPSG}",
        "mesh": {
            "nz": int(nz), "ny": int(ny), "nx": int(nx),
            "dx_m": pixel_m, "dy_m": pixel_m, "dz_m": float(dz_m),
            "z_top_m": z_top_m,
            "observation_height_m": float(observation_height_m),
        },
        "physics": {
            "background_density_kg_m3": float(background_density_kg_m3),
            "obs_plane_z_m": float(obs_z_const),
            "gravity_units": "mGal",
            "gravity_sign": "positive_downward",
        },
        "source": "structural-geogis sample tile",
    }

    return SampleTile(
        surface=surface,
        obs_z=obs_z,
        target=target,
        lithology=litho_zyx,
        gravity=gravity,
        latitude=lat,
        longitude=lon,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Save / load


def save_sample_tile(sample: SampleTile, path: str | Path) -> Path:
    """Save a SampleTile as ``<path>.npz`` + sidecar ``<path>.json``.

    Returns the .npz path.
    """
    path = Path(path)
    npz_path = path.with_suffix(".npz")
    json_path = path.with_suffix(".json")

    np.savez_compressed(
        npz_path,
        surface=sample.surface,
        obs_z=sample.obs_z,
        target=sample.target,
        lithology=sample.lithology.astype(np.int8),
        gravity=sample.gravity,
        latitude=sample.latitude,
        longitude=sample.longitude,
    )
    with open(json_path, "w") as f:
        json.dump(sample.metadata, f, indent=2)
    return npz_path


def load_sample_tile(path: str | Path) -> SampleTile:
    """Inverse of save_sample_tile."""
    path = Path(path)
    npz_path = path.with_suffix(".npz")
    json_path = path.with_suffix(".json")
    with np.load(npz_path) as z:
        arrays = {k: z[k] for k in z.files}
    metadata: Dict[str, Any] = {}
    if json_path.exists():
        with open(json_path) as f:
            metadata = json.load(f)
    return SampleTile(
        surface=arrays["surface"],
        obs_z=arrays["obs_z"],
        target=arrays["target"],
        lithology=arrays["lithology"],
        gravity=arrays["gravity"],
        latitude=arrays["latitude"],
        longitude=arrays["longitude"],
        metadata=metadata,
    )
