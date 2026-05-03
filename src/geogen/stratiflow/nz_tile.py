"""NZ-tile picker for stratiflow: real DEM + urban-content filter.

Pulls a 6.4 x 6.4 km tile from the Microsoft Planetary Computer for a
chosen NZ tectonic region, filters out city/town/water-dominated tiles
using ESA WorldCover 10m land-cover, and resamples the Copernicus DEM
30m to the stratiflow 128 x 128 / 50 m grid.

The returned tile is a :class:`geogen.gis.mpc.Tile` carrying the real
DEM in meters (absolute elevation), bbox in NZTM2000, and (lon, lat)
center.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from geogen.gis.mpc import (
    DEM_COLLECTION,
    NZTM_EPSG,
    PlanetaryComputerClient,
    Tile,
    _affine_for_bbox,    # reused
    _bbox_nztm_from_center,
    _bbox_nztm_to_wgs84,
)
from geogen.gis.regions import TectonicRegion
from geogen.stratiflow.config import GenerationConfig


WORLDCOVER_COLLECTION = "esa-worldcover"
# ESA WorldCover 10m class codes
WORLDCOVER_BUILT_UP = 50
WORLDCOVER_WATER = 80
WORLDCOVER_PERMANENT_SNOW = 70


# ---------------------------------------------------------------------------
# Urban / land-cover check


def land_cover_stats(
    client: PlanetaryComputerClient,
    bbox_nztm,
) -> Dict[str, float]:
    """Return per-class fractions from ESA WorldCover for the given bbox.

    Keys: ``built_up``, ``water``, ``snow_ice``, ``vegetation``, ``rural``,
    plus ``raw_classes`` (full histogram dict).
    """
    import xarray as xr

    bbox_wgs = _bbox_nztm_to_wgs84(bbox_nztm)
    items = client._search_items_with_retry(
        collections=[WORLDCOVER_COLLECTION],
        bbox=bbox_wgs,
        max_items=8, limit=8,
    )
    if not items:
        # Treat lack of data as "unknown but not obviously urban" so the
        # caller can decide. Conservative choice: assume rural.
        return {
            "built_up": 0.0, "water": 0.0, "snow_ice": 0.0,
            "vegetation": 1.0, "rural": 1.0, "raw_classes": {},
            "no_data": True,
        }
    item = items[0]
    href = item.assets["map"].href
    da = (
        xr.open_dataarray(href, engine="rasterio", masked=True)
        .squeeze()
        .rio.clip_box(*bbox_wgs, crs="EPSG:4326")
    )
    arr = np.asarray(da.values, dtype=np.int16).ravel()
    arr = arr[arr > 0]  # strip nodata
    if arr.size == 0:
        return {
            "built_up": 0.0, "water": 0.0, "snow_ice": 0.0,
            "vegetation": 1.0, "rural": 1.0, "raw_classes": {},
            "no_data": True,
        }
    counts = np.bincount(arr.astype(np.int64))
    n = arr.size
    built_up = float(counts[WORLDCOVER_BUILT_UP] / n) if WORLDCOVER_BUILT_UP < counts.size else 0.0
    water = float(counts[WORLDCOVER_WATER] / n) if WORLDCOVER_WATER < counts.size else 0.0
    snow = float(counts[WORLDCOVER_PERMANENT_SNOW] / n) if WORLDCOVER_PERMANENT_SNOW < counts.size else 0.0
    return {
        "built_up": built_up,
        "water": water,
        "snow_ice": snow,
        "vegetation": float(1.0 - built_up - water),
        "rural": float(1.0 - built_up),
        "raw_classes": {int(k): int(v) for k, v in enumerate(counts) if v > 0},
        "no_data": False,
    }


def is_acceptable_nz_tile(
    stats: Dict[str, float],
    max_built_up: float = 0.02,
    max_water: float = 0.30,
    max_snow_ice: float = 0.50,
) -> Tuple[bool, str]:
    """Return ``(ok, reason)``. Reasons include 'urban', 'water', 'snow', 'ok'."""
    if stats.get("no_data"):
        return True, "ok_no_landcover_data"
    if stats["built_up"] > max_built_up:
        return False, f"urban (built_up={stats['built_up']:.1%})"
    if stats["water"] > max_water:
        return False, f"water (water={stats['water']:.1%})"
    if stats["snow_ice"] > max_snow_ice:
        return False, f"snow_ice (snow={stats['snow_ice']:.1%})"
    return True, "ok"


# ---------------------------------------------------------------------------
# DEM fetch at stratiflow resolution


def fetch_dem_at_grid(
    client: PlanetaryComputerClient,
    bbox_nztm,
    ny: int,
    nx: int,
    pixel_m: float,
) -> np.ndarray:
    """Return a (ny, nx) DEM array on the requested NZTM grid."""
    import xarray as xr
    from rioxarray.merge import merge_arrays

    bbox_wgs = _bbox_nztm_to_wgs84(bbox_nztm)
    items = client._search_items_with_retry(
        collections=[DEM_COLLECTION], bbox=bbox_wgs,
        max_items=12, limit=12,
    )
    if not items:
        raise RuntimeError(f"No DEM items for bbox {bbox_wgs}")
    tiles = []
    for item in items:
        href = item.assets["data"].href
        da = (
            xr.open_dataarray(href, engine="rasterio", masked=True)
            .squeeze()
            .rio.reproject(f"EPSG:{NZTM_EPSG}", resolution=30.0)
        )
        tiles.append(da)
    mosaic = tiles[0] if len(tiles) == 1 else merge_arrays(tiles)
    xmin, ymin, xmax, ymax = bbox_nztm
    clipped = mosaic.rio.clip_box(xmin, ymin, xmax, ymax)
    target = clipped.rio.reproject(
        f"EPSG:{NZTM_EPSG}",
        shape=(ny, nx),
        transform=_affine_for_bbox_at(bbox_nztm, ny, nx),
        resampling=_average_resampler(),
    )
    arr = np.asarray(target.values, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        finite = arr[np.isfinite(arr)]
        fill = float(np.median(finite)) if finite.size else 0.0
        arr = np.where(np.isfinite(arr), arr, fill)
    return arr


def _affine_for_bbox_at(bbox_nztm, ny: int, nx: int):
    """Affine transform aligning a (ny, nx) grid to a bbox."""
    from affine import Affine
    xmin, ymin, xmax, ymax = bbox_nztm
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    return Affine.translation(xmin, ymax) * Affine.scale(dx, -dy)


def _average_resampler():
    """Use bilinear; average is heavier and not always available."""
    from rasterio.enums import Resampling
    return Resampling.bilinear


# ---------------------------------------------------------------------------
# Top-level tile picker


@dataclass
class NZTileResult:
    tile: Tile                        # bbox + DEM at stratiflow resolution
    dem_yx: np.ndarray                # absolute elevation (m)
    landcover_stats: Dict[str, float]
    attempts: int


def fetch_s2_rgb_at_grid(
    client: PlanetaryComputerClient,
    bbox_nztm,
    ny: int,
    nx: int,
    date_range: str = "2023-10-01/2024-03-31",
    max_cloud: float = 15.0,
) -> Optional[np.ndarray]:
    """Fetch a (3, ny, nx) RGB Sentinel-2 composite at the stratiflow grid.

    Returns ``None`` if no clear scenes are available. Mosaics multiple
    least-cloud items if no single scene covers the bbox. Applies the
    BOA_ADD_OFFSET = -1000 for baseline >= 04.00 products.
    """
    import time as _time
    import xarray as xr
    from rioxarray.merge import merge_arrays
    from geogen.gis.mpc import S2_COLLECTION

    bbox_wgs = _bbox_nztm_to_wgs84(bbox_nztm)
    try:
        raw_items = client._search_items_with_retry(
            collections=[S2_COLLECTION],
            bbox=bbox_wgs, datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud}},
            max_items=24, limit=24,
        )
    except Exception as e:
        print(f"  [MPC] S2 search failed ({e!s:.80}); skipping satellite imagery.")
        return None
    items = sorted(raw_items, key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
    if not items:
        return None

    def _bbox_contains(outer, inner):
        return (outer[0] <= inner[0] and outer[1] <= inner[1]
                and outer[2] >= inner[2] and outer[3] >= inner[3])

    fully_covering = [it for it in items if it.bbox and _bbox_contains(it.bbox, bbox_wgs)]
    use_items = [fully_covering[0]] if fully_covering else items[: min(4, len(items))]

    def _band_offset(item, band: str) -> float:
        try:
            ras_bands = item.assets[band].extra_fields.get("raster:bands") or []
            if ras_bands and "offset" in ras_bands[0]:
                return float(ras_bands[0]["offset"])
        except Exception:
            pass
        baseline = str(item.properties.get("s2:processing_baseline", "00.00"))
        return -1000.0 if baseline >= "04.00" else 0.0

    bands = ("B04", "B03", "B02")  # R, G, B
    layers = []
    for b in bands:
        band_arrays = []
        for item in use_items:
            href = item.assets[b].href
            da = xr.open_dataarray(href, engine="rasterio", masked=True).squeeze()
            band_arrays.append(da)
            _time.sleep(0.15)
        merged = band_arrays[0] if len(band_arrays) == 1 else merge_arrays(band_arrays)
        target = merged.rio.reproject(
            f"EPSG:{NZTM_EPSG}",
            shape=(ny, nx),
            transform=_affine_for_bbox_at(bbox_nztm, ny, nx),
        )
        arr = np.asarray(target.values, dtype=np.float32)
        offset = _band_offset(use_items[0], b)
        arr = (arr + offset) / 10000.0
        layers.append(np.clip(arr, 0.0, 1.0))
    return np.stack(layers, axis=0)


def pick_nz_tile_for_stratiflow(
    region: TectonicRegion,
    cfg: GenerationConfig,
    rng: np.random.Generator,
    *,
    max_attempts: int = 30,
    margin_deg: float = 0.05,
    max_built_up: float = 0.02,
    max_water: float = 0.30,
    client: Optional[PlanetaryComputerClient] = None,
) -> NZTileResult:
    """Sample tile centers in the region until one passes the urban/water filter
    and returns it with the resampled DEM in stratiflow resolution."""
    if client is None:
        client = PlanetaryComputerClient()
    tile_size_m = cfg.nx * cfg.dx_m  # square; spec is 6400 m
    lon_min, lat_min, lon_max, lat_max = region.bbox
    rejections: list = []

    for attempt in range(1, max_attempts + 1):
        lon = float(rng.uniform(lon_min + margin_deg, lon_max - margin_deg))
        lat = float(rng.uniform(lat_min + margin_deg, lat_max - margin_deg))
        bbox = _bbox_nztm_from_center(lon, lat, size_m=tile_size_m)

        try:
            stats = land_cover_stats(client, bbox)
        except Exception as e:
            rejections.append(f"  attempt {attempt} ({lon:.3f},{lat:.3f}): worldcover error: {e!s:.80}")
            continue
        ok, reason = is_acceptable_nz_tile(
            stats, max_built_up=max_built_up, max_water=max_water,
        )
        if not ok:
            rejections.append(f"  attempt {attempt} ({lon:.3f},{lat:.3f}): {reason}")
            continue
        # Passed urban check: fetch the DEM
        try:
            dem = fetch_dem_at_grid(client, bbox, ny=cfg.ny, nx=cfg.nx, pixel_m=cfg.dx_m)
        except Exception as e:
            rejections.append(f"  attempt {attempt} ({lon:.3f},{lat:.3f}): DEM fetch failed: {e!s:.80}")
            continue
        if not np.isfinite(dem).all() or float(dem.max() - dem.min()) < 5.0:
            rejections.append(f"  attempt {attempt}: DEM degenerate (relief={float(dem.max() - dem.min()):.1f} m)")
            continue
        tile = Tile(
            region=region,
            center_lonlat=(lon, lat),
            bbox_nztm=bbox,
            dem=dem,
            pixel_size_m=cfg.dx_m,
        )
        return NZTileResult(tile=tile, dem_yx=dem, landcover_stats=stats,
                            attempts=attempt)

    raise RuntimeError(
        f"Could not find an acceptable {region.name} tile in {max_attempts} attempts.\n"
        + "\n".join(rejections[-10:])
    )
