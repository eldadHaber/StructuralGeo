"""
Microsoft Planetary Computer client for sampling 7.68 x 7.68 km tiles.

Pulls a Copernicus DEM 30 m elevation grid and (optionally) a cloud-masked
Sentinel-2 L2A composite for each tile. All raster output is reprojected
to NZ Transverse Mercator 2000 (EPSG:2193) so that a 7.68 km tile maps
cleanly onto a 256x256 grid at 30 m spacing -- matching GeoGen's default
horizontal resolution.

External dependencies (lazy-imported so the rest of geogen stays usable
without GIS extras): ``pystac-client``, ``planetary-computer``, ``rioxarray``,
``pyproj``, ``shapely``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from geogen.gis.regions import TectonicRegion

# 7.68 km tile at 30 m -> 256 x 256, matches GeoGen default x/y resolution.
TILE_SIZE_M = 7680.0
DEM_PIXEL_M = 30.0
NZTM_EPSG = 2193  # NZGD2000 / New Zealand Transverse Mercator 2000
WGS84_EPSG = 4326

DEM_COLLECTION = "cop-dem-glo-30"
S2_COLLECTION = "sentinel-2-l2a"


@dataclass
class Tile:
    """A sampled tile with co-registered DEM and (optional) Sentinel-2 bands."""

    region: TectonicRegion
    center_lonlat: Tuple[float, float]
    bbox_nztm: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    dem: np.ndarray  # shape (H, W), meters
    s2_rgbnir: Optional[np.ndarray] = None  # shape (4, H, W), reflectance 0..1
    pixel_size_m: float = DEM_PIXEL_M

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return self.dem.shape


# ---------------------------------------------------------------------------
# Lazy import helpers


def _require(modname: str):
    try:
        return __import__(modname)
    except ImportError as exc:
        raise ImportError(
            f"geogen.gis requires '{modname}'. Install GIS extras with:\n"
            "    pip install pystac-client planetary-computer rioxarray pyproj shapely"
        ) from exc


def _imports():
    """Resolve all GIS deps in one place."""
    pystac_client = _require("pystac_client")
    planetary_computer = _require("planetary_computer")
    rioxarray = _require("rioxarray")  # noqa: F841 - registers .rio accessor
    pyproj = _require("pyproj")
    shapely_geom = _require("shapely.geometry")
    import xarray  # rioxarray dep, must be present
    return pystac_client, planetary_computer, pyproj, shapely_geom, xarray


# ---------------------------------------------------------------------------
# Sampling


def sample_tile_centers(
    region: TectonicRegion,
    n_tiles: int,
    seed: Optional[int] = None,
    margin_deg: float = 0.05,
) -> np.ndarray:
    """Draw uniformly random (lon, lat) tile centers inside a region bbox."""
    rng = np.random.default_rng(seed)
    lon_min, lat_min, lon_max, lat_max = region.bbox
    lons = rng.uniform(lon_min + margin_deg, lon_max - margin_deg, size=n_tiles)
    lats = rng.uniform(lat_min + margin_deg, lat_max - margin_deg, size=n_tiles)
    return np.column_stack([lons, lats])


def _bbox_nztm_from_center(lon: float, lat: float, size_m: float = TILE_SIZE_M):
    """Return an axis-aligned bbox in NZTM meters around a lon/lat center."""
    _, _, pyproj, _, _ = _imports()
    transformer = pyproj.Transformer.from_crs(WGS84_EPSG, NZTM_EPSG, always_xy=True)
    cx, cy = transformer.transform(lon, lat)
    half = size_m / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def _bbox_nztm_to_wgs84(bbox_nztm) -> Tuple[float, float, float, float]:
    _, _, pyproj, _, _ = _imports()
    t = pyproj.Transformer.from_crs(NZTM_EPSG, WGS84_EPSG, always_xy=True)
    xmin, ymin, xmax, ymax = bbox_nztm
    lon_a, lat_a = t.transform(xmin, ymin)
    lon_b, lat_b = t.transform(xmax, ymax)
    lon_c, lat_c = t.transform(xmin, ymax)
    lon_d, lat_d = t.transform(xmax, ymin)
    return (
        min(lon_a, lon_b, lon_c, lon_d),
        min(lat_a, lat_b, lat_c, lat_d),
        max(lon_a, lon_b, lon_c, lon_d),
        max(lat_a, lat_b, lat_c, lat_d),
    )


# ---------------------------------------------------------------------------
# STAC + raster loading


class PlanetaryComputerClient:
    """Thin wrapper around pystac-client for the Microsoft Planetary Computer.

    Uses ``planetary_computer.sign`` to attach SAS tokens to asset URLs so
    anonymous access works for public collections (DEM, Sentinel-2 L2A).
    """

    STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    # MPC occasionally returns 5xx; retry handful of times before giving up.
    MAX_RETRIES = 4
    RETRY_BACKOFF_S = 2.0
    REQUEST_TIMEOUT_S = 30.0

    def __init__(self):
        pystac_client, planetary_computer, _, _, _ = _imports()
        self._pc = planetary_computer
        # Configure StacApiIO with a tighter timeout so transient hangs fail
        # fast and we can retry instead of waiting on the gateway.
        try:
            stac_io = pystac_client.stac_api_io.StacApiIO(timeout=self.REQUEST_TIMEOUT_S)
            self._client = pystac_client.Client.open(
                self.STAC_URL,
                modifier=planetary_computer.sign_inplace,
                stac_io=stac_io,
            )
        except TypeError:
            # Older pystac-client may not accept stac_io / timeout
            self._client = pystac_client.Client.open(
                self.STAC_URL, modifier=planetary_computer.sign_inplace
            )

    def _search_items_with_retry(self, **kwargs):
        """Run a STAC search with bounded retry/backoff on transient 5xx errors."""
        import time
        pystac_client = sys.modules.get("pystac_client") or _imports()[0]
        APIError = pystac_client.exceptions.APIError

        last_err = None
        for attempt in range(self.MAX_RETRIES):
            try:
                search = self._client.search(**kwargs)
                return list(search.items())
            except APIError as e:
                last_err = e
                msg = str(e).lower()
                # Retry only on transient signals; raise hard on auth/4xx
                transient = any(s in msg for s in (
                    "exceeded the maximum allowed time",
                    "timeout", "timed out",
                    "502", "503", "504",
                    "gateway", "temporarily",
                ))
                if not transient or attempt == self.MAX_RETRIES - 1:
                    raise
                wait = self.RETRY_BACKOFF_S * (2 ** attempt)
                print(
                    f"  [MPC] transient error ({e!s:.120}); "
                    f"retry {attempt + 1}/{self.MAX_RETRIES - 1} in {wait:.0f}s..."
                )
                time.sleep(wait)
        raise last_err  # unreachable, but mypy-friendly

    # ---- DEM ----------------------------------------------------------------

    def fetch_dem(self, bbox_nztm) -> np.ndarray:
        """Return a (H, W) DEM array on a 256x256 NZTM grid at 30 m spacing."""
        import rioxarray  # noqa: F401
        import xarray as xr

        bbox_wgs = _bbox_nztm_to_wgs84(bbox_nztm)
        # Cap result count: a 7.68 km tile overlaps at most 1-4 Cop-DEM-30 items.
        items = self._search_items_with_retry(
            collections=[DEM_COLLECTION], bbox=bbox_wgs, max_items=12, limit=12,
        )
        if not items:
            raise RuntimeError(f"No DEM items returned for bbox {bbox_wgs}")

        tiles = []
        for item in items:
            href = item.assets["data"].href
            da = (
                xr.open_dataarray(href, engine="rasterio", masked=True)
                .squeeze()
                .rio.reproject(f"EPSG:{NZTM_EPSG}", resolution=DEM_PIXEL_M)
            )
            tiles.append(da)

        # Mosaic if multiple items overlap
        if len(tiles) == 1:
            mosaic = tiles[0]
        else:
            from rioxarray.merge import merge_arrays

            mosaic = merge_arrays(tiles)

        xmin, ymin, xmax, ymax = bbox_nztm
        clipped = mosaic.rio.clip_box(xmin, ymin, xmax, ymax)
        # Resample to exactly 256x256 to guarantee shape regardless of pixel alignment
        target = clipped.rio.reproject(
            f"EPSG:{NZTM_EPSG}",
            shape=(int(TILE_SIZE_M / DEM_PIXEL_M), int(TILE_SIZE_M / DEM_PIXEL_M)),
            transform=_affine_for_bbox(bbox_nztm),
        )
        arr = np.asarray(target.values, dtype=np.float32)
        # Replace nodata with nearest-finite or zero
        if not np.all(np.isfinite(arr)):
            arr = np.where(np.isfinite(arr), arr, np.nanmedian(arr))
        return arr

    # ---- Sentinel-2 ---------------------------------------------------------

    def fetch_s2_rgbnir(
        self,
        bbox_nztm,
        date_range: str = "2023-10-01/2024-03-31",
        max_cloud: float = 15.0,
    ) -> Optional[np.ndarray]:
        """Return a (4, H, W) RGB+NIR composite or None if no clear scenes."""
        import xarray as xr  # noqa: F401

        bbox_wgs = _bbox_nztm_to_wgs84(bbox_nztm)
        try:
            raw_items = self._search_items_with_retry(
                collections=[S2_COLLECTION],
                bbox=bbox_wgs,
                datetime=date_range,
                query={"eo:cloud_cover": {"lt": max_cloud}},
                max_items=24, limit=24,
            )
        except Exception as e:
            print(f"  [MPC] Sentinel-2 search failed ({e!s:.120}); skipping S2.")
            return None
        items = sorted(
            raw_items, key=lambda it: it.properties.get("eo:cloud_cover", 100.0),
        )
        if not items:
            return None

        # Items processed with baseline >= 04.00 (post 2022-01-25) carry an
        # additive offset that must be subtracted before scaling. The offset
        # is item-level metadata; default of -1000 is the documented value.
        def _band_offset(item, band: str) -> float:
            try:
                ras_bands = item.assets[band].extra_fields.get("raster:bands") or []
                if ras_bands and "offset" in ras_bands[0]:
                    return float(ras_bands[0]["offset"])
            except Exception:
                pass
            baseline = str(item.properties.get("s2:processing_baseline", "00.00"))
            return -1000.0 if baseline >= "04.00" else 0.0

        # Prefer items whose footprint *fully contains* the requested bbox;
        # otherwise mosaic several least-cloud items so partial coverage is
        # filled by neighboring scenes instead of becoming a flat-color blob.
        def _bbox_contains(outer, inner):
            return (outer[0] <= inner[0] and outer[1] <= inner[1]
                    and outer[2] >= inner[2] and outer[3] >= inner[3])

        fully_covering = [it for it in items if it.bbox and _bbox_contains(it.bbox, bbox_wgs)]
        if fully_covering:
            use_items = [fully_covering[0]]
        else:
            use_items = items[: min(4, len(items))]
            if len(use_items) > 1:
                print(f"  [MPC] No single S2 scene covers the tile; "
                      f"mosaicking {len(use_items)} least-cloud items.")

        from rioxarray.merge import merge_arrays
        import time

        bands = ("B04", "B03", "B02", "B08")  # R, G, B, NIR
        layers = []
        for b in bands:
            band_arrays = []
            for item in use_items:
                href = item.assets[b].href
                da = xr.open_dataarray(href, engine="rasterio", masked=True).squeeze()
                band_arrays.append(da)
                # Be polite to MPC: small inter-request sleep avoids 429s on
                # multi-band, multi-scene fetches without slowing things much.
                time.sleep(0.15)

            if len(band_arrays) > 1:
                merged = merge_arrays(band_arrays)
            else:
                merged = band_arrays[0]

            target = merged.rio.reproject(
                f"EPSG:{NZTM_EPSG}",
                shape=(int(TILE_SIZE_M / DEM_PIXEL_M), int(TILE_SIZE_M / DEM_PIXEL_M)),
                transform=_affine_for_bbox(bbox_nztm),
            )
            arr = np.asarray(target.values, dtype=np.float32)
            offset = _band_offset(use_items[0], b)
            arr = (arr + offset) / 10000.0
            # Keep NaN where there's no data; viz handles it (don't median-fill,
            # which paints the no-data region a flat colour and looks like a bug).
            layers.append(np.clip(arr, 0.0, 1.0))
        return np.stack(layers, axis=0)

    # ---- High-level ---------------------------------------------------------

    def fetch_tile(
        self,
        region: TectonicRegion,
        center_lonlat: Tuple[float, float],
        with_s2: bool = True,
    ) -> Tile:
        bbox = _bbox_nztm_from_center(*center_lonlat)
        dem = self.fetch_dem(bbox)
        s2 = self.fetch_s2_rgbnir(bbox) if with_s2 else None
        return Tile(
            region=region,
            center_lonlat=center_lonlat,
            bbox_nztm=bbox,
            dem=dem,
            s2_rgbnir=s2,
        )


def _affine_for_bbox(bbox_nztm):
    """Affine transform aligning a (256, 256) array to a 7680 m bbox."""
    from affine import Affine

    xmin, ymin, xmax, ymax = bbox_nztm
    nx = int(round((xmax - xmin) / DEM_PIXEL_M))
    ny = int(round((ymax - ymin) / DEM_PIXEL_M))
    return Affine.translation(xmin, ymax) * Affine.scale(DEM_PIXEL_M, -DEM_PIXEL_M) * Affine.scale(
        nx / int(TILE_SIZE_M / DEM_PIXEL_M), ny / int(TILE_SIZE_M / DEM_PIXEL_M)
    )


def fetch_tiles(
    regions: Sequence[TectonicRegion],
    tiles_per_region: int = 1,
    with_s2: bool = True,
    seed: Optional[int] = None,
) -> list[Tile]:
    """Convenience: sample N tiles per region from MPC."""
    client = PlanetaryComputerClient()
    rng = np.random.default_rng(seed)
    out: list[Tile] = []
    for region in regions:
        centers = sample_tile_centers(
            region, tiles_per_region, seed=int(rng.integers(0, 2**31 - 1))
        )
        for lon, lat in centers:
            out.append(client.fetch_tile(region, (float(lon), float(lat)), with_s2=with_s2))
    return out
