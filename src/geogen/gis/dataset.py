"""
PyTorch dataset that streams imagery-conditioned synthetic geology samples.

Each sample is a NZ tile drawn from MPC + a single Markov realization of a
plausible subsurface, biased by the tile's tectonic setting and DEM/S2
features. The dataset is meant for ML training, so it can either:

  * pre-fetch and cache a small set of tiles and then stream many
    realizations per tile (cheap, lots of variety per fetch), or
  * fetch a fresh tile on every call (expensive; rate-limited).

Default behavior is to pre-fetch ``n_tiles`` tiles once and then stream
realizations off them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from geogen.gis.conditioning import generator_for_tile
from geogen.gis.density import lithology_to_density
from geogen.gis.features import TileFeatures, extract_features
from geogen.gis.mpc import Tile, fetch_tiles
from geogen.gis.regions import NZ_REGIONS, TectonicRegion


@dataclass
class GISConditionedSample:
    """One streamed training example."""

    dem: torch.Tensor              # (1, 256, 256) elevation in meters
    lithology: torch.Tensor        # (1, 256, 256, 128) categorical codes
    density: torch.Tensor          # (1, 256, 256, 128) kg/m^3
    region_name: str
    features: TileFeatures


class NZGISConditionedDataset(Dataset):
    """Imagery-conditioned synthetic structural-geology dataset for NZ.

    Parameters
    ----------
    regions : sequence of TectonicRegion, optional
        Tectonic settings to sample from. Defaults to ``NZ_REGIONS``.
    tiles_per_region : int
        Number of distinct MPC tiles to fetch per region (cached on init).
    realizations_per_tile : int
        Number of synthetic subsurfaces returned per cached tile, controls
        the effective dataset epoch length: ``len(tiles) * realizations_per_tile``.
    with_s2 : bool
        Whether to fetch a Sentinel-2 composite (enables NDVI feature).
    model_bounds, model_resolution : tuple
        Forwarded to GeoGen. Defaults match the 7.68 x 7.68 x 3.84 km tile.
    seed : int, optional
        Seed for tile-center sampling and density jitter.
    device : str
        Torch device for the returned tensors.
    """

    def __init__(
        self,
        regions: Optional[Sequence[TectonicRegion]] = None,
        tiles_per_region: int = 2,
        realizations_per_tile: int = 64,
        with_s2: bool = True,
        model_bounds=((-3840, 3840), (-3840, 3840), (-1920, 1920)),
        model_resolution=(256, 256, 128),
        seed: Optional[int] = None,
        device: str = "cpu",
        prefetch: bool = True,
    ):
        self.regions = list(regions) if regions is not None else list(NZ_REGIONS)
        self.tiles_per_region = tiles_per_region
        self.realizations_per_tile = realizations_per_tile
        self.with_s2 = with_s2
        self.model_bounds = model_bounds
        self.model_resolution = model_resolution
        self.seed = seed
        self.device = device

        self._tiles: list[Tile] = []
        self._features: list[TileFeatures] = []
        if prefetch:
            self._prefetch()

    # ---- Tile cache ---------------------------------------------------------

    def _prefetch(self):
        self._tiles = fetch_tiles(
            self.regions,
            tiles_per_region=self.tiles_per_region,
            with_s2=self.with_s2,
            seed=self.seed,
        )
        self._features = [extract_features(t) for t in self._tiles]

    def add_tile(self, tile: Tile, features: Optional[TileFeatures] = None):
        """Inject a pre-built tile (useful for tests / offline use)."""
        self._tiles.append(tile)
        self._features.append(features or extract_features(tile))

    # ---- Dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        return max(1, len(self._tiles)) * self.realizations_per_tile

    def __getitem__(self, idx: int) -> GISConditionedSample:
        if not self._tiles:
            raise RuntimeError(
                "No tiles loaded. Call _prefetch() or add_tile() first."
            )
        tile_idx = (idx // self.realizations_per_tile) % len(self._tiles)
        tile = self._tiles[tile_idx]
        feats = self._features[tile_idx]

        gen = generator_for_tile(
            tile,
            features=feats,
            model_bounds=self.model_bounds,
            model_resolution=self.model_resolution,
        )
        model = gen.generate_model()
        model.fill_nans()
        litho = model.get_data_grid().astype(np.int8)
        density = lithology_to_density(litho, seed=idx + (self.seed or 0))

        return GISConditionedSample(
            dem=torch.from_numpy(tile.dem).float().unsqueeze(0).to(self.device),
            lithology=torch.from_numpy(litho).long().unsqueeze(0).to(self.device),
            density=torch.from_numpy(density).float().unsqueeze(0).to(self.device),
            region_name=tile.region.name,
            features=feats,
        )
