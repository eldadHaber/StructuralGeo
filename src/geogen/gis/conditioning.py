"""
Map per-tile structural features to GeoGen Markov-chain priors.

The pipeline:
  1. Start from the region's a-priori category weights (tectonic setting).
  2. Modulate them with features extracted from the DEM/Sentinel-2 tile
     (relief, slope, lineament fabric, NDVI).
  3. Build a biased :class:`MarkovGeostoryGenerator` that produces N
     plausible 3D realizations consistent with that surface signature.

This is *prior conditioning*, not subsurface inversion: the imagery
constrains which structural styles are plausible, then the Markov
sampler enumerates instances within that style space.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from pydtmc import MarkovChain

from geogen.generation.model_generators import (
    MarkovGeostoryGenerator,
    MarkovMatrixParser,
)
from geogen.gis.features import TileFeatures
from geogen.gis.mpc import Tile
from geogen.gis.regions import TectonicRegion


# Empirical thresholds for "is this a high-relief mountainous tile?" etc.
# Calibrated for NZ at 7.68 km / 30 m sampling.
_RELIEF_HIGH_M = 1200.0
_RELIEF_FLAT_M = 200.0
_SLOPE_STEEP_DEG = 25.0
_LINEAMENT_STRONG = 0.35
_NDVI_BARREN = 0.15


def feature_modulation(features: TileFeatures) -> Dict[str, float]:
    """Per-tile multiplicative bumps on top of region priors.

    Each rule is a soft hint: high relief boosts mountain/fault categories,
    flat low-relief boosts sediment, strong oriented lineaments boost
    folding/faulting, barren high-rock NDVI suppresses sediment cover, etc.
    """
    bumps: Dict[str, float] = {}

    # Relief axis -- mountain-belt vs basin
    if features.relief_m > _RELIEF_HIGH_M:
        bumps["Mountains"] = bumps.get("Mountains", 1.0) * 1.6
        bumps["Erosion"] = bumps.get("Erosion", 1.0) * 1.4
        bumps["Sediment"] = bumps.get("Sediment", 1.0) * 0.6
    elif features.relief_m < _RELIEF_FLAT_M:
        bumps["Mountains"] = 0.3
        bumps["Sediment"] = 1.8
        bumps["Erosion"] = 0.7
        bumps["Fold"] = 0.5

    # Slope std -- structurally complex terrain
    if features.slope_std_deg > _SLOPE_STEEP_DEG:
        bumps["Fault"] = bumps.get("Fault", 1.0) * 1.3
        bumps["Fold"] = bumps.get("Fold", 1.0) * 1.2

    # Lineament fabric -- aligned ridges/valleys hint at folds or faults
    if features.lineament_strength > _LINEAMENT_STRONG:
        bumps["Fold"] = bumps.get("Fold", 1.0) * 1.4
        bumps["Fault"] = bumps.get("Fault", 1.0) * 1.4
        bumps["Slip"] = bumps.get("Slip", 1.0) * 1.2

    # NDVI -- barren rock-heavy tiles suggest plutonic/orogenic exposure
    if features.ndvi_mean is not None and features.ndvi_mean < _NDVI_BARREN:
        bumps["Pluton"] = bumps.get("Pluton", 1.0) * 1.3
        bumps["BaseStrata"] = bumps.get("BaseStrata", 1.0) * 1.2
        bumps["Sediment"] = bumps.get("Sediment", 1.0) * 0.7

    return bumps


def combine_weights(
    region: TectonicRegion,
    features: Optional[TileFeatures] = None,
) -> Dict[str, float]:
    """Multiply region priors by per-tile feature modulation."""
    weights = dict(region.category_weights)
    if features is not None:
        for k, v in feature_modulation(features).items():
            weights[k] = weights.get(k, 1.0) * v
    return weights


def bias_transition_matrix(
    states: list,
    matrix: np.ndarray,
    weights: Dict[str, float],
) -> np.ndarray:
    """Multiply each column by its category weight, then re-row-normalize.

    Columns are *destination states*: scaling column ``c`` by ``w_c`` makes
    every state more (or less) likely to transition into category ``c``.
    Rows are renormalized so the result is still a valid stochastic matrix.
    """
    w = np.array([weights.get(s, 1.0) for s in states], dtype=np.float64)
    biased = matrix * w[None, :]
    row_sums = biased.sum(axis=1, keepdims=True)
    # Avoid divide-by-zero on absorbing rows (shouldn't happen in our matrix)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return biased / row_sums


class ConditionedMarkovGenerator(MarkovGeostoryGenerator):
    """A Markov generator whose transition matrix is biased toward a target
    set of structural categories.

    Parameters
    ----------
    category_weights : dict[str, float]
        Multiplicative biases per category name (state). Keys must match
        Markov state names (i.e. ``geogen.generation.categorical_events.__all__``).
        Missing keys default to 1.0 (no bias).
    """

    def __init__(self, category_weights: Optional[Dict[str, float]] = None, **kwargs):
        self._category_weights = dict(category_weights or {})
        super().__init__(**kwargs)
        self._apply_category_weights()

    def _apply_category_weights(self):
        if not self._category_weights:
            return
        parser: MarkovMatrixParser = self.markov_matrix_parser
        biased = bias_transition_matrix(
            list(parser.markov_states),
            parser.transition_matrix,
            self._category_weights,
        )
        self.mc = MarkovChain(biased, list(parser.markov_states))


def generator_for_tile(
    tile: Tile,
    features: Optional[TileFeatures] = None,
    model_bounds=((-3840, 3840), (-3840, 3840), (-1920, 1920)),
    model_resolution=(256, 256, 128),
) -> ConditionedMarkovGenerator:
    """Build a conditioned generator for a specific MPC tile."""
    weights = combine_weights(tile.region, features)
    return ConditionedMarkovGenerator(
        category_weights=weights,
        model_bounds=model_bounds,
        model_resolution=model_resolution,
    )
