"""Tests for the imagery -> Markov-prior conditioning math."""

import numpy as np

from geogen.gis.conditioning import (
    ConditionedMarkovGenerator,
    bias_transition_matrix,
    combine_weights,
    feature_modulation,
)
from geogen.gis.features import TileFeatures
from geogen.gis.regions import get_region


def test_bias_transition_matrix_preserves_row_stochastic():
    states = ["A", "B", "C"]
    M = np.array(
        [[0.5, 0.3, 0.2],
         [0.1, 0.6, 0.3],
         [0.2, 0.2, 0.6]]
    )
    weights = {"A": 2.0, "B": 0.5, "C": 1.0}
    biased = bias_transition_matrix(states, M, weights)
    assert biased.shape == M.shape
    np.testing.assert_allclose(biased.sum(axis=1), 1.0, atol=1e-12)


def test_bias_transition_matrix_increases_target_column():
    states = ["A", "B", "C"]
    M = np.full((3, 3), 1.0 / 3.0)
    biased = bias_transition_matrix(states, M, {"B": 5.0})
    # B column should dominate after biasing a uniform matrix
    assert (biased[:, 1] > biased[:, 0]).all()
    assert (biased[:, 1] > biased[:, 2]).all()


def test_bias_with_no_weights_is_identity():
    states = ["A", "B"]
    M = np.array([[0.7, 0.3], [0.4, 0.6]])
    np.testing.assert_allclose(bias_transition_matrix(states, M, {}), M)


def test_feature_modulation_respects_relief_axis():
    flat = TileFeatures(
        relief_m=80.0, mean_elev_m=20.0, slope_mean_deg=1.0, slope_std_deg=2.0,
        roughness=0.0, lineament_strength=0.05, lineament_azimuth_deg=10.0,
    )
    mountainous = TileFeatures(
        relief_m=2200.0, mean_elev_m=1500.0, slope_mean_deg=30.0, slope_std_deg=12.0,
        roughness=0.05, lineament_strength=0.5, lineament_azimuth_deg=45.0,
    )
    flat_b = feature_modulation(flat)
    mtn_b = feature_modulation(mountainous)
    assert flat_b.get("Sediment", 1.0) > 1.0
    assert flat_b.get("Mountains", 1.0) < 1.0
    assert mtn_b.get("Mountains", 1.0) > 1.0
    assert mtn_b.get("Fault", 1.0) > 1.0


def test_combine_weights_multiplies_region_and_feature():
    region = get_region("southern_alps")
    feats = TileFeatures(
        relief_m=2000.0, mean_elev_m=1000.0, slope_mean_deg=30.0, slope_std_deg=15.0,
        roughness=0.05, lineament_strength=0.5, lineament_azimuth_deg=120.0,
    )
    weights = combine_weights(region, feats)
    # Region prior already boosts Mountains; high relief should compound it.
    assert weights["Mountains"] > region.category_weights["Mountains"]


def test_conditioned_generator_runs_end_to_end():
    """Generates one model with biased weights -- catches integration regressions."""
    gen = ConditionedMarkovGenerator(
        category_weights={"Sediment": 3.0, "Mountains": 0.1},
        model_resolution=(32, 32, 16),  # tiny grid for speed
    )
    model = gen.generate_model()
    model.fill_nans()
    grid = model.get_data_grid()
    assert grid.shape == (32, 32, 16)
