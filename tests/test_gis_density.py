"""Density-mapping unit tests."""

import numpy as np

from geogen.gis.density import density_table, lithology_to_density


def test_density_classes_have_plausible_values():
    table = density_table()
    # Sediments < basement < intrusive rocks < ores
    sediment_means = [table[i][0] for i in (1, 2, 3, 4, 5)]
    assert all(s < table[0][0] for s in sediment_means)  # sed < basement
    assert table[8][0] > table[0][0]                     # mafic dike > basement
    assert table[12][0] > table[8][0]                    # ore > dike


def test_air_voxels_are_zero_density():
    litho = np.array([[-1, -1, 0], [1, 6, 12]], dtype=np.int8)
    rho = lithology_to_density(litho, jitter=False)
    assert (rho[litho == -1] == 0.0).all()
    assert (rho[litho == 0] > 2000.0).all()
    assert (rho[litho == 12] > 3000.0).all()


def test_jitter_is_reproducible_with_seed():
    litho = np.zeros((8, 8, 8), dtype=np.int8)
    a = lithology_to_density(litho, seed=42)
    b = lithology_to_density(litho, seed=42)
    np.testing.assert_array_equal(a, b)


def test_jitter_off_is_constant_per_class():
    litho = np.full((4, 4, 4), 3, dtype=np.int8)
    rho = lithology_to_density(litho, jitter=False)
    assert np.unique(rho).size == 1
