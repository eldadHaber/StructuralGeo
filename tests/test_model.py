import numpy as np
import pytest

import geogen.model as geo


def test_initialization_scalar_bounds():
    model = geo.GeoModel(bounds=(0, 10), resolution=10)
    assert model.bounds == ((0, 10), (0, 10), (0, 10))
    assert model.resolution == (10, 10, 10)


def test_initialization_tuple_bounds():
    model = geo.GeoModel(bounds=((0, 10), (0, 20), (0, 30)), resolution=(5, 10, 15))
    assert model.bounds == ((0, 10), (0, 20), (0, 30))
    assert model.resolution == (5, 10, 15)


def test_invalid_bounds():
    with pytest.raises(AssertionError):
        geo.GeoModel(bounds=(0, 1, 2))


def test_invalid_resolution_string():
    with pytest.raises(ValueError):
        geo.GeoModel(bounds=(0, 10), resolution="high")


def test_invalid_resolution_wrong_length():
    with pytest.raises(AssertionError):
        geo.GeoModel(bounds=(0, 10), resolution=(10, 10))


def test_mesh_setup():
    resolution = (3, 5, 7)
    model = geo.GeoModel(bounds=((0, 1), (0, 1), (0, 1)), resolution=resolution)
    model._setup_mesh()

    assert model.X.shape == resolution
    assert model.Y.shape == resolution
    assert model.Z.shape == resolution
    assert len(model.xyz) == int(np.prod(resolution))
