"""
Verify public `geogen` import works on a lean install (no optional extras)
"""

import importlib


def test_top_level_import():
    import geogen

    assert geogen.__title__ == "GeoGen"
    assert isinstance(geogen.__version__, str) and geogen.__version__


def test_submodules_importable():
    # Core submodules must import without torch / ipywidgets / pyvistaqt.
    for name in (
        "geogen.model",
        "geogen.plot",
        "geogen.generation",
        "geogen.probability",
        "geogen.filemanagement",
    ):
        mod = importlib.import_module(name)
        assert mod is not None


def test_model_construction():
    from geogen.model import GeoModel

    m = GeoModel(bounds=(0, 10), resolution=10)
    assert m.bounds == ((0, 10), (0, 10), (0, 10))
    assert m.resolution == (10, 10, 10)


def test_default_markov_matrix_loads():
    from geogen.generation import MarkovGeostoryGenerator

    gen = MarkovGeostoryGenerator()
    assert gen.mc is not None
    assert gen.event_dictionary


def test_dataset_requires_torch_extra_message():
    try:
        import torch  # noqa: F401

        has_torch = True
    except ImportError:
        has_torch = False

    if has_torch:
        from geogen.dataset import GeoData3DStreamingDataset  # noqa: F401
    else:
        import pytest

        with pytest.raises(ImportError):
            import geogen.dataset  # noqa: F401
