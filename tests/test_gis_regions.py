"""Sanity checks on NZ region metadata."""

from geogen.generation.categorical_events import __all__ as EVENT_NAMES
from geogen.gis.regions import NZ_REGIONS, REGION_LOOKUP, get_region


def test_nz_regions_have_unique_names():
    names = [r.name for r in NZ_REGIONS]
    assert len(names) == len(set(names))
    assert REGION_LOOKUP.keys() == set(names)


def test_bboxes_are_inside_nz_envelope():
    # Roughly the NZ extent (excludes Chatham Is).
    for r in NZ_REGIONS:
        lon_min, lat_min, lon_max, lat_max = r.bbox
        assert 165.0 <= lon_min < lon_max <= 179.0
        assert -47.5 <= lat_min < lat_max <= -34.0


def test_category_weights_reference_real_events():
    valid = set(EVENT_NAMES)
    for r in NZ_REGIONS:
        unknown = set(r.category_weights) - valid
        assert not unknown, f"{r.name}: unknown event names {unknown}"
        for name, w in r.category_weights.items():
            assert w > 0, f"{r.name}: weight for {name} must be positive"


def test_get_region_lookup_and_error():
    r = get_region("southern_alps")
    assert r.name == "southern_alps"
    try:
        get_region("not_a_region")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown region")
