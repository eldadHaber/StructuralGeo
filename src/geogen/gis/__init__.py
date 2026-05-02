"""
GIS extensions for GeoGen.

Adds Microsoft Planetary Computer ingestion of New Zealand satellite/DEM
tiles, structural-feature extraction, and imagery-conditioned synthetic
structural-geology generation. The submodule has additional dependencies
(``pystac-client``, ``planetary-computer``, ``rioxarray``, ``pyproj``);
install them with ``pip install -e .[gis]``.

Quick start::

    from geogen.gis import NZGISConditionedDataset

    ds = NZGISConditionedDataset(tiles_per_region=1, realizations_per_tile=8)
    sample = ds[0]
    print(sample.region_name, sample.density.shape)
"""

from geogen.gis.conditioning import (
    ConditionedMarkovGenerator,
    bias_transition_matrix,
    combine_weights,
    feature_modulation,
    generator_for_tile,
)
from geogen.gis.dataset import GISConditionedSample, NZGISConditionedDataset
from geogen.gis.density import density_table, lithology_to_density
from geogen.gis.features import TileFeatures, extract_features
from geogen.gis.gravity import G_NEWTON, forward_gz, point_mass_gz, slab_gz_analytic
from geogen.gis.io import (
    SCHEMA_VERSION,
    SampleTile,
    build_sample_tile,
    load_sample_tile,
    save_sample_tile,
)
from geogen.gis.regions import NZ_REGIONS, REGION_LOOKUP, TectonicRegion, get_region

# mpc imports trigger lazy GIS-deps resolution; expose at top level too
from geogen.gis import mpc  # noqa: F401
from geogen.gis import viz  # noqa: F401  -- pyvista/matplotlib lazily imported inside

__all__ = [
    # regions
    "NZ_REGIONS",
    "REGION_LOOKUP",
    "TectonicRegion",
    "get_region",
    # features
    "TileFeatures",
    "extract_features",
    # conditioning
    "ConditionedMarkovGenerator",
    "bias_transition_matrix",
    "combine_weights",
    "feature_modulation",
    "generator_for_tile",
    # density
    "lithology_to_density",
    "density_table",
    # dataset
    "GISConditionedSample",
    "NZGISConditionedDataset",
    # gravity
    "forward_gz",
    "point_mass_gz",
    "slab_gz_analytic",
    "G_NEWTON",
    # sample tile io
    "SampleTile",
    "SCHEMA_VERSION",
    "build_sample_tile",
    "save_sample_tile",
    "load_sample_tile",
    # submodules
    "mpc",
    "viz",
]
