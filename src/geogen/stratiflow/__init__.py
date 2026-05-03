"""Stratiflow synthetic-dataset generator (StructuralGeo + derived channels).

See the synthetic-test-dataset spec for the full contract. Quick start:

    from geogen.stratiflow import GenerationConfig, generate_sample
    cfg = GenerationConfig(seed=20260502)
    generate_sample(cfg, out_dir="stratiflow_samples/sample_001")
"""

from geogen.stratiflow.config import GenerationConfig
from geogen.stratiflow.generate import generate_sample, generate_sample_from_nz_tile
from geogen.stratiflow.qc import QCFailure
from geogen.stratiflow import viz  # noqa: F401  (matplotlib lazily imported)

__all__ = [
    "GenerationConfig",
    "generate_sample",
    "generate_sample_from_nz_tile",
    "QCFailure",
    "viz",
]
