"""
Curated New Zealand tectonic settings for sampling structural-geology training tiles.

Each region defines a lat/lon bounding box and a set of category-weight priors that
bias the GeoGen Markov generator toward the structural styles characteristic of
that setting. Per-tile DEM/imagery features further modulate these priors at
sample time (see :mod:`geogen.gis.features`, :mod:`geogen.gis.conditioning`).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class TectonicRegion:
    name: str
    description: str
    # (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
    bbox: Tuple[float, float, float, float]
    # Multiplicative biases over Markov category names (default 1.0 = unchanged).
    # Names must match keys in geogen.generation.categorical_events.__all__.
    category_weights: Dict[str, float] = field(default_factory=dict)


# Tectonic settings spanning NZ's main structural styles.
# Bboxes intentionally generous so per-tile sampling has room to roam.
NZ_REGIONS: List[TectonicRegion] = [
    TectonicRegion(
        name="southern_alps",
        description=(
            "Alpine Fault transpressional thrust belt; uplifted greywacke/schist "
            "with high relief, oblique reverse faulting, tight folds."
        ),
        bbox=(168.6, -44.4, 171.0, -42.8),
        category_weights={
            "Fault": 2.2,
            "Fold": 1.8,
            "Mountains": 2.0,
            "Slip": 1.4,
            "Erosion": 1.3,
            "Sediment": 0.7,
        },
    ),
    TectonicRegion(
        name="taupo_volcanic_zone",
        description=(
            "Active continental rift arc with calderas, rhyolitic volcanism, "
            "ignimbrite plateaus and shallow plutons."
        ),
        bbox=(175.5, -39.4, 176.7, -38.1),
        category_weights={
            "Dike": 2.5,
            "Sills": 2.2,
            "Pluton": 2.0,
            "Mountains": 1.4,
            "Fold": 0.4,
            "Fault": 1.2,
        },
    ),
    TectonicRegion(
        name="marlborough_fault_system",
        description=(
            "Strike-slip transfer zone between the Hikurangi subduction margin "
            "and the Alpine Fault; dextral faulting, pull-apart basins."
        ),
        bbox=(172.6, -42.4, 174.4, -41.4),
        category_weights={
            "Fault": 2.6,
            "Slip": 2.0,
            "Fold": 1.2,
            "Mountains": 1.3,
            "Sediment": 0.9,
        },
    ),
    TectonicRegion(
        name="otago_schist",
        description=(
            "Folded basement schist with broad antiforms/synforms and "
            "range-and-basin topography from late Cenozoic reverse faulting."
        ),
        bbox=(168.8, -46.0, 170.6, -44.6),
        category_weights={
            "Fold": 2.4,
            "BaseStrata": 1.4,
            "Fault": 1.5,
            "Erosion": 1.1,
            "Dike": 0.5,
        },
    ),
    TectonicRegion(
        name="canterbury_plains",
        description=(
            "Thick alluvial fan sequences over greywacke basement; "
            "near-flat topography, dominantly sedimentary section."
        ),
        bbox=(171.5, -44.2, 172.8, -43.3),
        category_weights={
            "Sediment": 2.6,
            "Erosion": 1.6,
            "Fold": 0.3,
            "Fault": 0.6,
            "Mountains": 0.3,
            "Dike": 0.2,
            "Pluton": 0.2,
        },
    ),
    TectonicRegion(
        name="fiordland",
        description=(
            "Exhumed lower-crustal orogen of Paleozoic-Mesozoic plutons, "
            "gneiss, and mafic-ultramafic complexes; deeply dissected."
        ),
        bbox=(166.5, -45.8, 167.8, -44.5),
        category_weights={
            "Pluton": 2.4,
            "BaseStrata": 1.6,
            "Mountains": 1.5,
            "Fold": 1.2,
            "OreDeposit": 1.4,
            "Sediment": 0.4,
        },
    ),
]


REGION_LOOKUP: Dict[str, TectonicRegion] = {r.name: r for r in NZ_REGIONS}


def get_region(name: str) -> TectonicRegion:
    if name not in REGION_LOOKUP:
        raise KeyError(
            f"Unknown region '{name}'. Available: {sorted(REGION_LOOKUP)}"
        )
    return REGION_LOOKUP[name]
