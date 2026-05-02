"""Lithology + stratigraphic vocabulary for stratiflow samples.

Maps GeoGen integer codes to spec-compliant unit names, density contrasts
(kg/m^3 relative to ``reference_density_kg_m3``), magnetic susceptibilities
(SI), and chemistry templates. Builds the density and susceptibility
volumes by lookup.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Bands of GeoGen lithology codes (mirrors geogen.generation.geowords)
AIR_CODE = -1
BASEMENT_CODE = 0
SEDIMENT_CODES = (1, 2, 3, 4, 5)
DIKE_CODES = (6, 7, 8)
INTRUSION_CODES = (9, 10, 11)
ORE_CODES = (12, 13)


# Per-spec representative density-contrast ranges (kg/m^3) and a typical
# susceptibility (SI). Chemistry templates are major-oxide percentages
# plus a few trace-element concentrations in ppm.
_VOCAB: Dict[int, dict] = {
    BASEMENT_CODE: dict(
        unit="basement",
        lithology="granite",
        category="basement",
        density_range=(0.0, 100.0),
        susceptibility=5e-4,
        chemistry={"SiO2_pct": 72.0, "Al2O3_pct": 14.0, "Fe2O3_pct": 2.0,
                   "K2O_pct": 4.0, "Cu_ppm": 20.0, "Au_ppm": 0.005},
    ),
    1: dict(
        unit="lower_sediments",
        lithology="shale",
        category="sediment",
        density_range=(-300.0, -150.0),
        susceptibility=2e-5,
        chemistry={"SiO2_pct": 60.0, "Al2O3_pct": 18.0, "Fe2O3_pct": 6.0,
                   "K2O_pct": 3.0, "Cu_ppm": 50.0, "Au_ppm": 0.002},
    ),
    2: dict(
        unit="middle_sediments",
        lithology="sandstone",
        category="sediment",
        density_range=(-250.0, -120.0),
        susceptibility=3e-5,
        chemistry={"SiO2_pct": 78.0, "Al2O3_pct": 9.0, "Fe2O3_pct": 2.5,
                   "K2O_pct": 1.5, "Cu_ppm": 15.0, "Au_ppm": 0.001},
    ),
    3: dict(
        unit="upper_sediments",
        lithology="siltstone",
        category="sediment",
        density_range=(-220.0, -110.0),
        susceptibility=2e-5,
        chemistry={"SiO2_pct": 66.0, "Al2O3_pct": 14.0, "Fe2O3_pct": 4.5,
                   "K2O_pct": 2.5, "Cu_ppm": 30.0, "Au_ppm": 0.002},
    ),
    4: dict(
        unit="conglomerate",
        lithology="conglomerate",
        category="sediment",
        density_range=(-260.0, -130.0),
        susceptibility=4e-5,
        chemistry={"SiO2_pct": 70.0, "Al2O3_pct": 11.0, "Fe2O3_pct": 3.5,
                   "K2O_pct": 2.0, "Cu_ppm": 25.0, "Au_ppm": 0.001},
    ),
    5: dict(
        unit="alluvium",
        lithology="alluvium",
        category="alluvium",
        density_range=(-500.0, -300.0),
        susceptibility=1e-5,
        chemistry={"SiO2_pct": 65.0, "Al2O3_pct": 12.0, "Fe2O3_pct": 4.0,
                   "K2O_pct": 2.0, "Cu_ppm": 20.0, "Au_ppm": 0.001},
    ),
    6: dict(
        unit="mafic_dike",
        lithology="basalt",
        category="volcanic",
        density_range=(200.0, 400.0),
        susceptibility=2e-2,
        chemistry={"SiO2_pct": 49.0, "Al2O3_pct": 15.0, "Fe2O3_pct": 12.0,
                   "K2O_pct": 0.6, "Cu_ppm": 80.0, "Au_ppm": 0.003},
    ),
    7: dict(
        unit="andesitic_dike",
        lithology="andesite",
        category="volcanic",
        density_range=(150.0, 300.0),
        susceptibility=1.2e-2,
        chemistry={"SiO2_pct": 58.0, "Al2O3_pct": 17.0, "Fe2O3_pct": 7.5,
                   "K2O_pct": 1.8, "Cu_ppm": 60.0, "Au_ppm": 0.003},
    ),
    8: dict(
        unit="rhyolitic_dike",
        lithology="rhyolite",
        category="volcanic",
        density_range=(100.0, 250.0),
        susceptibility=4e-3,
        chemistry={"SiO2_pct": 73.0, "Al2O3_pct": 13.0, "Fe2O3_pct": 1.5,
                   "K2O_pct": 4.5, "Cu_ppm": 18.0, "Au_ppm": 0.004},
    ),
    9: dict(
        unit="felsic_pluton",
        lithology="granodiorite",
        category="intrusive",
        density_range=(200.0, 350.0),
        susceptibility=5e-3,
        chemistry={"SiO2_pct": 67.0, "Al2O3_pct": 16.0, "Fe2O3_pct": 4.0,
                   "K2O_pct": 3.5, "Cu_ppm": 40.0, "Au_ppm": 0.005},
    ),
    10: dict(
        unit="intermediate_pluton",
        lithology="diorite",
        category="intrusive",
        density_range=(300.0, 450.0),
        susceptibility=2e-2,
        chemistry={"SiO2_pct": 56.0, "Al2O3_pct": 17.0, "Fe2O3_pct": 8.0,
                   "K2O_pct": 2.0, "Cu_ppm": 70.0, "Au_ppm": 0.006},
    ),
    11: dict(
        unit="mafic_pluton",
        lithology="gabbro",
        category="intrusive",
        density_range=(350.0, 500.0),
        susceptibility=8e-2,
        chemistry={"SiO2_pct": 48.0, "Al2O3_pct": 16.0, "Fe2O3_pct": 13.0,
                   "K2O_pct": 0.8, "Cu_ppm": 100.0, "Au_ppm": 0.007},
    ),
    12: dict(
        unit="sulfide_ore",
        lithology="massive_sulfide",
        category="ore",
        density_range=(700.0, 950.0),
        susceptibility=3e-2,
        chemistry={"SiO2_pct": 30.0, "Al2O3_pct": 5.0, "Fe2O3_pct": 25.0,
                   "K2O_pct": 0.5, "Cu_ppm": 8000.0, "Au_ppm": 0.5,
                   "S_pct": 30.0},
    ),
    13: dict(
        unit="oxide_ore",
        lithology="magnetite_ore",
        category="ore",
        density_range=(800.0, 1000.0),
        susceptibility=1.5e-1,
        chemistry={"SiO2_pct": 25.0, "Al2O3_pct": 4.0, "Fe2O3_pct": 50.0,
                   "K2O_pct": 0.2, "Cu_ppm": 200.0, "Au_ppm": 0.05},
    ),
}


_TRACE_ELEMENT_KEYS = ("Cu_ppm", "Au_ppm")


def is_intrusive_code(code: int) -> bool:
    return code in DIKE_CODES + INTRUSION_CODES + ORE_CODES


def build_vocabulary(
    present_codes: List[int], rng: np.random.Generator
) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    """Build lithology + stratigraphic dicts for the codes that appear.

    Returns
    -------
    lithology_codes : dict[int, dict]
        Per-code: name, density_kg_m3 (sampled within range), susceptibility_si,
        chemistry_template, lithology, category.
    stratigraphic_codes : dict[int, dict]
        Per-code: name + age_order index (0 = oldest).
    """
    lithology_codes: Dict[int, dict] = {}
    for code in present_codes:
        if code == AIR_CODE:
            continue
        if code not in _VOCAB:
            # Fallback: treat unknowns as basement-like
            entry = _VOCAB[BASEMENT_CODE].copy()
            entry["unit"] = f"unknown_{code}"
            entry["lithology"] = f"unknown_{code}"
        else:
            entry = _VOCAB[code]
        lo, hi = entry["density_range"]
        density = float(rng.uniform(lo, hi))
        lithology_codes[int(code)] = {
            "name": entry["unit"],
            "lithology": entry["lithology"],
            "category": entry["category"],
            "density_kg_m3": density,
            "susceptibility_si": float(entry["susceptibility"]),
            "chemistry_template": dict(entry["chemistry"]),
        }

    # Stratigraphic age ordering: depositional units ordered by median depth
    # (deeper = older). Intrusives come last as "late-stage" per spec.
    stratigraphic_codes: Dict[int, dict] = {}
    sed_basement_codes = [
        c for c in lithology_codes
        if not is_intrusive_code(c)
    ]
    intrusive_codes_sorted = sorted(
        [c for c in lithology_codes if is_intrusive_code(c)]
    )
    # The caller passes present_codes already in stratigraphic order if
    # possible; here we leave order for the caller to decide via depth.
    for order, code in enumerate(sed_basement_codes + intrusive_codes_sorted):
        stratigraphic_codes[int(code)] = {
            "name": lithology_codes[code]["name"],
            "age_order": int(order),
        }
    return lithology_codes, stratigraphic_codes


def reorder_by_median_depth(
    lithology_volume: np.ndarray, codes: List[int]
) -> List[int]:
    """Return codes sorted by ascending median z_index of voxels with that
    code (deeper = larger z_index = older). Intrusives moved to the end."""
    medians = {}
    for c in codes:
        if c == AIR_CODE:
            continue
        idx = np.argwhere(lithology_volume == c)
        if idx.size == 0:
            continue
        medians[c] = float(np.median(idx[:, 0]))  # axis 0 is z (top=0, bottom=nz-1)
    non_intr = sorted([c for c in medians if not is_intrusive_code(c)],
                      key=lambda c: -medians[c])  # deepest (largest z) first = oldest
    intr = sorted([c for c in medians if is_intrusive_code(c)])
    return non_intr + intr


def build_density_volume(
    lithology_volume: np.ndarray,
    lithology_codes: Dict[int, dict],
) -> np.ndarray:
    """Map lithology codes to density CONTRAST (kg/m^3). Air = 0 contrast.

    No noise is added at the volume level (spec rule).
    """
    out = np.zeros_like(lithology_volume, dtype=np.float32)
    for code, entry in lithology_codes.items():
        out[lithology_volume == code] = float(entry["density_kg_m3"])
    return out


def build_susceptibility_volume(
    lithology_volume: np.ndarray,
    lithology_codes: Dict[int, dict],
) -> np.ndarray:
    """Map lithology codes to susceptibility (SI)."""
    out = np.zeros_like(lithology_volume, dtype=np.float32)
    for code, entry in lithology_codes.items():
        out[lithology_volume == code] = float(entry["susceptibility_si"])
    return out


def is_major_oxide(key: str) -> bool:
    return key not in _TRACE_ELEMENT_KEYS and key.endswith("_pct")
