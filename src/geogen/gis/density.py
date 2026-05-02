"""
Map GeoGen lithology codes to bulk-density volumes (kg/m^3).

GeoGen voxel data is a categorical grid with these codes (see
``geogen.generation.geowords``):

  -1  : air / above topography (after ``compute_model(normalize=True)``)
   0  : basement / bedrock
   1-5: sediments (5 fining/coarsening classes)
   6-8: dikes
   9-11: intrusions / plutons
  12-13: ore-deposit blobs

We assign each class a typical bulk density with a small per-tile jitter
so that the density volume is plausible without pretending to be measured.
This is intended for ML training targets, not geophysical modeling.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# Bulk density (kg/m^3) and per-class jitter (1-sigma). Values from common
# geophysics tables (e.g. Telford et al., Applied Geophysics).
_DENSITY_TABLE: Dict[int, tuple] = {
    -1: (0.0, 0.0),       # air
    0:  (2700.0, 80.0),   # crystalline basement
    1:  (2050.0, 120.0),  # loose / fine sediment
    2:  (2150.0, 120.0),
    3:  (2300.0, 100.0),  # mid sediment
    4:  (2450.0, 100.0),
    5:  (2600.0, 80.0),   # consolidated / lithified sediment
    6:  (2900.0, 80.0),   # mafic dike
    7:  (2950.0, 80.0),
    8:  (3000.0, 80.0),
    9:  (2650.0, 70.0),   # felsic pluton
    10: (2750.0, 80.0),   # intermediate pluton
    11: (2850.0, 90.0),   # mafic pluton
    12: (3500.0, 200.0),  # high-density ore (sulfide / oxide)
    13: (3700.0, 200.0),
}


def lithology_to_density(
    lithology: np.ndarray,
    seed: Optional[int] = None,
    jitter: bool = True,
) -> np.ndarray:
    """Convert a categorical lithology volume into a density volume.

    Parameters
    ----------
    lithology : np.ndarray
        Integer voxel codes from GeoModel.get_data_grid(). Any code not in
        the lookup table is mapped to basement density.
    seed : int, optional
        Seed for per-voxel density jitter (set for reproducibility).
    jitter : bool
        If True, perturb each voxel's density by a Gaussian draw with the
        class-specific sigma. If False, use mean density only.

    Returns
    -------
    np.ndarray (float32)
        Density in kg/m^3, same shape as ``lithology``.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros_like(lithology, dtype=np.float32)
    codes = np.unique(lithology)
    for code in codes:
        mean, sigma = _DENSITY_TABLE.get(int(code), _DENSITY_TABLE[0])
        mask = lithology == code
        if jitter and sigma > 0:
            out[mask] = rng.normal(mean, sigma, size=int(mask.sum())).astype(np.float32)
        else:
            out[mask] = mean
    return out


def density_table() -> Dict[int, tuple]:
    """Return a copy of the lithology-code -> (mean, sigma) table."""
    return dict(_DENSITY_TABLE)
