"""Generation config for the stratiflow synthetic dataset spec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GenerationConfig:
    """All knobs from the synthetic-dataset spec, with defaults."""

    # Mesh per spec: nz=64, ny=128, nx=128 at 50 m
    nz: int = 64
    ny: int = 128
    nx: int = 128
    dx_m: float = 50.0
    dy_m: float = 50.0
    dz_m: float = 50.0

    # Reference density used to convert to contrast everywhere
    reference_density_kg_m3: float = 2670.0

    # Boreholes
    n_boreholes: int = 8
    borehole_min_spacing_m: float = 800.0
    inclined_fraction: float = 0.25
    borehole_length_m_min: float = 800.0
    borehole_length_m_max: float = 2400.0
    sample_step_m: float = 5.0
    sample_min_length_m: float = 5.0
    sample_max_length_m: float = 50.0
    density_sample_noise_kg_m3: float = 25.0
    chemistry_relative_noise_major: float = 0.05
    chemistry_relative_noise_trace: float = 0.20

    # Surface
    surface_relief_amplitude_m: float = 150.0
    surface_relief_min_m: float = 200.0  # acceptance threshold
    surface_mapping_unmapped_fraction: float = 0.0
    surface_mapping_error_fraction: float = 0.0

    # Gravity
    obs_height_m: float = 50.0
    gravity_station_noise_mgal: float = 0.08
    gravity_regional_trend_amplitude_mgal: float = 3.0

    # Fault subset for user input
    user_known_fault_fraction: float = 0.5

    # Acceptance constraints (re-rolling the geomodel until satisfied)
    min_faults: int = 1
    min_units: int = 4
    max_units: int = 10
    max_geomodel_attempts: int = 20

    # RNG
    seed: int = 20260502

    @property
    def model_bounds(self) -> Tuple[Tuple[float, float], ...]:
        """((x0, x1), (y0, y1), (z0, z1)) in meters, origin at corner.

        Spec: origin at (0, 0, top_of_model). Model extends downward in z.
        """
        return (
            (0.0, self.nx * self.dx_m),
            (0.0, self.ny * self.dy_m),
            (-self.nz * self.dz_m, 0.0),
        )

    @property
    def model_resolution(self) -> Tuple[int, int, int]:
        """GeoGen takes (nx, ny, nz)."""
        return (self.nx, self.ny, self.nz)
