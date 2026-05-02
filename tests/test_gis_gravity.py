"""Verify the gravity forward operator against analytic limits."""

import numpy as np

from geogen.gis.gravity import (
    forward_gz,
    point_mass_gz,
    slab_gz_analytic,
)


def test_zero_density_yields_zero_gravity():
    rho = np.zeros((4, 8, 8), dtype=np.float32)
    g = forward_gz(rho, dx_m=50, dy_m=50, dz_m=50, obs_z_const_m=200, z_top_m=0)
    assert g.shape == (8, 8)
    assert np.allclose(g, 0.0)


def test_single_point_mass_matches_analytic():
    """One isolated voxel of dense rock; observed at center should match
    the analytic G*M/r^2 with r the vertical separation."""
    nx, ny, nz = 33, 33, 9
    dx = dy = dz = 50.0  # meters
    rho_value = 1000.0   # kg/m^3 contrast
    rho = np.zeros((nz, ny, nx), dtype=np.float32)
    # Place mass at the center voxel of a middle layer
    rho[4, 16, 16] = rho_value
    z_top = 0.0
    obs_z = 1000.0  # 1 km above z_top

    g = forward_gz(rho, dx_m=dx, dy_m=dy, dz_m=dz,
                   obs_z_const_m=obs_z, z_top_m=z_top)

    mass = rho_value * dx * dy * dz
    voxel_z_center = z_top - (4 + 0.5) * dz
    h = obs_z - voxel_z_center
    expected_at_center = point_mass_gz(mass, 0.0, 0.0, h)

    # Center observation cell value should agree with the analytic point-mass formula
    assert g[16, 16] == g[16, 16]  # not nan
    np.testing.assert_allclose(g[16, 16], expected_at_center, rtol=1e-6, atol=1e-12)

    # Off-axis cell should match the off-axis formula too
    expected_offset = point_mass_gz(mass, 5 * dx, 3 * dy, h)
    np.testing.assert_allclose(g[16 + 3, 16 + 5], expected_offset, rtol=1e-6, atol=1e-12)


def test_uniform_layer_approaches_bouguer_slab():
    """A wide thin slab should approach the Bouguer slab gravity at the center.

    Use a single-layer volume, observed only mildly above; with enough
    horizontal extent, the center cell should approach the analytic slab.
    """
    nx = ny = 129
    nz = 1
    dx = dy = dz = 50.0
    rho = 500.0  # kg/m^3 contrast
    arr = np.full((nz, ny, nx), rho, dtype=np.float32)
    z_top = 0.0
    # Observer is well above the slab top
    obs_z = z_top + 10.0

    g = forward_gz(arr, dx_m=dx, dy_m=dy, dz_m=dz,
                   obs_z_const_m=obs_z, z_top_m=z_top)

    expected = slab_gz_analytic(thickness_m=dz, density_kg_m3=rho)
    center = float(g[ny // 2, nx // 2])
    # Finite-extent slab approaches infinite-slab Bouguer to within ~10% at
    # the centre of a 129-cell wide grid.
    assert abs(center - expected) / abs(expected) < 0.10, (
        f"center={center:.4f}, expected~{expected:.4f}"
    )


def test_finite_cube_is_a_fraction_of_bouguer_slab():
    """A finite cube of dense rock observed just above one face must be
    *less than* the infinite-slab Bouguer value (because the slab is
    truncated laterally) but a meaningful fraction of it (here we expect
    around 30-60% for a roughly cube-shaped column)."""
    nx = ny = 65
    nz = 60
    dx = dy = dz = 50.0
    rho = np.full((nz, ny, nx), 200.0, dtype=np.float32)
    g = forward_gz(rho, dx_m=dx, dy_m=dy, dz_m=dz,
                   obs_z_const_m=50.0, z_top_m=0.0)
    center = float(g[ny // 2, nx // 2])
    bouguer = slab_gz_analytic(60 * dz, 200.0)
    assert center > 0
    assert 0.25 * bouguer < center < bouguer, (
        f"center={center:.2f} mGal, bouguer={bouguer:.2f} mGal"
    )
