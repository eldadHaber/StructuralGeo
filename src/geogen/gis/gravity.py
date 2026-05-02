"""
Forward gravity operator for a regular density-contrast voxel volume.

Implements a layer-wise point-mass forward via 2D FFT convolution:

    gz(x_o, y_o) = G * dV * sum_z [ rho_contrast(x, y, z) * (z_o - z) /
                   ((x_o - x)^2 + (y_o - y)^2 + (z_o - z)^2)^(3/2) ]

The observation plane is constant in elevation (a single ``obs_z_const``).
This matches "draped at constant elevation" airborne survey geometry. For
strictly variable observation heights, downstream code can compute gz at
two heights and interpolate, or call this operator with a sliced volume
per station; that is left to callers since a single-FFT shortcut requires
constant obs_z.

Sign: ``gz`` is positive for downward acceleration. Output units are mGal
(1 mGal = 1e-5 m/s^2).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Newton's gravitational constant (m^3 kg^-1 s^-2)
G_NEWTON = 6.6743e-11
# Conversion m/s^2 -> mGal
M_S2_TO_MGAL = 1.0e5


def forward_gz(
    density_contrast: np.ndarray,  # shape [nz, ny, nx], kg/m^3
    dx_m: float,
    dy_m: float,
    dz_m: float,
    obs_z_const_m: float,
    z_top_m: float,
) -> np.ndarray:
    """Compute gz on a horizontal plane from a [nz, ny, nx] density contrast volume.

    Parameters
    ----------
    density_contrast : np.ndarray
        Density contrast (kg/m^3) in spec layout: ``[nz, ny, nx]`` with
        ``z_index = 0`` at the top (shallowest) layer.
    dx_m, dy_m, dz_m : float
        Voxel dimensions in meters.
    obs_z_const_m : float
        Constant observation-plane elevation (meters, same datum as ``z_top_m``).
    z_top_m : float
        Elevation of the *top face* of the topmost (z_index=0) voxel.
        Voxel z=0 spans [z_top_m - dz_m, z_top_m]; voxel z=k spans
        [z_top_m - (k+1)*dz_m, z_top_m - k*dz_m].

    Returns
    -------
    gz : np.ndarray
        Shape ``[ny, nx]``, gravity in mGal, positive downward.
    """
    if density_contrast.ndim != 3:
        raise ValueError(f"density_contrast must be 3D [nz, ny, nx], got {density_contrast.shape}")

    nz, ny, nx = density_contrast.shape
    dV = float(dx_m * dy_m * dz_m)

    # Per-voxel-center horizontal offsets in meters from the kernel origin.
    # The convolution kernel must cover the *full* extent of station -> voxel
    # offsets, i.e. (-(nx-1)..+(nx-1)) * dx_m. Use FFT zero-padding to handle
    # this without wraparound artifacts.
    pad_y, pad_x = ny, nx
    fft_ny = ny + pad_y
    fft_nx = nx + pad_x

    # Build kernel offsets centered at zero
    kx = (np.arange(-pad_x // 2, pad_x // 2 + nx) - 0) * dx_m  # placeholder
    # Actually we want kernel sized (fft_ny, fft_nx) with the (0,0)-shift in
    # the center, then we'll fft-shift to put origin at top-left.
    iy = np.arange(fft_ny) - ny  # offsets in pixels [-ny .. nx-1]
    ix = np.arange(fft_nx) - nx
    offy_m = iy[:, None] * dy_m
    offx_m = ix[None, :] * dx_m
    horiz2 = offy_m * offy_m + offx_m * offx_m  # shape (fft_ny, fft_nx)

    # Pre-FFT: pad density layers to (fft_ny, fft_nx).
    rho_pad = np.zeros((nz, fft_ny, fft_nx), dtype=np.float64)
    rho_pad[:, :ny, :nx] = density_contrast.astype(np.float64)
    rho_fft = np.fft.rfft2(rho_pad, s=(fft_ny, fft_nx), axes=(-2, -1))

    gz_fft = np.zeros_like(rho_fft[0])
    for k in range(nz):
        # Voxel-center elevation for layer k (z=0 is top)
        z_center = z_top_m - (k + 0.5) * dz_m
        dz_obs = obs_z_const_m - z_center  # positive when obs is above voxel
        denom = (horiz2 + dz_obs * dz_obs) ** 1.5
        # Kernel value at horizontal offset (dy, dx) for this layer
        K = G_NEWTON * dV * dz_obs / denom
        # Center the kernel so that lag (0,0) lives at index (ny, nx) before fft-shift
        K_shifted = np.fft.ifftshift(K)
        K_fft = np.fft.rfft2(K_shifted, s=(fft_ny, fft_nx))
        gz_fft += rho_fft[k] * K_fft

    gz_full = np.fft.irfft2(gz_fft, s=(fft_ny, fft_nx))
    # The convolution puts the station response at the same indices as the input
    # padding origin: output station (j_y, j_x) lives at index (j_y, j_x).
    gz = gz_full[:ny, :nx] * M_S2_TO_MGAL
    return gz.astype(np.float32)


def slab_gz_analytic(thickness_m: float, density_kg_m3: float) -> float:
    """Bouguer slab gravity in mGal for an infinite horizontal slab."""
    # gz = 2 * pi * G * rho * h, in m/s^2 -> mGal
    return 2.0 * np.pi * G_NEWTON * density_kg_m3 * thickness_m * M_S2_TO_MGAL


def point_mass_gz(
    mass_kg: float,
    dx_m: float,
    dy_m: float,
    dz_m: float,
) -> float:
    """gz at horizontal offset (dx_m, dy_m) and vertical separation dz_m below
    a point mass of ``mass_kg``. Positive when observer is above the mass."""
    r2 = dx_m * dx_m + dy_m * dy_m + dz_m * dz_m
    return G_NEWTON * mass_kg * dz_m / (r2 ** 1.5) * M_S2_TO_MGAL
