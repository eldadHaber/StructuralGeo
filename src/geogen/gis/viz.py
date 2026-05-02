"""
Lightweight viewers for sample tiles.

Two views:

  * :func:`satellite_panel`     -- 2D matplotlib figure: Sentinel-2 RGB,
    DEM hillshade, NDVI, and forward-modeled gravity.
  * :func:`pyvista_terrain_view` -- 3D PyVista plotter showing the surface
    topography draped (optionally textured with the satellite RGB) and the
    subsurface density volume rendered with a clipped/sliced view.

Both viewers operate on a :class:`SampleTile` (from :mod:`geogen.gis.io`)
and an optional source :class:`Tile` (for the satellite RGB).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from geogen.gis.io import AIR_LITHOLOGY, SampleTile
from geogen.gis.mpc import Tile


# ---------------------------------------------------------------------------
# 2D satellite panel


def _hillshade(dem: np.ndarray, az_deg: float = 315.0, alt_deg: float = 45.0,
               pixel_m: float = 30.0) -> np.ndarray:
    """Standard cosine-of-incidence hillshade in [0, 1]."""
    az = np.deg2rad(360.0 - az_deg + 90.0)
    alt = np.deg2rad(alt_deg)
    gy, gx = np.gradient(dem.astype(np.float32), pixel_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shaded = (
        np.sin(alt) * np.cos(slope)
        + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    )
    return np.clip(shaded, 0.0, 1.0)


def _stretch_rgb(rgb: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """Per-band percentile stretch to [0, 1] for display.

    Robust to NaNs (uses nanpercentile, fills NaN output with 0).
    """
    out = np.empty_like(rgb)
    for c in range(rgb.shape[-1]):
        band = rgb[..., c]
        finite = np.isfinite(band)
        if not finite.any():
            out[..., c] = 0.0
            continue
        lo, hi = np.nanpercentile(band, [p_lo, p_hi])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
            out[..., c] = 0.0
            continue
        scaled = (band - lo) / (hi - lo)
        scaled = np.where(np.isfinite(scaled), scaled, 0.0)
        out[..., c] = np.clip(scaled, 0.0, 1.0)
    return out


def satellite_panel(
    sample: SampleTile,
    source_tile: Optional[Tile] = None,
    figsize=(13, 11),
):
    """Build a 2x2 matplotlib figure of the tile's surface data.

    Returns the matplotlib Figure (not shown). Caller can ``fig.savefig`` or
    ``plt.show()``.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    pixel_m = sample.metadata.get("mesh", {}).get("dx_m", 30.0)

    # 1. Sentinel-2 true-color (or hillshade if not available)
    ax = axes[0]
    if source_tile is not None and source_tile.s2_rgbnir is not None:
        rgb_in = np.transpose(source_tile.s2_rgbnir[:3], (1, 2, 0))
        rgb = _stretch_rgb(rgb_in)
        # Build an RGBA so no-data pixels are transparent over a hillshade
        nodata_mask = ~np.all(np.isfinite(rgb_in), axis=-1)
        alpha = np.where(nodata_mask, 0.0, 1.0)
        rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
        ax.imshow(_hillshade(sample.surface, pixel_m=pixel_m), cmap="gray", alpha=0.5)
        ax.imshow(rgba)
        coverage = 100.0 * (1.0 - nodata_mask.mean())
        title = "Sentinel-2 RGB (true-color)"
        if coverage < 99.0:
            title += f"  [{coverage:.0f}% coverage]"
        ax.set_title(title)
    else:
        ax.imshow(_hillshade(sample.surface, pixel_m=pixel_m), cmap="gray")
        ax.set_title("DEM hillshade (no S2 available)")
    ax.set_xticks([]); ax.set_yticks([])

    # 2. DEM colormap
    ax = axes[1]
    im = ax.imshow(sample.surface, cmap="terrain")
    ax.set_title(f"Topography (m)  relief={float(np.ptp(sample.surface)):.0f} m")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 3. NDVI (or DEM hillshade as fallback)
    ax = axes[2]
    if source_tile is not None and source_tile.s2_rgbnir is not None:
        red, _, _, nir = source_tile.s2_rgbnir
        denom = nir + red
        ndvi = np.where(denom > 1e-4, (nir - red) / denom, 0.0)
        im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
        ax.set_title("NDVI")
    else:
        im = ax.imshow(_hillshade(sample.surface, pixel_m=pixel_m), cmap="gray")
        ax.set_title("Hillshade")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 4. Forward gravity gz
    ax = axes[3]
    g = sample.gravity
    vmax = float(np.nanmax(np.abs(g))) if np.isfinite(g).any() else 1.0
    im = ax.imshow(g, cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Forward gravity gz (mGal)  range=[{g.min():.1f}, {g.max():.1f}]")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    region = sample.metadata.get("region_name", "tile")
    fig.suptitle(f"Sample tile: {region}", fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3D PyVista viewer


def pyvista_terrain_view(
    sample: SampleTile,
    source_tile: Optional[Tile] = None,
    show_density: bool = True,
    density_mode: str = "slice",
    z_exaggeration: float = 1.0,
    plotter=None,
):
    """Build a PyVista 3D scene with full topography + subsurface density.

    The topography surface is always rendered fully (LIDAR-DEM-style). The
    subsurface density is shown via one of:

      * ``"slice"``       -- a single vertical orthogonal slice through the
        middle of the volume. Surface stays full and intact (default).
      * ``"corner"``      -- a 25% x 25% corner removed from both surface
        and density, exposing two cross-section walls (block diagram).
      * ``"y_half"``      -- legacy 50% y-clip on both surface and density.
      * ``"x_half"``      -- legacy 50% x-clip on both surface and density.
      * ``"none"``        -- topography only.

    Topography is textured with the Sentinel-2 RGB if ``source_tile`` has
    one, otherwise coloured by elevation. Pass ``z_exaggeration > 1`` to
    accentuate relief.
    """
    import pyvista as pv

    mesh = sample.metadata.get("mesh", {})
    dx = float(mesh.get("dx_m", 30.0))
    dy = float(mesh.get("dy_m", 30.0))
    dz = float(mesh.get("dz_m", 30.0))
    z_top = float(mesh.get("z_top_m", float(sample.surface.max())))
    ny, nx = sample.surface.shape
    nz = sample.target.shape[0]

    # ---- Topography surface (always full) ---------------------------------
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    XX, YY = np.meshgrid(xs, ys)
    ZZ = sample.surface.astype(np.float32) * float(z_exaggeration)
    surf = pv.StructuredGrid(XX, YY, ZZ)
    surf["elevation_m"] = sample.surface.ravel(order="C")

    texture = None
    if source_tile is not None and source_tile.s2_rgbnir is not None:
        rgb_arr = np.transpose(source_tile.s2_rgbnir[:3], (1, 2, 0))
        rgb = _stretch_rgb(rgb_arr)
        tex_img = (rgb * 255).astype(np.uint8)
        u = np.tile(np.linspace(0, 1, nx), ny)
        v = np.repeat(np.linspace(1, 0, ny), nx)
        surf.active_texture_coordinates = np.column_stack([u, v]).astype(np.float32)
        texture = pv.numpy_to_texture(tex_img)

    # ---- Subsurface density volume ----------------------------------------
    density_artifact = None  # either a slice mesh or a clipped block
    if show_density and density_mode != "none":
        grid = pv.ImageData(
            dimensions=(nx + 1, ny + 1, nz + 1),
            spacing=(dx, dy, dz),
            origin=(0.0, 0.0, z_top - nz * dz),
        )
        density_pv_order = sample.target[::-1, :, :]
        grid.cell_data["density_contrast"] = (
            density_pv_order.transpose(2, 1, 0).ravel(order="F").astype(np.float32)
        )
        grid.cell_data["air_mask"] = (
            (sample.lithology[::-1, :, :] == AIR_LITHOLOGY)
            .transpose(2, 1, 0).ravel(order="F").astype(np.uint8)
        )
        rock_volume = grid.threshold(0.5, scalars="air_mask", invert=True)

        if density_mode == "slice":
            # A single vertical y-slice through the middle. Doesn't clip the
            # topography surface, so the LIDAR DEM stays fully intact.
            density_artifact = rock_volume.slice(
                normal="y", origin=(0.0, ny * dy / 2.0, 0.0)
            )
        elif density_mode == "corner":
            # Cut a 25% x 25% corner from both surface and volume so the
            # remaining surface still covers most of the tile.
            cx = nx * dx * 0.75
            cy = ny * dy * 0.75
            keep = (
                rock_volume.clip(normal="x", origin=(cx, 0, 0), invert=False)
                .merge(rock_volume.clip(normal="y", origin=(0, cy, 0), invert=False))
            )
            # Simpler: just take the box outside the cut corner
            density_artifact = rock_volume.clip_box(
                bounds=(cx, nx * dx, cy, ny * dy, z_top - nz * dz, z_top),
                invert=True,
            )
            surf = surf.clip_box(
                bounds=(cx, nx * dx, cy, ny * dy, ZZ.min() - 1, ZZ.max() + 1),
                invert=True,
            )
        elif density_mode == "y_half":
            density_artifact = rock_volume.clip(
                normal="y", origin=(0, ny * dy / 2, 0)
            )
            surf = surf.clip(normal="y", origin=(0, ny * dy / 2, 0))
        elif density_mode == "x_half":
            density_artifact = rock_volume.clip(
                normal="x", origin=(nx * dx / 2, 0, 0)
            )
            surf = surf.clip(normal="x", origin=(nx * dx / 2, 0, 0))
        else:
            raise ValueError(f"Unknown density_mode: {density_mode!r}")

    # ---- Compose plotter --------------------------------------------------
    if plotter is None:
        plotter = pv.Plotter(window_size=(1200, 800))

    if texture is not None:
        plotter.add_mesh(surf, texture=texture, name="topography")
    else:
        plotter.add_mesh(
            surf, scalars="elevation_m", cmap="terrain",
            name="topography", show_scalar_bar=True,
            scalar_bar_args={"title": "Elevation (m)"},
            lighting=True, smooth_shading=True,
        )

    if density_artifact is not None and density_artifact.n_cells > 0:
        plotter.add_mesh(
            density_artifact,
            scalars="density_contrast",
            cmap="RdBu_r",
            clim=(-800, 800),
            opacity=1.0 if density_mode == "slice" else 0.9,
            name="density",
            scalar_bar_args={"title": "Density contrast (kg/m^3)"},
        )

    plotter.add_axes()
    plotter.show_grid()
    region = sample.metadata.get("region_name", "tile")
    plotter.add_text(
        f"{region}  |  {nx}x{ny}x{nz} @ {dx:.0f} m  |  density={density_mode}",
        font_size=10,
    )
    return plotter
