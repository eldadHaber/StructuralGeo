"""
Build one spec-compliant sample tile and visualize it.

End-to-end:
  1. Fetch (or synthesize, with --offline) a 7.68 km tile from MPC for a
     chosen NZ tectonic setting.
  2. Generate a feature-conditioned synthetic subsurface with GeoGen.
  3. Stitch the synthetic subsurface against the satellite DEM and forward
     model gravity into a SampleTile that conforms to GEOSCIENCE_IO_SPEC.md.
  4. Save to .npz + .json sidecar.
  5. Render a 2x2 satellite/DEM/NDVI/gravity figure (matplotlib).
  6. Open a 3D PyVista viewer with topography and subsurface density.

Run::

    pip install -e .[gis]
    python code_examples/build_and_view_sample_tile.py --region southern_alps

Use ``--offline`` to skip MPC and synthesize a DEM (handy for smoke testing
the visualization without network or GIS extras installed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geogen.gis import (
    NZ_REGIONS,
    REGION_LOOKUP,
    build_sample_tile,
    extract_features,
    generator_for_tile,
    save_sample_tile,
)
from geogen.gis.mpc import TILE_SIZE_M, Tile


def _synthetic_tile(region):
    """Synthetic 256x256 DEM matching the real tile shape, no network required."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:256, 0:256] * (TILE_SIZE_M / 256.0)
    dem = (
        700.0
        + 900.0 * np.sin(xx / 1500.0)
        + 700.0 * np.cos(yy / 1700.0)
        + rng.normal(0.0, 30.0, (256, 256))
    ).astype(np.float32)
    return Tile(
        region=region,
        center_lonlat=(0.5 * (region.bbox[0] + region.bbox[2]),
                       0.5 * (region.bbox[1] + region.bbox[3])),
        bbox_nztm=(0.0, 0.0, TILE_SIZE_M, TILE_SIZE_M),
        dem=dem,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="southern_alps",
                        choices=list(REGION_LOOKUP.keys()))
    parser.add_argument("--offline", action="store_true",
                        help="Skip MPC; use a synthetic DEM.")
    parser.add_argument("--out", default="outputs/sample_tile",
                        help="Output path stem (.npz + .json appended).")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip matplotlib + PyVista rendering.")
    parser.add_argument("--save-figure", action="store_true",
                        help="Save the matplotlib figure as <out>_panel.png.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density-mode", default="slice",
                        choices=["slice", "corner", "y_half", "x_half", "none"],
                        help="How to expose the subsurface in the 3D view.")
    parser.add_argument("--z-exag", type=float, default=1.0,
                        help="Vertical exaggeration for the 3D topography.")
    args = parser.parse_args()

    region = REGION_LOOKUP[args.region]
    print(f"Region: {region.name}")
    print(f"  {region.description}")

    # 1. Acquire tile
    if args.offline:
        print("[offline] using synthetic DEM (no MPC, no S2)")
        tile = _synthetic_tile(region)
    else:
        from geogen.gis.mpc import PlanetaryComputerClient, sample_tile_centers
        print(f"Fetching tile from Microsoft Planetary Computer...")
        client = PlanetaryComputerClient()
        center = tuple(sample_tile_centers(region, 1, seed=args.seed)[0].tolist())
        tile = client.fetch_tile(region, center, with_s2=True)
        print(f"  center lon/lat: {tile.center_lonlat}")
        print(f"  DEM shape: {tile.dem.shape}, relief={float(np.ptp(tile.dem)):.0f} m")

    # 2. Feature-conditioned subsurface synthesis
    feats = extract_features(tile)
    print(f"\nFeatures: relief={feats.relief_m:.0f} m, slope_mean={feats.slope_mean_deg:.1f}, "
          f"lin_strength={feats.lineament_strength:.2f}, lin_az={feats.lineament_azimuth_deg:.0f} deg")

    gen = generator_for_tile(tile, features=feats)
    model = gen.generate_model()
    model.fill_nans()
    litho_xyz = model.get_data_grid()
    print(f"  GeoGen lithology shape: {litho_xyz.shape}, unique codes: {np.unique(litho_xyz).tolist()}")

    # 3. Build spec-compliant sample tile
    skip_geo = args.offline  # synthetic NZTM bbox isn't a real lon/lat
    sample = build_sample_tile(
        tile, litho_xyz,
        terrain_id=f"sample-{region.name}",
        density_seed=args.seed,
        skip_geographic=skip_geo,
    )
    print(f"\nSample tile built:")
    print(f"  surface  : {sample.surface.shape}, range [{sample.surface.min():.0f}, {sample.surface.max():.0f}] m")
    print(f"  target   : {sample.target.shape}, range [{sample.target.min():.0f}, {sample.target.max():.0f}] kg/m^3 contrast")
    print(f"  gravity  : {sample.gravity.shape}, range [{sample.gravity.min():.2f}, {sample.gravity.max():.2f}] mGal")
    if not skip_geo:
        print(f"  lat/lon  : center ({sample.latitude.mean():.4f}, {sample.longitude.mean():.4f})")

    # 4. Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    npz_path = save_sample_tile(sample, out)
    print(f"\nSaved: {npz_path}  +  {out.with_suffix('.json')}")

    # 5 + 6. Visualize
    if args.no_viz:
        return
    try:
        from geogen.gis import viz
    except ImportError as e:
        print(f"[viz] skipped: {e}")
        return

    print("\nBuilding 2D satellite panel...")
    fig = viz.satellite_panel(sample, source_tile=tile)
    if args.save_figure:
        fig_path = out.parent / f"{out.name}_panel.png"
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        print(f"  saved {fig_path}")
    try:
        import matplotlib.pyplot as plt
        plt.show()
    except Exception as e:
        print(f"  could not show matplotlib window: {e}")

    print("\nLaunching 3D PyVista viewer (q to close)...")
    try:
        plotter = viz.pyvista_terrain_view(
            sample, source_tile=tile,
            density_mode=args.density_mode,
            z_exaggeration=args.z_exag,
        )
        plotter.show()
    except Exception as e:
        print(f"  PyVista failed: {e}")


if __name__ == "__main__":
    main()
