"""
Example: stream imagery-conditioned synthetic geology samples for NZ.

Pulls 7.68 x 7.68 km tiles from the Microsoft Planetary Computer for several
NZ tectonic settings, extracts structural-geology features from each tile's
DEM (and Sentinel-2, if available), then generates plausible 3D subsurface
realizations whose lithology + density volumes are biased toward the
structural style of each setting.

Run::

    pip install -e .[gis]
    python code_examples/nz_gis_dataset_example.py

The full run hits MPC and takes a couple of minutes. To dry-run the
generation pipeline without network, use ``--offline`` to substitute a
synthetic DEM.
"""

from __future__ import annotations

import argparse

import numpy as np

from geogen.gis import (
    NZ_REGIONS,
    NZGISConditionedDataset,
    extract_features,
    generator_for_tile,
    lithology_to_density,
)
from geogen.gis.mpc import TILE_SIZE_M, Tile


def _synthetic_tile(region):
    """Build a fake 256x256 DEM so the rest of the pipeline can run offline."""
    yy, xx = np.mgrid[0:256, 0:256] * (TILE_SIZE_M / 256.0)
    dem = (
        500.0
        + 800.0 * np.sin(xx / 1500.0)
        + 600.0 * np.cos(yy / 1800.0)
        + np.random.default_rng(0).normal(0.0, 25.0, (256, 256))
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
    parser.add_argument("--offline", action="store_true",
                        help="Skip MPC; use a synthetic DEM (for smoke testing).")
    parser.add_argument("--realizations", type=int, default=2,
                        help="Subsurface realizations per tile.")
    args = parser.parse_args()

    if args.offline:
        print("[offline] running with a synthetic DEM, no network access\n")
        for region in NZ_REGIONS:
            tile = _synthetic_tile(region)
            feats = extract_features(tile)
            gen = generator_for_tile(tile, features=feats)
            print(f"== {region.name} ==")
            print(f"  features: relief={feats.relief_m:.0f} m  "
                  f"slope_mean={feats.slope_mean_deg:.1f}  "
                  f"lin_strength={feats.lineament_strength:.2f}")
            for k in range(args.realizations):
                model = gen.generate_model()
                model.fill_nans()
                litho = model.get_data_grid()
                density = lithology_to_density(litho, seed=k)
                print(f"    realization {k}: lithology unique={np.unique(litho).size}, "
                      f"density mean={density[density > 0].mean():.0f} kg/m^3")
        return

    print("[online] fetching tiles from Microsoft Planetary Computer...")
    ds = NZGISConditionedDataset(
        tiles_per_region=1,
        realizations_per_tile=args.realizations,
        with_s2=True,
    )
    print(f"Dataset has {len(ds)} samples across {len(NZ_REGIONS)} NZ tectonic settings.\n")
    for i in range(min(len(ds), 3 * args.realizations)):
        s = ds[i]
        print(f"sample {i:>3} | region={s.region_name:<25} "
              f"DEM relief={s.features.relief_m:.0f} m | "
              f"density mean={s.density.mean().item():.0f} kg/m^3 | "
              f"litho shape={tuple(s.lithology.shape)}")


if __name__ == "__main__":
    main()
