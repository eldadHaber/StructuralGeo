"""One-off: fetch Sentinel-2 RGB for every stratiflow NZ sample under a root.

Reads each sample's `ground_truth/metadata.json` to recover the NZTM bbox
and grid shape, then fetches a cloud-low S2 RGB composite from MPC and
saves it to `inputs/satellite_rgb.npy` (float32, [3, ny, nx], reflectance
in [0, 1]). Skips samples that already have the file.

Usage::

    python code_examples/backfill_s2_for_stratiflow.py
    python code_examples/backfill_s2_for_stratiflow.py --root other_dir/ --force
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="stratiflow_samples_nz")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if satellite_rgb.npy exists.")
    args = parser.parse_args()

    from geogen.gis.mpc import PlanetaryComputerClient
    from geogen.stratiflow.nz_tile import fetch_s2_rgb_at_grid

    client = PlanetaryComputerClient()
    root = Path(args.root)
    sample_dirs = sorted(p for p in root.iterdir()
                         if p.is_dir() and (p / "ground_truth").is_dir())

    for sd in sample_dirs:
        out = sd / "inputs" / "satellite_rgb.npy"
        meta_path = sd / "ground_truth" / "metadata.json"
        if not meta_path.exists():
            continue
        if out.exists() and not args.force:
            print(f"  {sd.name}: already has satellite_rgb.npy (skip)")
            continue
        meta = json.loads(meta_path.read_text())
        prov = meta.get("provenance") or {}
        bbox = prov.get("bbox_nztm")
        if not bbox:
            print(f"  {sd.name}: no bbox_nztm in metadata, skipping")
            continue
        ny = meta["mesh"]["ny"]; nx = meta["mesh"]["nx"]
        t0 = time.time()
        try:
            rgb = fetch_s2_rgb_at_grid(client, tuple(bbox), ny=ny, nx=nx)
        except Exception as e:
            print(f"  {sd.name}: FAILED ({type(e).__name__}: {e!s:.80})")
            continue
        if rgb is None:
            print(f"  {sd.name}: no S2 scene available")
            continue
        np.save(out, rgb.astype(np.float32))
        coverage = 100.0 * float(np.isfinite(rgb).all(axis=0).mean())
        print(f"  {sd.name}: saved ({rgb.shape}, {coverage:.0f}% coverage, "
              f"{time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
