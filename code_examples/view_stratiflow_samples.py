"""Render visual-validation previews for stratiflow samples.

Produces ``<sample>/preview.png`` for each sample, plus a
``stratiflow_samples/montage.png`` stitching them together.

Usage::

    python code_examples/view_stratiflow_samples.py
    python code_examples/view_stratiflow_samples.py --root other_dir/
    python code_examples/view_stratiflow_samples.py --sample stratiflow_samples/sample_003
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Headless-friendly default; user can switch via env var.
matplotlib.use("Agg")

from geogen.stratiflow.viz import plot_all, plot_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="stratiflow_samples",
                        help="Directory containing sample_NNN subfolders.")
    parser.add_argument("--sample", default=None,
                        help="Render just this one sample directory.")
    parser.add_argument("--no-montage", action="store_true",
                        help="Skip the multi-sample montage image.")
    args = parser.parse_args()

    if args.sample:
        path = plot_sample(Path(args.sample))
        print(f"Wrote {path}")
        return

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Sample root does not exist: {root}")
    paths = plot_all(root, montage_path=None if not args.no_montage else False)
    print(f"\nRendered {len(paths['per_sample'])} per-sample previews.")
    if paths.get("montage"):
        print(f"Montage: {paths['montage']}")


if __name__ == "__main__":
    main()
