"""Generate N stratiflow synthetic-dataset samples.

Default: 5 samples in ./stratiflow_samples/sample_001..005/

Usage::

    python code_examples/build_stratiflow_samples.py
    python code_examples/build_stratiflow_samples.py --n 3 --out other_dir/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from geogen.stratiflow import GenerationConfig, generate_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of samples")
    parser.add_argument("--out", default="stratiflow_samples",
                        help="Output directory root")
    parser.add_argument("--seed-base", type=int, default=20260502,
                        help="Each sample uses seed-base + i for sample i.")
    parser.add_argument("--n-boreholes", type=int, default=8)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for i in range(args.n):
        sample_id = f"sample_{i + 1:03d}"
        sample_dir = out_root / sample_id
        cfg = GenerationConfig(
            seed=args.seed_base + i,
            n_boreholes=args.n_boreholes,
        )
        print(f"\n=== {sample_id}  (seed={cfg.seed}) ===")
        t0 = time.time()
        try:
            summary = generate_sample(cfg, sample_dir, sample_id=sample_id)
            summary["sample_id"] = sample_id
            summary["seconds"] = round(time.time() - t0, 1)
            summaries.append(summary)
            print(
                f"  units={summary['n_units']} "
                f"faults={summary['n_faults']} "
                f"boreholes={summary['n_boreholes']} "
                f"relief={summary['surface_relief_m']:.0f} m "
                f"({summary['seconds']}s)"
            )
            for w in summary.get("qc_warnings", []):
                print(f"  [warn] {w}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            summaries.append({"sample_id": sample_id, "error": str(e)})

    index_path = out_root / "index.json"
    index_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote summary index: {index_path}")


if __name__ == "__main__":
    main()
