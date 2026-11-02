"""Thin entry point: right-view metrics on the part visible from the left image, one frozen model.

usage: evaluate_visible_region.py SPEC.json MODEL OUT_DIR [--reference ROWS.jsonl] [--every 20] [--limit N] [--smoke]   (GPU pinned by the caller's environment)
The reference model (ours_base) computes the strip widths and takes no --reference; every other model reads them from the reference rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reliable_endo_gs.evaluation.visible_region import run_visible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("model")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--every", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    result = run_visible(
        json.loads(args.spec.read_text()),
        args.model,
        args.out_dir,
        reference_rows=args.reference,
        every=args.every,
        limit=args.limit,
        smoke=args.smoke,
        device=torch.device("cuda:0"),
    )
    print(
        json.dumps({k: result[k] for k in ("model", "frames_in_file", "wall_seconds_this_process")})
    )


if __name__ == "__main__":
    main()
