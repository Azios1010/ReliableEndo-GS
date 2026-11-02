"""Thin entry point: pairwise 3D distance error of one frozen model on the SCARED test frames.

usage: evaluate_pair_distance_model.py SPEC.json MODEL OUT_DIR [--every 20] [--limit N] [--smoke]   (GPU pinned by the caller's environment)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reliable_endo_gs.evaluation.pair_distance_runner import run_pair_distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("model")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--every", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    result = run_pair_distance(
        json.loads(args.spec.read_text()),
        args.model,
        args.out_dir,
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
