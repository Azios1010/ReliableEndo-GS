"""Thin entry point: evaluate one model on one split (see reliable_endo_gs.evaluation.comparison_runner).

usage: evaluate_comparison_model.py SPEC.json MODEL val|test OUT_DIR [--shift] [--limit N]   (GPU pinned by the caller's environment)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reliable_endo_gs.evaluation.comparison_runner import run_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("model")
    parser.add_argument("split", choices=("val", "test"))
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--every", type=int, default=None)
    parser.add_argument("--memory-fraction", type=float, default=0.30)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    spec = json.loads(args.spec.read_text())
    result = run_model(
        spec,
        args.model,
        args.split,
        args.out_dir,
        shift=args.shift,
        every=args.every,
        limit=args.limit,
        device=torch.device("cuda:0"),
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("model", "split", "frames_in_file", "wall_seconds_this_process")
            }
        )
    )


if __name__ == "__main__":
    main()
