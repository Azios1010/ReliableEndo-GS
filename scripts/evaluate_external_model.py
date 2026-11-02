"""Thin entry point: zero-shot rendering evaluation of one frozen model on the external EndoNeRF sample data.

usage: evaluate_external_model.py SPEC.json MODEL OUT_DIR [--baseline-scale S] [--offset zero|fitted] [--limit N] [--smoke]
(GPU pinned by the caller's environment; see reliable_endo_gs.evaluation.external_runner)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reliable_endo_gs.evaluation.external_runner import run_external_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("model")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--baseline-scale", type=float, default=1.0)
    parser.add_argument("--offset", choices=("zero", "fitted"), default="zero")
    parser.add_argument("--every", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    result = run_external_model(
        json.loads(args.spec.read_text()),
        args.model,
        args.out_dir,
        baseline_scale=args.baseline_scale,
        offset_mode=args.offset,
        every=args.every,
        limit=args.limit,
        smoke=args.smoke,
        device=torch.device("cuda:0"),
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "model",
                    "frames_in_file",
                    "baseline_scale",
                    "offset_mode",
                    "wall_seconds_this_process",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
