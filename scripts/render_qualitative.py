"""Thin entry point: render the frozen qualitative frames with one frozen model (see reliable_endo_gs.evaluation.qualitative).

usage: render_qualitative.py SPEC MODEL FRAMES.json OUT_DIR [--reference DIR] [--smoke]   (GPU pinned by the caller's environment)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reliable_endo_gs.evaluation.qualitative import run_qualitative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("model")
    parser.add_argument("frames", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    spec = json.loads(args.spec.read_text())
    run_qualitative(
        spec,
        args.model,
        args.frames,
        args.out_dir,
        reference_dir=args.reference,
        smoke=args.smoke,
        device=torch.device("cuda:0"),
    )


if __name__ == "__main__":
    main()
