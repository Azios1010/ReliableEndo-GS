"""Thin entry point for the development-only P1b reliability rescue study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reliable_endo_gs.reliability.p1b_runner import run_p1b


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run P1b reliability signal rescue")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research/p1b_reliability_rescue.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--enable-shadow", action="store_true")
    args = parser.parse_args(argv)
    result = run_p1b(
        repo=args.repo.resolve(),
        config_path=args.config.resolve(),
        output_path=args.output.resolve(),
        device=torch.device(args.device),
        max_samples=args.max_samples,
        enable_shadow=True if args.enable_shadow else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
