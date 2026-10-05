"""Student learning, joint learning and teacher-free student deployment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Explicit assets; checkpoints pin inference construction and learning config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "train", "infer"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--routing-config", type=Path)
    parser.add_argument("--construction-config", type=Path)
    parser.add_argument("--roles", type=Path, default=Path("configs/teachers/a01_scared.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Strict resume or deployment checkpoint")
    parser.add_argument("--max-steps", type=int, help="Bound consumed training samples")
    parser.add_argument("--sample-id")
    parser.add_argument("--data-role", choices=("final_train", "final_test"), default="final_train")
    parser.add_argument(
        "--selected", action="store_true", help="Execute one selected repair at deployment"
    )
    parser.add_argument(
        "--teacher-factory", help="Optional training-only module:function KD teacher"
    )
    parser.add_argument(
        "--cache-root", type=Path, help="Read existing pinned KD cache; missing items fail"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Resolve scientific/runtime work only after argument parsing."""
    args = build_parser().parse_args(argv)
    from reliable_endo_gs.training.student_runtime import run_student_command

    try:
        result = run_student_command(args)
    except (OSError, ValueError, RuntimeError, KeyError, AssertionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
