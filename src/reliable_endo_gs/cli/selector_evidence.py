"""Derive the repair Gate A report from hashed native development evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--native-evaluation", type=Path)
    mode.add_argument("--study-dir", type=Path)
    parser.add_argument("--evaluation-index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.study_dir is not None:
        if args.evaluation_index is None:
            parser.error("promotion requires --evaluation-index for every chosen artifact/seed")
        from reliable_endo_gs.evaluation.selector_promotion import evaluate_selector_promotion

        result = evaluate_selector_promotion(
            study_dir=args.study_dir,
            evaluation_index=args.evaluation_index,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, indent=2))
        return 0
    from reliable_endo_gs.evaluation.selector_evidence import stage_a_gate
    from reliable_endo_gs.utils.io import write_json_atomic

    repo = Path(__file__).resolve().parents[3]
    if (
        args.output_dir.exists()
        or repo == args.output_dir.resolve()
        or repo in args.output_dir.resolve().parents
    ):
        raise ValueError("gate evidence requires a new external output directory")
    result = stage_a_gate(args.native_evaluation)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(args.output_dir / "gate_a.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
