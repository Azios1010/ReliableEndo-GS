"""Audit TRAIN utility coefficients without fitting a model or using final test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from reliable_endo_gs.evaluation.utility_sensitivity import audit_utility_coefficients

    result = audit_utility_coefficients(
        outcomes=args.outcomes, partition_path=args.partition, output_dir=args.output_dir
    )
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
