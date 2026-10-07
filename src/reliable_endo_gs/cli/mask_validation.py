"""Explicit small native joint-mask validation with CPU preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-root",
        "training-manifest",
        "development-manifest",
        "worklist-path",
        "target-mask-index",
        "cache-root",
        "initial-checkpoint",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--roles-path", type=Path, default=Path("configs/teachers/a01_scared.yaml"))
    parser.add_argument(
        "--mask-config-path", type=Path, default=Path("configs/routing/target_masks.yaml")
    )
    parser.add_argument(
        "--intervention-config-path",
        type=Path,
        default=Path("configs/routing/intervention_balanced.yaml"),
    )
    parser.add_argument(
        "--routing-config-path", type=Path, default=Path("configs/routing/frozen_a04.yaml")
    )
    parser.add_argument(
        "--metric-config-path",
        type=Path,
        default=Path("configs/routing/metric_tradeoff_balanced.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    from reliable_endo_gs.training.mask_validation import run_mask_validation

    try:
        result = run_mask_validation(**vars(args))
    except (OSError, ValueError, RuntimeError, KeyError, AssertionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
