"""Frozen repair artifact: TRAIN gain/risk refit and native DEV comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "data-root",
        "training-manifest",
        "development-manifest",
        "roles-path",
        "worklist-path",
        "refit-config-path",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--target-mask-index", type=Path)
    parser.add_argument("--study-protocol-path", type=Path)
    parser.add_argument("--gate-a-report", type=Path)
    parser.add_argument("--cost-calibration-path", type=Path)
    parser.add_argument("--device-name", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="collect frozen TRAIN outcomes, then stop before selector fitting",
    )
    parser.add_argument("--routing-profile-reference", type=Path)
    parser.add_argument("--routing-profile-selector", type=Path)
    parser.add_argument(
        "--prepare-costs",
        action="store_true",
        help="GT-free native TRAIN cost calibration, then stop",
    )
    parser.add_argument(
        "--resume", action="store_true", help="verify and reuse completed immutable shards/stages"
    )
    args = parser.parse_args(argv)
    from reliable_endo_gs.training.selector_refit import run_selector_refit

    try:
        result = run_selector_refit(**vars(args))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise
    display = dict(result)
    if "partition" in display:
        partition = display["partition"]
        # Complete identities are sealed on disk; avoid printing thousands of IDs.
        display["partition"] = {
            "partition_sha256": partition["partition_sha256"],
            "roles": {
                name: {
                    "samples": len(partition[name + "_sample_ids"]),
                    "keyframes": partition[name + "_keyframes"],
                }
                for name in ("fit", "validation", "calibration")
            },
        }
    print(json.dumps(display, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
