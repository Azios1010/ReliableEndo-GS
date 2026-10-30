"""Audit/gate and fit controlled variants from one frozen TRAIN outcome store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("collection-dir", "study-protocol-path", "refit-config-path", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gate-a-report", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--device-name", default="cpu")
    args = parser.parse_args(argv)
    from reliable_endo_gs.training.selector_study import run_selector_study

    result = run_selector_study(**vars(args))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("status", "selector_fitted", "promotion", "FPS")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
