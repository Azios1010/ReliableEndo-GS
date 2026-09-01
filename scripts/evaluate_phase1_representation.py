"""Thin local entry point for Plan 07 representation contract inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/p1d_probabilistic_gs.yaml"),
        help="Plan 07 experiment configuration path",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def main() -> int:
    """Print local-only configuration status; do not run a scientific experiment."""

    args = _parse_args()
    config = _load_config(args.config)
    print(
        json.dumps(
            {
                "status": "LOCAL_CONTRACT_ONLY",
                "config": str(args.config),
                "schema_version": config.get("schema_version"),
                "scientific_selection": config.get("scientific_selection"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
