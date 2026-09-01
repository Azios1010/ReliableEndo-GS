"""Validate the bounded local Plan 08 evaluation envelope without data access."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    value = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Plan-I evaluation configuration must be a mapping")
    if value.get("mode") != "development_only" or value.get("device") != "cpu":
        raise ValueError("local Plan-I evaluation must be development_only on cpu")
    if value.get("scientific_status") != "development_only":
        raise ValueError("evaluation status must be development_only")
    print(
        "validated local Plan-I cross-view evaluation protocol; no real data, "
        "network, GPU, or production renderer was accessed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
