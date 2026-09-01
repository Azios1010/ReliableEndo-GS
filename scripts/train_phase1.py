"""Validate the bounded local Plan 08 training envelope without running data."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml


def _load(path: Path) -> Mapping[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Plan-I training configuration must be a mapping")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = _load(args.config)
    if config.get("mode") != "development_only":
        raise ValueError("local Plan-I training must declare mode: development_only")
    if config.get("device") != "cpu":
        raise ValueError("bounded Plan-I training must declare device: cpu")
    checkpoint = config.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("scientific_status") != "development_only"
    ):
        raise ValueError("Plan-I checkpoint status must be development_only")
    print(
        "validated local Plan-I training protocol; no dataset, network, GPU, "
        "native renderer, or epoch loop was executed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
