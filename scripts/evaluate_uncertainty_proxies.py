"""Thin entry point for the configured Phase-I proxy evaluation protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    """Validate the experiment envelope; data execution needs an accepted artifact adapter."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    loaded = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("experiment"), dict):
        raise ValueError("config must contain an experiment mapping")
    name = loaded["experiment"].get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("experiment.name must be a non-empty string")
    print(
        f"validated uncertainty proxy protocol {name!r}; "
        "provide frozen baseline evidence through a future artifact adapter to execute it"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
