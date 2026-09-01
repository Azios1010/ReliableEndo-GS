"""Validate the bounded learned-uncertainty training protocol.

This entry point intentionally does not load real data, access a network, use a
GPU, or run the full Phase-I training loop.  A future artifact adapter owns
those operations after the protocol and split identities are accepted.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml


def _load_protocol(path: Path) -> Mapping[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("training protocol must be a mapping")
    experiment = raw.get("experiment")
    if not isinstance(experiment, Mapping) or not experiment.get("name"):
        raise ValueError("training protocol requires experiment.name")
    runtime = raw.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("device") != "cpu":
        raise ValueError("bounded training protocol must declare runtime.device: cpu")
    protocol = raw.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("training protocol requires a protocol mapping")
    for key in ("baseline_artifact_id", "training_split_hash", "calibration_split_hash"):
        value = protocol.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"protocol.{key} must be an explicit identity")
    if protocol.get("test_tuning_forbidden") is not True:
        raise ValueError("training protocol must forbid test tuning")
    return raw


def main(argv: Sequence[str] | None = None) -> int:
    """Validate configuration and report the intentionally bounded protocol."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        loaded = _load_protocol(arguments.config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.exit(2, f"error: {error}\n")
    experiment = loaded["experiment"]
    assert isinstance(experiment, Mapping)
    print(
        f"validated bounded uncertainty training protocol {experiment['name']!r}; "
        "no dataset, network, GPU, or full Phase-I loop was executed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
