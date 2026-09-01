"""Validate the validation-only uncertainty calibration protocol.

Calibration fitting is deliberately not run here: a future artifact adapter
must provide frozen validation tensors and their immutable split identity.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml


def _load_protocol(path: Path) -> Mapping[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("calibration protocol must be a mapping")
    experiment = raw.get("experiment")
    if not isinstance(experiment, Mapping) or not experiment.get("name"):
        raise ValueError("calibration protocol requires experiment.name")
    protocol = raw.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("calibration protocol requires a protocol mapping")
    if protocol.get("calibration_split_role") != "validation":
        raise ValueError("calibration must use validation split only")
    if protocol.get("test_tuning_forbidden") is not True:
        raise ValueError("calibration protocol must forbid test tuning")
    for key in ("baseline_artifact_id", "calibration_split_hash"):
        value = protocol.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"protocol.{key} must be an explicit identity")
    return raw


def main(argv: Sequence[str] | None = None) -> int:
    """Validate configuration without reading data or fitting parameters."""

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
        f"validated validation-only uncertainty calibration protocol {experiment['name']!r}; "
        "no ground truth, test split, network, or GPU was accessed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
