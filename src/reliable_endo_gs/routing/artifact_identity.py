"""Execution identity is independent of selector heads, utility and fit seeds."""

from __future__ import annotations

import hashlib
from pathlib import Path

from reliable_endo_gs.config.intervention import InterventionConfig


def repair_execution_config(config: InterventionConfig) -> dict[str, object]:
    """Exactly the intervention settings consumed by RepairActions construction.

    Routing and reconstruction construction are bound separately. Training losses,
    physical-label policies and selector topology cannot change an action forward.
    """
    return {
        name: getattr(config, name)
        for name in (
            "matching_width",
            "search_radius_px",
            "search_step_px",
            "matching_temperature",
            "repair_max_disparity_px",
            "max_effective_fraction",
        )
    }


def repair_execution_source_sha256() -> str:
    """Pin forward/geometry/renderer/data code without coupling identity to selector fitting.

    Parent and current full source hashes remain separate provenance fields. An
    action implementation change must invalidate labels even if tensors match.
    """
    root = Path(__file__).resolve().parents[1]
    files = [root / "data/scared_a00.py"]
    for area in (
        "contracts",
        "reconstruction",
        "geometry",
        "probabilistic_gs",
        "rendering",
        "actions",
        "uncertainty",
    ):
        files.extend((root / area).rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
