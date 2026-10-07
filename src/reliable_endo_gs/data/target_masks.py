"""Explicit external target masks, bound to sample/grid and producer identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
import torch

from reliable_endo_gs.contracts.regions import TargetROI


class TargetMaskIndex:
    """Immutable index of SHA-256-pinned bool HxW .npy deployment masks.

    No implicit resizing, GT fallback or foreground inference. Paths resolve
    relative to the external index directory. Declarations of mask origin are
    recorded, not a substitute for auditing the upstream producer.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        repo = Path(__file__).resolve().parents[3]
        if self.path == repo or repo in self.path.parents:
            raise ValueError("target masks and their index must live outside the repository")
        content = self.path.read_bytes()
        raw = json.loads(content)
        if not isinstance(raw, dict) or set(raw) != {"schema", "samples"}:
            raise ValueError("target mask index needs schema and samples")
        if raw["schema"] != "deployment_target_masks.v1" or not isinstance(raw["samples"], dict):
            raise ValueError("unsupported target mask index schema")
        entries: dict[str, dict[str, object]] = {}
        for sample_id, entry in raw["samples"].items():
            if (
                not isinstance(sample_id, str)
                or not sample_id.strip()
                or not isinstance(entry, dict)
                or set(entry) != {"path", "sha256", "output_hw", "source", "source_identity"}
                or entry["source"] not in ("input_roi", "rgb_prediction")
                or not isinstance(entry["source_identity"], str)
                or not entry["source_identity"].strip()
                or not isinstance(entry["path"], str)
                or not entry["path"].strip()
                or not isinstance(entry["sha256"], str)
                or len(entry["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in entry["sha256"])
                or not isinstance(entry["output_hw"], list)
                or len(entry["output_hw"]) != 2
                or any(type(v) is not int or v < 1 for v in entry["output_hw"])
            ):
                raise ValueError(f"invalid deployment target-mask entry: {sample_id}")
            entries[sample_id] = entry
            array_path = (self.path.parent / entry["path"]).resolve()
            if array_path == repo or repo in array_path.parents:
                raise ValueError("target mask arrays must live outside the repository")
        self._entries = MappingProxyType(entries)
        self.identity = hashlib.sha256(content).hexdigest()

    def require_samples(self, sample_ids: Sequence[str], image_hw: tuple[int, int]) -> None:
        """Check index coverage/paths/grid before constructing GPU or teacher assets.

        File bytes and shape are independently verified by load at consumption.
        """
        missing = [s for s in sample_ids if s not in self._entries]
        if missing:
            raise ValueError(f"missing deployment target masks: {len(missing)}; first={missing[0]}")
        for sample_id in sample_ids:
            entry = self._entries[sample_id]
            if tuple(cast(list[int], entry["output_hw"])) != image_hw:
                raise ValueError(f"target mask grid mismatch: {sample_id}")
            path = (self.path.parent / cast(str, entry["path"])).resolve()
            if not path.is_file():
                raise ValueError(f"target mask file missing: {sample_id}")

    def load(self, sample_id: str, image_hw: tuple[int, int], device: torch.device) -> TargetROI:
        """Verify bytes, exact grid and bool values on every sample load."""
        if sample_id not in self._entries:
            raise ValueError(f"missing deployment target ROI for {sample_id}")
        entry = self._entries[sample_id]
        if tuple(cast(list[int], entry["output_hw"])) != image_hw:
            raise ValueError("target mask grid differs from model output; no implicit resize")
        path = (self.path.parent / cast(str, entry["path"])).resolve()
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"target mask SHA-256 mismatch: {sample_id}")
        import io

        array = np.load(io.BytesIO(content), allow_pickle=False)
        if array.shape != image_hw or array.dtype != np.bool_:
            raise ValueError("target mask must be bool HxW on declared output grid")
        return TargetROI(
            torch.from_numpy(array.copy()).to(device)[None, None],
            cast(Literal["input_roi", "rgb_prediction"], entry["source"]),
            f"{entry['source_identity']};mask_sha256={entry['sha256']}",
        )
