"""HK-Shi bidirectional semi-supervised dual-branch CNN (official SCARED weights) behind the comparison interface (geometry only).

Inference follows the official ``test.py``: ImageNet normalisation, no padding for the 16-divisible SCARED grid, both branches
evaluated and the one with the larger summed confidence kept. The network was trained at the native 1024x1280 resolution, so
(as for the released Endo-E2E-GS weights) it runs at 1024x1280 and the disparity is reduced to the 512x640 protocol grid with
nearest-exact selection and divided by 2 (disclosed bridge). Launch from the HK-Shi repository directory with its ``models``
package importable; the weights file is hashed by the caller.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, ScaredA00Sample
from reliable_endo_gs.evaluation.multi_model import Prediction

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class HkShiPredictor(nn.Module):
    def __init__(
        self, network: nn.Module, reload_full: Callable[[ScaredA00Sample], ScaredA00Sample]
    ) -> None:
        super().__init__()
        self.network, self.reload_full = network, reload_full
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, sample: ScaredA00Sample) -> Prediction:
        full = self.reload_full(sample)
        left = (full.batch.left - self.mean) / self.std
        right = (full.batch.right - self.mean) / self.std
        estimates_l, confidence_l = self.network(left, right, step=1)
        estimates_r, confidence_r = self.network(left, right, step=2)
        if torch.sum(confidence_l[0]) >= torch.sum(confidence_r[0]):
            disparity = estimates_l[0]
        else:
            disparity = estimates_r[0]
        disparity = disparity.reshape(1, 1, *disparity.shape[-2:]).float()
        reduced = (
            F.interpolate(disparity, size=tuple(sample.batch.left.shape[-2:]), mode="nearest-exact")
            / 2.0
        )
        return Prediction(reduced)


def build_stereo_predictor(
    entry: dict[str, Any], adapter: ScaredA00Adapter, data_role: str, device: torch.device
) -> nn.Module:
    if entry.get("model") != "hkshi":
        raise ValueError("only the HK-Shi network is wired; MSDESIS weights could not be retrieved")
    upstream = Path(entry["upstream"])
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != entry["upstream_commit"]:
        raise ValueError("HK-Shi upstream commit differs from the frozen one")
    weights = Path(entry["checkpoint"])
    if hashlib.sha256(weights.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("HK-Shi checkpoint hash differs from the frozen one")
    models = importlib.import_module("models")
    network = models.__models__["Bidirectional-DualBranch-CNN"](192)
    state = torch.load(weights, map_location="cpu", weights_only=False)["model"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    network.load_state_dict(state, strict=True)

    def reload_full(sample: ScaredA00Sample) -> ScaredA00Sample:
        return adapter.load(
            sample.sample_id,
            data_role=data_role,
            include_supervision=False,
            output_hw=(1024, 1280),
            device=device,
        )

    return HkShiPredictor(network, reload_full).to(device).eval().requires_grad_(False)
