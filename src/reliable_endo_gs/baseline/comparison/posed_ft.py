"""Fine-tuned MVSplat / DepthSplat behind the comparison interface (principal-point projection applied, fp32 inference)."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from reliable_endo_gs.baseline.finetune.projection import apply_principal_point_projection
from reliable_endo_gs.baseline.posed_gaussians_native import build_posed_predictor
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.evaluation.multi_model import Prediction


class PosedPredictor(nn.Module):
    """Wraps the native predictor: native renders of the two input cameras, plane-sweep disparity, diagnostic field."""

    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, sample: ScaredA00Sample) -> Prediction:
        out = self.predictor(sample)
        # Native spherical-harmonic colours can leave [0, 1]; every novel-view evaluation clamps the render before the metrics.
        left = replace(out.left, image=out.left.image.clamp(0.0, 1.0))
        right = replace(out.right, image=out.right.image.clamp(0.0, 1.0))
        return Prediction(out.disparity, left, right, out.field, out.source_index)


def build_finetuned(
    kind: str,
    upstream: Path,
    upstream_commit: str,
    official_checkpoint: Path,
    finetuned_final: Path | None,
    official_sha256: str,
    finetuned_sha256: str | None,
    near_m: float,
    far_m: float,
    device: torch.device,
    left_source_only: bool = False,
) -> PosedPredictor:
    """Official architecture and constants, then the fine-tuned encoder weights (strict). ``finetuned_final=None`` = zero-shot."""
    if hashlib.sha256(official_checkpoint.read_bytes()).hexdigest() != official_sha256:
        raise ValueError("official checkpoint hash differs from the frozen one")
    predictor, _ = build_posed_predictor(
        kind, upstream, upstream_commit, official_checkpoint, near_m=near_m, far_m=far_m
    )
    if finetuned_final is not None:
        if hashlib.sha256(finetuned_final.read_bytes()).hexdigest() != finetuned_sha256:
            raise ValueError("fine-tuned checkpoint hash differs from the frozen one")
        payload = torch.load(finetuned_final, map_location="cpu", weights_only=False)
        predictor.encoder.load_state_dict(payload["encoder"], strict=True)
    if left_source_only:
        predictor.config = replace(predictor.config, left_source_only=True)
    apply_principal_point_projection()
    return PosedPredictor(predictor).to(device).eval().requires_grad_(False)
