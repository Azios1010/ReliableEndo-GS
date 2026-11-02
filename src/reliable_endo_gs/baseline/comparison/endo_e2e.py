"""Endo-E2E-GS behind the comparison interface: the matched retrained model (render + geometry) and the released SCARED-C checkpoints.

* matched: final Stage-2 checkpoint of the retraining on train_1236 at 512x640, upstream Softplus/clamp kept, exact-zero scale ->
  float32 tiny (a disclosed numerical recovery policy). Native parity with the paper is NOT verified.
* released: Stage-1 / Stage-2 SCARED-C weights trained on 1280x1024 inputs: inference at 1024x1280, disparity reduced to the
  512x640 grid with nearest-exact selection and divided by 2 (disclosed bridge). Geometry only: the Stage-2 Gaussian head was trained
  on min/max-normalized depth, so no rendering is reported for it.
Launch from a process whose ``reliable_endo_gs`` is the pinned snapshot; the upstream repository is imported through the project's
pinned-upstream helpers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from reliable_endo_gs.baseline.a00_model import A00EndoModel, build_a00_endo_model, render_a00_endo
from reliable_endo_gs.baseline.upstream import import_upstream_module, require_pinned_upstream
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, ScaredA00Sample
from reliable_endo_gs.evaluation.multi_model import Prediction
from reliable_endo_gs.rendering.production import ProductionFieldRenderer


def _checked_state(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = module.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint parameter/buffer keys differ")
    for key, tensor in state.items():
        if tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype:
            raise ValueError("checkpoint shape/dtype mismatch: " + key)
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("non-finite checkpoint tensor: " + key)
    module.load_state_dict(state, strict=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class MatchedEndoPredictor(nn.Module):
    def __init__(self, model: A00EndoModel, renderer: ProductionFieldRenderer) -> None:
        super().__init__()
        self.model, self.renderer = model, renderer

    def forward(self, sample: ScaredA00Sample) -> Prediction:
        out = self.model(sample.batch, with_gaussians=True, mixed_precision=True)
        h, w = sample.batch.left.shape[-2:]
        return Prediction(
            out.disparity,
            render_a00_endo(out, sample.batch, self.renderer, view="left"),
            render_a00_endo(out, sample.batch, self.renderer, view="right"),
            out.field,
            torch.arange(h * w, device=out.disparity.device).unsqueeze(0),
        )


def build_matched(
    checkpoint: Path, expected_sha256: str, upstream: Path, device: torch.device
) -> MatchedEndoPredictor:
    if sha256_file(checkpoint) != expected_sha256:
        raise ValueError("matched baseline checkpoint hash differs from the frozen one")
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (raw["stage"], raw["stage_updates"], raw["total_updates"]) != ("stage2", 10000, 110100):
        raise ValueError("matched baseline Stage 2 is not complete")
    if raw["identity"]["frozen"]["scale_underflow_policy"] != "zero_to_float32_tiny":
        raise ValueError("matched baseline numerical policy differs")
    model = build_a00_endo_model(
        upstream, corr_implementation="reg", scale_underflow_policy="zero_to_float32_tiny"
    )
    _checked_state(model.native, raw["network"])
    config = model.native.cfg
    config.defrost()
    config.raft.mixed_precision = True
    config.freeze()
    return (
        MatchedEndoPredictor(model, ProductionFieldRenderer())
        .to(device)
        .eval()
        .requires_grad_(False)
    )


class ReleasedEndoPredictor(nn.Module):
    """Geometry-only bridge for a released SCARED-C checkpoint (needs the full-resolution sample)."""

    def __init__(
        self,
        model: A00EndoModel,
        reload_full: Callable[[ScaredA00Sample], ScaredA00Sample],
    ) -> None:
        super().__init__()
        self.model, self.reload_full = model, reload_full

    def forward(self, sample: ScaredA00Sample) -> Prediction:
        full = self.reload_full(sample)
        disparity = self.model(full.batch).disparity
        disparity = (
            F.interpolate(disparity, size=tuple(sample.batch.left.shape[-2:]), mode="nearest-exact")
            / 2.0
        )
        return Prediction(disparity)


def build_released(
    checkpoint: Path,
    expected_sha256: str,
    upstream: Path,
    adapter: ScaredA00Adapter,
    data_role: str,
    device: torch.device,
) -> ReleasedEndoPredictor:
    if sha256_file(checkpoint) != expected_sha256:
        raise ValueError("released checkpoint hash differs from the frozen one")
    require_pinned_upstream(upstream)
    config_module = import_upstream_module("config.stereo_config", upstream)
    loader = config_module.ConfigStereo()
    loader.load(str(upstream / "config/stage2.yaml"))
    config = loader.get_cfg()
    config.defrost()
    config.stage1_ckpt = None
    config.restore_ckpt = None
    config.batch_size = 1
    config.raft.mixed_precision = False
    config.raft.corr_implementation = (
        "reg"  # Stage 1 was trained with the PyTorch-equivalent correlation
    )
    config.freeze()
    network = import_upstream_module("lib.network", upstream, require_corr_sampler=False)
    losses = import_upstream_module("lib.loss", upstream)
    native = network.StereoEndoModel(config, with_gs_render=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = native.load_state_dict(payload["network"], strict=False)
    missing = sorted({key.split(".")[0] for key in result.missing_keys})
    if list(result.unexpected_keys) or not set(missing) <= {"gs_parm_regresser"}:
        raise ValueError(f"unexpected checkpoint/model mismatch: missing={missing}")
    model = (
        A00EndoModel(
            native,
            losses.sequence_loss,
            train_iters=int(config.raft.train_iters),
            val_iters=int(config.raft.val_iters),
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )

    def reload_full(sample: ScaredA00Sample) -> ScaredA00Sample:
        return adapter.load(
            sample.sample_id,
            data_role=data_role,
            include_supervision=False,
            output_hw=(1024, 1280),
            device=device,
        )

    return ReleasedEndoPredictor(model, reload_full)
