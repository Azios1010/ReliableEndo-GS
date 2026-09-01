"""CPU-only synthetic Plan 08 orchestration and checkpoint smoke tests."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from reliable_endo_gs.contracts import CameraBatch, GaussianField, StereoBatch, StereoPrediction
from reliable_endo_gs.rendering import ReferenceRenderer, RobustLossConfig, cross_view_loss
from reliable_endo_gs.training.phase1 import (
    BaselineOutput,
    Phase1CheckpointMetadata,
    Phase1Config,
    Phase1Trainer,
    PhaseIForward,
)
from reliable_endo_gs.uncertainty.records import CalibratedDisparitySigma


def _batch() -> StereoBatch:
    dtype = torch.float64
    left = torch.zeros((1, 3, 2, 2), dtype=dtype)
    right = torch.zeros_like(left)
    intrinsics = torch.tensor([[[4.0, 0.0, 0.5], [0.0, 5.0, 0.5], [0.0, 0.0, 1.0]]], dtype=dtype)
    left_transform = torch.eye(4, dtype=dtype).unsqueeze(0)
    right_transform = left_transform.clone()
    right_transform[:, 0, 3] = 0.25
    return StereoBatch(
        left=left,
        right=right,
        left_camera=CameraBatch(intrinsics, left_transform),
        right_camera=CameraBatch(intrinsics, right_transform),
        sample_ids=["synthetic"],
        sequence_ids=["sequence"],
    )


class FrozenBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def predict(self, batch: StereoBatch) -> BaselineOutput:
        dtype = batch.left.dtype
        device = batch.device
        means = (
            torch.tensor(
                [[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]]],
                dtype=dtype,
                device=device,
            )
            + self.anchor * 0.0
        )
        colors = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]],
            dtype=dtype,
            device=device,
        )
        rotations = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 4], dtype=dtype, device=device)
        scales = torch.full((1, 4, 3), 0.15, dtype=dtype, device=device)
        opacities = torch.full((1, 4, 1), 0.7, dtype=dtype, device=device)
        disparity = torch.full((1, 1, 2, 2), 2.0, dtype=dtype, device=device)
        stereo = StereoPrediction(
            disparity=disparity, valid_mask=torch.ones_like(disparity, dtype=torch.bool)
        )
        field = GaussianField(
            means3d=means,
            colors=colors,
            rotations=rotations,
            scales=scales,
            opacities=opacities,
            valid_mask=torch.ones((1, 4), dtype=torch.bool, device=device),
        )
        return BaselineOutput(stereo, field)


class TrainableSigma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.tensor(-0.2, dtype=torch.float64))

    def predict(self, prediction: StereoPrediction) -> CalibratedDisparitySigma:
        sigma = torch.nn.functional.softplus(self.raw) + 0.05
        values = sigma.expand_as(prediction.disparity)
        return CalibratedDisparitySigma(values, prediction.valid_mask, "synthetic", "calibration")


class NearZeroSigma:
    def predict(self, prediction: StereoPrediction) -> CalibratedDisparitySigma:
        values = torch.full_like(prediction.disparity, 1.0e-8)
        return CalibratedDisparitySigma(values, prediction.valid_mask, "near_zero", "calibration")


def _pipeline(baseline: FrozenBaseline, provider: TrainableSigma) -> PhaseIForward:
    return PhaseIForward(
        baseline,
        provider,
        ReferenceRenderer(radius_px=0.5),
        config=Phase1Config(
            baseline_m=0.5,
            loss=RobustLossConfig(kind="l1", left_weight=1.0, right_weight=2.0),
            config_identity="synthetic_phase1_config",
        ),
    )


def test_same_state_uses_distinct_camera_requests_and_provider_neutral_path() -> None:
    batch = _batch()
    baseline = FrozenBaseline()
    provider = TrainableSigma()
    result = _pipeline(baseline, provider).forward(batch)
    assert result.camera_left.view.value == "left"
    assert result.camera_right.view.value == "right"
    assert result.camera_left.camera is batch.left_camera
    assert result.camera_right.camera is batch.right_camera
    assert result.render_left.image.shape == batch.left.shape
    assert not torch.equal(result.render_left.image, result.render_right.image)
    assert result.representation.cov_center is not None
    assert result.provenance["scientific_status"] == "development_only"


def test_correct_right_camera_matches_target_better_than_reused_left_camera() -> None:
    batch = _batch()
    baseline = FrozenBaseline()
    provider = TrainableSigma()
    pipeline = _pipeline(baseline, provider)
    result = pipeline.forward(batch)
    correct = cross_view_loss(
        result.render_right,
        result.render_right.image.detach(),
        result.mask_right.combined_mask,
        RobustLossConfig(kind="l1"),
    )
    wrong = cross_view_loss(
        result.render_left,
        result.render_right.image.detach(),
        result.mask_right.combined_mask,
        RobustLossConfig(kind="l1"),
    )
    assert correct.loss is not None and correct.loss.item() == 0.0
    assert wrong.loss is not None and wrong.loss.item() > 0.0


def test_provider_neutrality_and_zero_uncertainty_recover_baseline_representation() -> None:
    batch = _batch()
    baseline = FrozenBaseline()
    first = _pipeline(baseline, TrainableSigma()).forward(batch)
    second = _pipeline(baseline, TrainableSigma()).forward(batch)
    assert torch.equal(first.sigma_geo, second.sigma_geo)
    assert torch.equal(first.representation.cov_effective, second.representation.cov_effective)
    assert torch.equal(first.render_left.image, second.render_left.image)

    zero_provider = NearZeroSigma()
    zero_pipeline = PhaseIForward(
        baseline,
        zero_provider,
        ReferenceRenderer(radius_px=0.5),
        config=Phase1Config(
            baseline_m=0.5,
            representation_variant="stereo_covariance_corrected",
            loss=RobustLossConfig(kind="l1"),
        ),
    )
    zero = zero_pipeline.forward(batch)
    assert zero.representation.cov_surface is not None
    assert zero.representation.cov_effective is not None
    assert torch.allclose(
        zero.representation.cov_surface, zero.representation.cov_effective, atol=1e-4
    )
    assert torch.allclose(
        zero.representation.effective_opacity, zero.representation.base_opacity, atol=1e-4
    )


def test_evaluate_restores_nested_provider_head_mode() -> None:
    batch = _batch()
    baseline = FrozenBaseline()
    provider = TrainableSigma()
    trainer = Phase1Trainer(
        _pipeline(baseline, provider),
        torch.optim.Adam(provider.parameters(), lr=0.01),
        trainable_modules={"uncertainty": provider},
    )
    provider.eval()
    trainer.evaluate(batch)
    assert provider.training is False


def test_one_step_gradients_frozen_baseline_and_checkpoint_resume(tmp_path: Path) -> None:
    torch.manual_seed(7)
    batch = _batch()
    baseline = FrozenBaseline()
    provider = TrainableSigma()
    pipeline = _pipeline(baseline, provider)
    optimizer = torch.optim.Adam(provider.parameters(), lr=0.01)
    trainer = Phase1Trainer(
        pipeline,
        optimizer,
        trainable_modules={"uncertainty": provider},
        frozen_modules={"baseline": baseline},
    )
    baseline_before = baseline.anchor.detach().clone()
    step = trainer.train_step(batch)
    assert torch.isfinite(torch.tensor(step.loss))
    assert step.gradient_norm > 0
    assert step.parameter_update_norm > 0
    assert torch.equal(baseline.anchor.detach(), baseline_before)
    assert trainer.records[-1].scientific_status == "development_only"

    metadata = Phase1CheckpointMetadata(
        step=trainer.step,
        config_identity="synthetic_phase1_config",
        provider_identity="synthetic",
        geometry_schema="geometry_center_covariance.v1",
        representation_schema="probabilistic_gaussian_representation.v1",
        renderer_backend=ReferenceRenderer.backend_id,
        seed=7,
    )
    path = tmp_path / "phase1.pt"
    trainer.save_checkpoint(path, metadata)

    resumed_provider = TrainableSigma()
    resumed_baseline = FrozenBaseline()
    resumed_trainer = Phase1Trainer(
        _pipeline(resumed_baseline, resumed_provider),
        torch.optim.Adam(resumed_provider.parameters(), lr=0.01),
        trainable_modules={"uncertainty": resumed_provider},
        frozen_modules={"baseline": resumed_baseline},
    )
    loaded = resumed_trainer.load_checkpoint(path)
    assert loaded == metadata
    assert resumed_trainer.step == trainer.step
    assert torch.equal(resumed_provider.raw, provider.raw)
    resumed = resumed_trainer.train_step(batch)
    assert resumed.step == 2
