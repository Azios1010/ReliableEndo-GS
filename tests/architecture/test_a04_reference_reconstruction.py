"""Focused CPU contracts for the A04 reference composition."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera
from reliable_endo_gs.contracts.rendering import RenderOutput
from reliable_endo_gs.gaussians import (
    AttributeResiduals,
    GeometryPriorPolicy,
    ReferenceAttributeHead,
    ResidualBounds,
    build_geometry_prior,
    compose_gaussian_field,
    context_from_geometry_prior,
    quaternion_to_rotation_matrix,
    stable_axis_angle_exp,
)
from reliable_endo_gs.reconstruction import (
    ReferenceConfig,
    ReferenceConfigError,
    ReferenceReconstructionModel,
    ReferenceRuntimeHooks,
    load_reference_checkpoint,
    load_reference_config,
    reference_checkpoint_path,
    save_reference_checkpoint,
    write_reference_config,
)
from reliable_endo_gs.teachers import AlignedTeacherTarget, TeacherIdentity, TeacherPrediction


def _camera(batch_size: int = 1, dtype: torch.dtype = torch.float64) -> RectifiedStereoCamera:
    left = torch.tensor(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]], dtype=dtype
    ).unsqueeze(0).repeat(batch_size, 1, 1)
    right = left.clone()
    right[:, 0, 2] -= 3.0
    transform = torch.eye(4, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    transform[:, 0, 3] = -0.1
    return RectifiedStereoCamera(left, right, transform, True)


def _batch(batch_size: int = 1) -> DeploymentStereoBatch:
    height, width = 5, 5
    left = torch.full((batch_size, 3, height, width), 0.25, dtype=torch.float64)
    right = torch.full_like(left, 0.2)
    support = torch.ones((batch_size, 1, height, width), dtype=torch.bool)
    return DeploymentStereoBatch(left, right, _camera(batch_size), support)


def _target(batch: DeploymentStereoBatch, *, alpha_support: bool = True) -> AlignedTeacherTarget:
    disparity = torch.full(
        (batch.left.shape[0], 1, batch.left.shape[2], batch.left.shape[3]),
        8.0,
        dtype=torch.float64,
    )
    validity = torch.ones_like(disparity, dtype=torch.bool)
    support = torch.full_like(disparity, alpha_support, dtype=torch.bool)
    return AlignedTeacherTarget(
        disparity,
        validity,
        support,
        metadata={
            "alignment_schema": "a00_rectified_input_grid_v1",
            "target_disparity_units": "a00_input_grid_pixels",
            "disparity_sign": "u_left_minus_right",
            "calibration": {"calibration_id": "analytic-v1"},
            "target_cache_key": "cache-key-v1",
        },
    )


class _FakeTeacher(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.identity = TeacherIdentity(
            candidate_id="FoundationStereo",
            variant="analytic-test",
            api_version="test-api-v1",
            checkpoint_id="test-checkpoint",
            checkpoint_sha256="a" * 64,
            license="test-license",
            preprocessing={"status": "verified"},
            output={
                "status": "verified",
                "disparity_units": "teacher_grid_pixels",
                "disparity_sign": "u_left_minus_right",
            },
            source="focused-test",
        )

    def infer(self, left: torch.Tensor, right: torch.Tensor) -> TeacherPrediction:
        disparity = torch.full(
            (left.shape[0], 1, left.shape[2], left.shape[3]),
            8.0,
            dtype=left.dtype,
            device=left.device,
        )
        return TeacherPrediction(
            disparity * self.weight.detach(),
            validity=torch.ones_like(disparity, dtype=torch.bool),
            support=torch.ones_like(disparity, dtype=torch.bool),
            metadata={
                "disparity_units": "teacher_grid_pixels",
                "disparity_sign": "u_left_minus_right",
            },
        )


class _FakeRenderer:
    identity = "analytic-field-renderer-v1"

    def __init__(self) -> None:
        self.requests: list[object] = []

    def render(self, request: object) -> RenderOutput:
        from reliable_endo_gs.rendering.field import FieldRenderRequest

        assert isinstance(request, FieldRenderRequest)
        self.requests.append(request)
        height, width = request.image_hw
        colors = request.field.colors
        image = colors.mean(dim=1).view(colors.shape[0], 3, 1, 1).expand(
            colors.shape[0], 3, height, width
        )
        depth = request.field.means3d[..., 2].mean(dim=1).view(colors.shape[0], 1, 1, 1)
        depth = depth.expand(colors.shape[0], 1, height, width)
        return RenderOutput(image=image, depth=depth)


def test_zero_residual_identity_and_bound_composition() -> None:
    batch = _batch()
    camera = batch.camera
    disparity = torch.full((1, 1, 5, 5), 8.0, dtype=torch.float64)
    policy = GeometryPriorPolicy(alpha_valid=0.5)
    prior = build_geometry_prior(disparity, camera, batch.support, batch.left, policy=policy)
    head = ReferenceAttributeHead(hidden_dims=(8, 8))
    context = context_from_geometry_prior(prior)
    residuals = head(context, prior.valid_mask)
    assert torch.equal(residuals.delta_rotation, torch.zeros_like(residuals.delta_rotation))
    field = compose_gaussian_field(prior, residuals, bounds=head.bounds)
    rotation = quaternion_to_rotation_matrix(field.rotations)
    torch.testing.assert_close(rotation[prior.valid_mask], prior.R0[prior.valid_mask], atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(field.scales[prior.valid_mask], prior.S0[prior.valid_mask])
    torch.testing.assert_close(field.opacities[prior.valid_mask], prior.alpha0[prior.valid_mask])

    bounds = ResidualBounds(max_rotation_rad=0.1, max_log_scale=0.2, max_alpha_logit=0.3)
    oversized = AttributeResiduals(
        torch.full_like(prior.mu, 100.0),
        torch.full_like(prior.S0, -100.0),
        torch.full_like(prior.alpha0, 100.0),
    )
    bounded = compose_gaussian_field(prior, oversized, bounds=bounds)
    assert torch.all(bounded.scales[prior.valid_mask] >= bounds.min_scale_m)
    assert torch.all(bounded.scales[prior.valid_mask] <= bounds.max_scale_m)
    assert torch.all((bounded.opacities >= 0) & (bounded.opacities <= 1))
    assert torch.isfinite(bounded.means3d).all()


def test_composition_has_finite_gradients_and_separate_invalid_mask() -> None:
    batch = _batch(batch_size=2)
    support = batch.support.clone()
    support[1, :, 0, :] = False
    support[1, :, :, 0] = False
    disparity = torch.full((2, 1, 5, 5), 8.0, dtype=torch.float64, requires_grad=True)
    prior = build_geometry_prior(disparity, batch.camera, support, batch.left)
    head = ReferenceAttributeHead(hidden_dims=(8, 8))
    residuals = head(context_from_geometry_prior(prior), prior.valid_mask)
    field = compose_gaussian_field(prior, residuals, bounds=head.bounds)
    loss = field.means3d.square().sum() + field.scales.sum() + field.opacities.sum()
    loss.backward()
    assert disparity.grad is not None and torch.isfinite(disparity.grad).all()
    assert any(parameter.grad is not None for parameter in head.parameters())
    assert bool((~field.valid_mask[1]).any())
    assert torch.all(field.opacities[~field.valid_mask] == 0)
    assert torch.isfinite(field.means3d).all()
    assert torch.isfinite(field.cov_surface).all()


def test_zero_axis_angle_has_finite_gradients() -> None:
    delta = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    matrix = stable_axis_angle_exp(delta)
    matrix.sum().backward()
    assert torch.isfinite(matrix).all()
    assert delta.grad is not None and torch.isfinite(delta.grad).all()


def test_reference_forward_freezes_teacher_records_provenance_and_profiles() -> None:
    batch = _batch()
    teacher = _FakeTeacher()
    renderer = _FakeRenderer()
    config = replace(
        ReferenceConfig(),
        prior_policy=GeometryPriorPolicy(alpha_valid=0.5),
        head_hidden_dims=(8, 8),
        assessment_views=("left", "right"),
    )
    model = ReferenceReconstructionModel(teacher=teacher, renderer=renderer, config=config)
    result = model(batch, sample_ids=("dataset_1/keyframe/frame_000",))
    assert not teacher.training
    assert not teacher.weight.requires_grad
    assert result.render_right is not None
    assert len(renderer.requests) == 2
    assert result.provenance_record.teacher_identity["candidate_id"] == "FoundationStereo"
    assert result.provenance_record.target_source == "live_teacher_inference"
    assert result.provenance_record.calibration["calibration_identity"] == (
        "explicit-calibration-at-forward"
    )
    assert result.provenance_record.data_role == "teacher_fit"
    assert result.provenance_record.excluded_dataset_names == ("SCARED-C",)
    assert result.metrics["quality_target_status"] == (
        "not_assessed_without_declared_observation_metric"
    )
    profile, profiled = model.profile(
        batch,
        target=result.target,
        hooks=ReferenceRuntimeHooks(warmup=0, repetitions=2, peak_memory=lambda: 123),
    )
    assert profiled.metrics["valid_gaussian_count"] == result.metrics["valid_gaussian_count"]
    assert len(profile.latencies_ms) == 2
    assert profile.peak_memory_bytes == 123


def test_checkpoint_and_config_roundtrip_keep_teacher_renderer_external(tmp_path: Path) -> None:
    config = replace(ReferenceConfig(), head_hidden_dims=(8, 8))
    config_path = tmp_path / "reference.yaml"
    write_reference_config(config_path, config)
    loaded_config = load_reference_config(config_path)
    assert loaded_config.to_dict() == config.to_dict()
    assert loaded_config.config_hash == config.config_hash

    model = ReferenceReconstructionModel(
        teacher=_FakeTeacher(), renderer=_FakeRenderer(), config=config
    )
    result = model(_batch(), target=_target(_batch()), sample_ids=("dataset_1/keyframe/frame_000",))
    checkpoint_path = reference_checkpoint_path(config, root=tmp_path / "checkpoints")
    save_reference_checkpoint(checkpoint_path, model, provenance=result.provenance_record)
    restored = ReferenceReconstructionModel(
        teacher=_FakeTeacher(), renderer=_FakeRenderer(), config=config
    )
    record = load_reference_checkpoint(checkpoint_path, restored)
    assert record.provenance_hash == result.provenance_record.provenance_hash
    for first, second in zip(model.head.parameters(), restored.head.parameters(), strict=True):
        torch.testing.assert_close(first, second)


def test_reference_rejects_unverified_target_units_and_collapsed_views() -> None:
    batch = _batch()
    model = ReferenceReconstructionModel(teacher=None, renderer=_FakeRenderer())
    bad_metadata = {
        "alignment_schema": "a00_rectified_input_grid_v1",
        "target_disparity_units": "normalized_depth",
        "disparity_sign": "u_left_minus_right",
        "calibration": {"calibration_id": "analytic-v1"},
    }
    bad_target = replace(_target(batch), metadata=bad_metadata)
    with pytest.raises(ValueError, match="a00_input_grid_pixels"):
        model(batch, target=bad_target)

    with pytest.raises(ReferenceConfigError, match="held out"):
        replace(ReferenceConfig(), assessment_views=("left",))


def test_reference_import_isolated_from_historical_and_optional_packages() -> None:
    root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc, sys
blocked = ("reliable_endo_gs.baseline", "reliable_endo_gs.probabilistic_gs", "lib", "third_party")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("forbidden import: " + fullname)
        return None
sys.meta_path.insert(0, Block())
from reliable_endo_gs.reconstruction import ReferenceConfig, ReferenceReconstructionModel
from reliable_endo_gs.gaussians import ReferenceAttributeHead
assert ReferenceConfig().quality_first
assert ReferenceAttributeHead
assert ReferenceReconstructionModel
"""
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True, capture_output=True, text=True)


def test_reference_rejects_scared_c_and_unverified_sample_roles() -> None:
    model = ReferenceReconstructionModel(teacher=None, renderer=_FakeRenderer())
    batch = _batch()
    target = _target(batch)
    with pytest.raises(ValueError, match="dataset_1 through dataset_7|unknown dataset"):
        model(batch, target=target, sample_ids=("SCARED-C/frame_000",))
