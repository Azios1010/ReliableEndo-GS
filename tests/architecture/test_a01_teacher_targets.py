"""Focused CPU contracts for A01 teacher targets and cache provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.teachers import (
    A01_FINAL_DATASET_IDS,
    AlignedTeacherTarget,
    CandidateAvailability,
    RectifiedAlignment,
    TargetCache,
    TargetCacheError,
    TargetCacheKey,
    TargetSchema,
    TeacherContractError,
    TeacherPrediction,
    TeacherUnavailableError,
    align_teacher_prediction,
    convert_teacher_output_to_a00,
    default_teacher_candidates,
    load_a01_roles,
    make_foundation_stereo_candidate,
)
from reliable_endo_gs.teachers.targets import _resize_map


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
def test_resize_map_preserves_tensor_values_and_gradients(mode: str) -> None:
    value = torch.arange(6.0).reshape(1, 1, 2, 3).requires_grad_()
    actual = _resize_map(value, (4, 6), mode=mode)
    options = {} if mode == "nearest" else {"align_corners": False}
    expected = torch.nn.functional.interpolate(value, size=(4, 6), mode=mode, **options)
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert value.grad is not None and bool(torch.isfinite(value.grad).all())


def test_resize_map_rejects_a_non_tensor_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "reliable_endo_gs.teachers.targets.functional.interpolate", lambda *args, **kwargs: object()
    )
    with pytest.raises(TeacherContractError, match="interpolation must return a Tensor"):
        _resize_map(torch.ones(1, 1, 2, 3), (4, 6), mode="nearest")


def _target() -> AlignedTeacherTarget:
    disparity = torch.full((1, 1, 2, 3), 4.0)
    validity = torch.ones_like(disparity, dtype=torch.bool)
    support = torch.ones_like(disparity, dtype=torch.bool)
    return AlignedTeacherTarget(
        disparity,
        validity,
        support,
        metadata={
            "alignment_schema": "a00_rectified_input_grid_v1",
            "calibration": {"camera_transform_id": "calib-v1"},
        },
    )


def test_fake_teacher_alignment_records_grid_and_confidence_semantics() -> None:
    prediction = TeacherPrediction(
        disparity=torch.full((1, 1, 2, 6), 4.0),
        predicted_confidence=torch.full((1, 1, 2, 6), 0.75),
        metadata={
            "candidate_output": "fake",
            "disparity_units": "teacher_grid_pixels",
            "disparity_sign": "u_left_minus_right",
        },
    )
    alignment = RectifiedAlignment(
        input_hw=(4, 6),
        teacher_hw=(2, 6),
        crop_hw=(2, 3),
        left_crop_xy=(1, 1),
        right_crop_xy=(0, 1),
        image_transform_id="resize-crop-v1",
        calibration_transform_id="calibration-v1",
    )
    target = align_teacher_prediction(prediction, alignment)

    assert target.disparity.shape == (1, 1, 4, 6)
    assert target.validity[:, :, 1:3, 1:4].all()
    assert target.support[:, :, 1:3, 1:4].all()
    assert not target.support[:, :, :1].any()
    # 4 teacher pixels / sx=2 plus the left-right crop-origin correction (+1).
    torch.testing.assert_close(target.disparity[:, :, 1:3, 1:4], torch.full((1, 1, 2, 3), 3.0))
    torch.testing.assert_close(
        target.predicted_confidence[:, :, 1:3, 1:4], torch.full((1, 1, 2, 3), 0.75)
    )
    assert target.calibrated_risk is None
    assert target.metadata["target_disparity_units"] == "a00_input_grid_pixels"
    assert target.metadata["disparity_sign"] == "u_left_minus_right"
    assert target.metadata["confidence_semantics"] == "predicted_confidence_not_calibrated_risk"


def test_lazy_fake_teacher_is_frozen_eval_and_no_gradient() -> None:
    class FakeTeacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return (left[:, :1] - right[:, :1]) * self.weight + 2.0

    model = FakeTeacher()
    calls = {"load": 0}

    def load() -> FakeTeacher:
        calls["load"] += 1
        return model

    adapter = make_foundation_stereo_candidate(
        loader=load,
        infer_fn=lambda loaded, left, right: loaded(left, right),
        checkpoint_id="fake-checkpoint",
        checkpoint_sha256="a" * 64,
        api_version="fake-api-v1",
        license_name="fake-license",
        preprocessing={"image_range": "0..1"},
        output={
            "disparity_units": "teacher_grid_pixels",
            "disparity_sign": "u_left_minus_right",
        },
    )
    left = torch.ones((1, 3, 2, 3), requires_grad=True)
    right = torch.zeros_like(left)
    result = adapter.infer(left, right)

    assert calls["load"] == 1
    assert adapter.model_loaded
    assert not model.training
    assert not model.weight.requires_grad
    assert not result.disparity.requires_grad
    assert adapter.identity.candidate == "FoundationStereo"
    assert adapter.identity.identity_hash
    assert adapter.availability().status == "available"

    model.train()
    model.weight.requires_grad_(True)
    second = adapter.infer(left, right)
    assert not model.training
    assert not model.weight.requires_grad
    assert not second.disparity.requires_grad


def test_live_teacher_rejects_unverified_identity() -> None:
    model = torch.nn.Identity()
    adapter = make_foundation_stereo_candidate(
        loader=lambda: model,
        infer_fn=lambda loaded, left, right: left[:, :1],
    )
    with pytest.raises(TeacherContractError, match="unverified identity"):
        adapter.infer(torch.zeros((1, 3, 2, 2)), torch.zeros((1, 3, 2, 2)))


def test_unavailable_candidates_are_reported_without_external_imports() -> None:
    foundation, defom = default_teacher_candidates()
    foundation_report = foundation.availability()
    defom_report = defom.availability()
    assert foundation_report == CandidateAvailability(
        "FoundationStereo", False, "no lazy loader was injected", True
    )
    assert defom_report.available is False
    with pytest.raises(TeacherUnavailableError, match="no lazy loader"):
        foundation.infer(torch.zeros((1, 3, 2, 2)), torch.zeros((1, 3, 2, 2)))


def test_teacher_output_conversion_is_explicit_about_sign_scale_and_validity() -> None:
    converted = convert_teacher_output_to_a00(
        torch.full((1, 2, 2), -4.0),
        disparity_sign="u_right_minus_left",
        disparity_scale=2.0,
    )
    assert converted.disparity.shape == (1, 1, 2, 2)
    assert torch.equal(converted.validity, torch.ones_like(converted.validity))
    assert torch.all(converted.disparity == 8.0)
    assert converted.metadata["disparity_sign"] == "u_left_minus_right"


def test_cache_resume_checksum_manifest_and_key_invalidation(tmp_path: Path) -> None:
    cache_root = tmp_path / "external-target-cache"
    cache = TargetCache(cache_root)
    key = TargetCacheKey(
        sample_id="dataset_1/keyframe_1/frame_000",
        split_identity="a01-scared-final-train-1-7-test-8-9-v1",
        data_role="final_train",
        image_transform_hash="image-v1",
        calibration_transform_hash="calib-v1",
        teacher_identity_hash="teacher-v1",
    )
    calls = {"count": 0}

    def produce() -> AlignedTeacherTarget:
        calls["count"] += 1
        return _target()

    first = cache.get_or_compute(key, produce)
    second = TargetCache(cache_root).get_or_compute(key, produce)
    changed = TargetCacheKey(
        sample_id=key.sample_id,
        split_identity=key.split_identity,
        data_role=key.data_role,
        image_transform_hash="image-v2",
        calibration_transform_hash=key.calibration_transform_hash,
        teacher_identity_hash=key.teacher_identity_hash,
    )
    third = TargetCache(cache_root).get_or_compute(changed, produce)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert third.cache_hit is False
    assert calls["count"] == 2
    manifest = (cache_root / "manifest.json").read_text(encoding="utf-8")
    assert "manifest_sha256" in manifest
    assert len(TargetCache(cache_root).entries) == 2
    assert TargetCache(cache_root).get(key) is not None


def test_cache_manifest_and_item_integrity_fail_closed(tmp_path: Path) -> None:
    cache_root = tmp_path / "integrity-cache"
    cache = TargetCache(cache_root)
    key = TargetCacheKey(
        sample_id="dataset_1/keyframe_1/frame_000",
        split_identity="a01-scared-final-train-1-7-test-8-9-v1",
        data_role="final_train",
        image_transform_hash="image-v1",
        calibration_transform_hash="calib-v1",
        teacher_identity_hash="teacher-v1",
    )
    cache.put(key, _target())

    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("manifest_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TargetCacheError, match="checksum is required"):
        TargetCache(cache_root)

    item_cache_root = tmp_path / "item-integrity-cache"
    item_cache = TargetCache(item_cache_root)
    item_cache.put(key, _target())
    item_path = item_cache_root / item_cache.entries[0]["relative_path"]
    item_path.write_bytes(item_path.read_bytes() + b"tamper")
    assert TargetCache(item_cache_root).get(key) is None

    valid_cache_root = tmp_path / "manifest-shape-cache"
    valid_cache = TargetCache(valid_cache_root)
    valid_cache.put(key, _target())
    valid_manifest_path = valid_cache_root / "manifest.json"
    valid_manifest = json.loads(valid_manifest_path.read_text(encoding="utf-8"))
    valid_manifest["entries"][0]["shape"] = []
    without_hash = {
        "manifest_schema_version": valid_manifest["manifest_schema_version"],
        "target_schema": valid_manifest["target_schema"],
        "role_contract_hash": valid_manifest["role_contract_hash"],
        "entries": valid_manifest["entries"],
    }
    valid_manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(without_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    valid_manifest_path.write_text(json.dumps(valid_manifest), encoding="utf-8")
    with pytest.raises(TargetCacheError, match="shape is invalid"):
        TargetCache(valid_cache_root)


def test_role_config_uses_dataset_1_to_7_for_training_and_protects_8_to_9(
    tmp_path: Path,
) -> None:
    roles = load_a01_roles(Path("configs/teachers/a01_scared.yaml"))
    assert roles.dataset_ids == tuple(f"dataset_{index}" for index in range(1, 10))
    assert roles.final_dataset_ids == A01_FINAL_DATASET_IDS == ("dataset_8", "dataset_9")
    assert roles.datasets_for_role("final_train") == tuple(
        f"dataset_{index}" for index in range(1, 8)
    )
    assert roles.datasets_for_role("final_test") == ("dataset_8", "dataset_9")
    assert roles.final_roles == ("final_test",)
    assert roles.cache_roles == ("final_train",)
    assert (
        roles.validate_sample_id("dataset_1/keyframe_1/frame_000", role="final_train", cache=True)
        == "dataset_1/keyframe_1/frame_000"
    )
    dataset_six_key = TargetCacheKey(
        sample_id="dataset_6/keyframe_1/frame_000",
        split_identity="a01-scared-final-train-1-7-test-8-9-v1",
        data_role="final_train",
        image_transform_hash="image-v1",
        calibration_transform_hash="calib-v1",
        teacher_identity_hash="teacher-v1",
    )
    TargetCache(tmp_path / "dataset-six-cache", roles=roles).put(dataset_six_key, _target())

    assert (
        roles.validate_sample_id("dataset_8/keyframe_1/frame_000", role="final_test", cache=False)
        == "dataset_8/keyframe_1/frame_000"
    )
    final_test_key = TargetCacheKey(
        sample_id="dataset_8/keyframe_1/frame_000",
        split_identity="a01-scared-final-train-1-7-test-8-9-v1",
        data_role="final_test",
        image_transform_hash="image-v1",
        calibration_transform_hash="calib-v1",
        teacher_identity_hash="teacher-v1",
    )
    with pytest.raises(TargetCacheError, match="not approved for A01 target caching"):
        TargetCache(tmp_path / "final-test-cache", roles=roles).put(final_test_key, _target())

    for sample_id in (
        "dataset_10/keyframe_1/frame_000",
        "SCARED-C/sequence/frame_000",
        "arbitrary/keyframe/frame_000",
    ):
        with pytest.raises(TargetCacheError):
            TargetCache(tmp_path / "invalid-cache", roles=roles).put(
                TargetCacheKey(
                    sample_id=sample_id,
                    split_identity="a01-scared-final-train-1-7-test-8-9-v1",
                    data_role="final_test",
                    image_transform_hash="image-v1",
                    calibration_transform_hash="calib-v1",
                    teacher_identity_hash="teacher-v1",
                ),
                _target(),
            )


def test_cache_rejects_unvalidated_roles_and_identity_mismatches(tmp_path: Path) -> None:
    with pytest.raises(TargetCacheError, match="validated SCARED"):
        TargetCache(tmp_path / "unvalidated", roles=None)  # type: ignore[arg-type]

    key = TargetCacheKey(
        sample_id="dataset_1/keyframe_1/frame_000",
        split_identity="a01-scared-final-train-1-7-test-8-9-v1",
        data_role="final_train",
        image_transform_hash="image-v1",
        calibration_transform_hash="calib-v1",
        teacher_identity_hash="teacher-v1",
    )
    custom_schema = TargetSchema(version=2)
    with pytest.raises(TargetCacheError, match="target schema"):
        TargetCache(tmp_path / "schema-mismatch", target_schema=custom_schema).put(key, _target())

    bad_role_key = TargetCacheKey(
        sample_id=key.sample_id,
        split_identity=key.split_identity,
        data_role=key.data_role,
        image_transform_hash=key.image_transform_hash,
        calibration_transform_hash=key.calibration_transform_hash,
        teacher_identity_hash=key.teacher_identity_hash,
        role_contract_hash="0" * 64,
    )
    with pytest.raises(TargetCacheError, match="role contract"):
        TargetCache(tmp_path / "role-mismatch").put(bad_role_key, _target())


def test_teacher_package_does_not_reference_historical_model_boundary() -> None:
    package_root = Path("src/reliable_endo_gs/teachers")
    source = "\n".join(path.read_text(encoding="utf-8") for path in package_root.glob("*.py"))
    assert "reliable_endo_gs.baseline" not in source
    assert "third_party" not in source
    assert "scared_c" not in source.casefold()
