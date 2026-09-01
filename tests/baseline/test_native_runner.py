"""Synthetic structural tests for the bounded native stage-two runner.

These tests substitute upstream-shaped modules and tensors only. They do not
exercise CUDA, the rasterizer, corr_sampler, a medical dataset, or an official
checkpoint and therefore make no native numerical-parity claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from reliable_endo_gs.baseline.checkpoint import CheckpointError, CheckpointLoadResult
from reliable_endo_gs.baseline.config import BaselineConfig
from reliable_endo_gs.baseline.native import (
    NativeBaselineRunner,
    NativeInferenceInput,
    NativeInputError,
    NativeRunnerError,
)
from reliable_endo_gs.baseline.upstream import PINNED_COMMIT, UpstreamGitState


def _native_input() -> NativeInferenceInput:
    batch_size, height, width = 1, 2, 3
    return NativeInferenceInput(
        sample_names=("synthetic-0",),
        left_image=torch.zeros(batch_size, 3, height, width),
        right_image=torch.ones(batch_size, 3, height, width) * 0.25,
        left_mask=torch.ones(batch_size, 1, height, width),
        left_disparity_constant=torch.tensor([1000.0]),
        left_intrinsics=torch.eye(3).unsqueeze(0),
        right_intrinsics=torch.eye(3).unsqueeze(0),
        left_extrinsics=torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1).unsqueeze(0),
        left_fov_x=torch.tensor([0.5]),
        left_fov_y=torch.tensor([0.5]),
        left_width=torch.tensor([width]),
        left_height=torch.tensor([height]),
        left_world_view_transform=torch.eye(4).unsqueeze(0),
        left_full_projection_transform=torch.eye(4).unsqueeze(0),
        left_camera_center=torch.zeros(batch_size, 3),
    )


def _bound_config(tmp_path: Path, *, device: str = "cuda") -> BaselineConfig:
    checkpoint_path = tmp_path / "synthetic-stage2.pth"
    checkpoint_path.write_bytes(b"synthetic checkpoint identity")
    return BaselineConfig(
        upstream_root=Path("third_party/endo_e2e_gs"),
        upstream_commit=PINNED_COMMIT,
        upstream_config=Path("config/stage2.yaml"),
        entry_point="render.py",
        device=device,
        enforce_clean=True,
        checkpoint_id="synthetic-only",
        checkpoint_path=checkpoint_path,
        checkpoint_sha256="a" * 64,
    )


class _FakeConfigStereo:
    corr_implementation = "reg_cuda"
    loaded_path: str | None = None

    def load(self, config_file: str) -> None:
        type(self).loaded_path = config_file

    def get_cfg(self) -> object:
        return SimpleNamespace(
            raft=SimpleNamespace(corr_implementation=type(self).corr_implementation),
            dataset=SimpleNamespace(bg_color=[0, 0, 0]),
        )


class _FakeStereoEndoModel(torch.nn.Module):
    last_instance: _FakeStereoEndoModel | None = None

    def __init__(self, config: object, *, with_gs_render: bool) -> None:
        super().__init__()
        self.config = config
        self.with_gs_render = with_gs_render
        self.is_train_argument: bool | None = None
        type(self).last_instance = self

    def forward(
        self, data: dict[str, object], is_train: bool = True
    ) -> tuple[dict[str, object], None, None]:
        self.is_train_argument = is_train
        left_value = data["lmain"]
        assert isinstance(left_value, dict)
        image = left_value["img"]
        assert isinstance(image, torch.Tensor)
        batch_size, _, height, width = image.shape
        count = height * width
        left_value["disp_const"] = torch.ones(batch_size, device=image.device).view(
            batch_size, 1, 1, 1
        )
        left_value["flow_pred"] = torch.ones(batch_size, 1, height, width, device=image.device)
        left_value["depth_pred"] = torch.ones(batch_size, 1, height, width, device=image.device)
        left_value["xyz"] = torch.zeros(batch_size, count, 3, device=image.device)
        left_value["pts_valid"] = torch.ones(
            batch_size, count, dtype=torch.bool, device=image.device
        )
        left_value["rot_maps"] = torch.ones(batch_size, 4, height, width, device=image.device)
        left_value["scale_maps"] = torch.ones(batch_size, 3, height, width, device=image.device)
        left_value["opacity_maps"] = torch.ones(batch_size, 1, height, width, device=image.device)
        return data, None, None


def _fake_pts2render(
    data: dict[str, object], *, bg_color: object
) -> tuple[dict[str, object], list[object]]:
    del bg_color
    left_value = data["lmain"]
    assert isinstance(left_value, dict)
    image = left_value["img"]
    assert isinstance(image, torch.Tensor)
    left_value["img_pred"] = image * 0.5 + 0.5
    return data, ["synthetic-rgb", "synthetic-xyz", None, "rot", "scale", "opacity"]


def _install_synthetic_upstream(
    monkeypatch: pytest.MonkeyPatch,
    *,
    corr_implementation: str = "reg_cuda",
) -> dict[str, object]:
    from reliable_endo_gs.baseline import native

    _FakeConfigStereo.corr_implementation = corr_implementation
    _FakeConfigStereo.loaded_path = None
    _FakeStereoEndoModel.last_instance = None
    observed: dict[str, object] = {"imports": []}

    modules = {
        "config.stereo_config": SimpleNamespace(ConfigStereo=_FakeConfigStereo),
        "lib.network": SimpleNamespace(StereoEndoModel=_FakeStereoEndoModel),
        "lib.GaussianRender": SimpleNamespace(pts2render=_fake_pts2render),
    }

    def import_module(
        module_name: str,
        upstream_root: Path,
        **kwargs: object,
    ) -> object:
        del upstream_root
        observed_imports = observed["imports"]
        assert isinstance(observed_imports, list)
        observed_imports.append((module_name, kwargs))
        return modules[module_name]

    def strict_checkpoint_loader(
        model: torch.nn.Module,
        path: Path,
        *,
        expected_sha256: str,
        map_location: str | torch.device,
        strict: bool,
    ) -> CheckpointLoadResult:
        observed["checkpoint"] = {
            "model": model,
            "path": path,
            "expected_sha256": expected_sha256,
            "map_location": map_location,
            "strict": strict,
        }
        return CheckpointLoadResult(
            path=path,
            sha256=expected_sha256,
            missing_keys=(),
            unexpected_keys=(),
        )

    monkeypatch.setattr(native, "import_upstream_module", import_module)
    monkeypatch.setattr(
        native,
        "require_pinned_upstream",
        lambda upstream_root, enforce_clean: UpstreamGitState(PINNED_COMMIT, False),
    )
    monkeypatch.setattr(native, "load_upstream_checkpoint", strict_checkpoint_loader)
    # The production runner rejects CPU. Synthetic tests replace only device
    # resolution so the fake, non-rasterizing module can run in CPU CI.
    monkeypatch.setattr(
        native, "_resolve_cuda_device", lambda configured_device: torch.device("cpu")
    )
    return observed


def test_native_input_rejects_non_upstream_image_shape() -> None:
    values = _native_input()
    with pytest.raises(NativeInputError, match="left_image must have shape"):
        NativeInferenceInput(
            sample_names=values.sample_names,
            left_image=torch.zeros(1, 1, 2, 3),
            right_image=values.right_image,
            left_mask=values.left_mask,
            left_disparity_constant=values.left_disparity_constant,
            left_intrinsics=values.left_intrinsics,
            right_intrinsics=values.right_intrinsics,
            left_extrinsics=values.left_extrinsics,
            left_fov_x=values.left_fov_x,
            left_fov_y=values.left_fov_y,
            left_width=values.left_width,
            left_height=values.left_height,
            left_world_view_transform=values.left_world_view_transform,
            left_full_projection_transform=values.left_full_projection_transform,
            left_camera_center=values.left_camera_center,
        )


def test_native_runner_requires_a_bound_checkpoint_identity(tmp_path: Path) -> None:
    config = BaselineConfig(
        upstream_root=Path("third_party/endo_e2e_gs"),
        upstream_commit=PINNED_COMMIT,
        upstream_config=Path("config/stage2.yaml"),
        entry_point="render.py",
        device="cuda",
        enforce_clean=True,
    )

    with pytest.raises(NativeRunnerError, match="explicit checkpoint_id"):
        NativeBaselineRunner(config).run(_native_input())


def test_native_runner_rejects_cpu_before_upstream_import(tmp_path: Path) -> None:
    with pytest.raises(NativeRunnerError, match="requires a CUDA device"):
        NativeBaselineRunner(_bound_config(tmp_path, device="cpu")).run(_native_input())


def test_native_runner_constructs_stage_two_strictly_and_captures_pre_adapter_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed = _install_synthetic_upstream(monkeypatch)
    native_input = _native_input()
    original_disparity_constant = native_input.left_disparity_constant.clone()

    result = NativeBaselineRunner(_bound_config(tmp_path)).run(native_input)

    model = _FakeStereoEndoModel.last_instance
    assert model is not None
    assert model.with_gs_render is True
    assert model.is_train_argument is False
    assert model.training is False
    assert _FakeConfigStereo.loaded_path is not None
    assert _FakeConfigStereo.loaded_path.endswith("config\\stage2.yaml")
    assert observed["imports"] == [
        ("config.stereo_config", {"enforce_clean": True}),
        ("lib.network", {"require_corr_sampler": True, "enforce_clean": True}),
        ("lib.GaussianRender", {"require_rasterizer": True, "enforce_clean": True}),
    ]
    checkpoint = observed["checkpoint"]
    assert isinstance(checkpoint, dict)
    assert checkpoint["strict"] is True
    assert checkpoint["map_location"] == torch.device("cpu")

    native_left = result.native_output["lmain"]
    assert isinstance(native_left, Mapping)
    native_disparity = native_left["flow_pred"]
    assert isinstance(native_disparity, torch.Tensor)
    assert result.parity.stereo.disparity is native_disparity
    assert result.parity.render_left.image is native_left["img_pred"]
    assert tuple(result.renderer_payload) == (
        "synthetic-rgb",
        "synthetic-xyz",
        None,
        "rot",
        "scale",
        "opacity",
    )
    assert native_input.left_disparity_constant.shape == (1,)
    assert torch.equal(native_input.left_disparity_constant, original_disparity_constant)
    assert result.upstream_git_state.commit == PINNED_COMMIT
    assert result.checkpoint.sha256 == "a" * 64


def test_native_runner_refuses_reg_fallback_before_model_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_synthetic_upstream(monkeypatch, corr_implementation="reg")

    with pytest.raises(NativeRunnerError, match="pure-PyTorch 'reg' backend"):
        NativeBaselineRunner(_bound_config(tmp_path)).run(_native_input())

    assert _FakeStereoEndoModel.last_instance is None


def test_native_runner_preserves_checkpoint_load_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_synthetic_upstream(monkeypatch)
    from reliable_endo_gs.baseline import native

    def reject_checkpoint(
        model: torch.nn.Module,
        path: Path,
        *,
        expected_sha256: str,
        map_location: str | torch.device,
        strict: bool,
    ) -> CheckpointLoadResult:
        del model, path, expected_sha256, map_location, strict
        raise CheckpointError("synthetic checkpoint rejection")

    monkeypatch.setattr(native, "load_upstream_checkpoint", reject_checkpoint)

    with pytest.raises(CheckpointError, match="synthetic checkpoint rejection"):
        NativeBaselineRunner(_bound_config(tmp_path)).run(_native_input())
