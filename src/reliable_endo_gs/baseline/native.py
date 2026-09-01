"""Bounded execution of the pinned Endo-E2E-GS stage-two inference path.

The upstream project is not imported until :class:`NativeBaselineRunner` is
asked to run.  This module owns the exact stage-two sequence observed at the
pinned revision: construct ``ConfigStereo`` from ``config/stage2.yaml``,
construct ``StereoEndoModel(..., with_gs_render=True)``, load
``checkpoint['network']`` strictly, call ``model(data, is_train=False)``, and
then call ``pts2render``.  It deliberately does not construct inputs from
``StereoBatch`` because the authoritative dataset/calibration bridge belongs
to Plan 02.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import torch

from reliable_endo_gs.baseline.adapter import EndoE2EGSAdapter
from reliable_endo_gs.baseline.checkpoint import CheckpointLoadResult, load_upstream_checkpoint
from reliable_endo_gs.baseline.config import BaselineConfig
from reliable_endo_gs.baseline.upstream import (
    PINNED_COMMIT,
    UpstreamGitState,
    import_upstream_module,
    require_pinned_upstream,
)
from reliable_endo_gs.contracts import GaussianField, RenderOutput, StereoPrediction


class NativeInputError(ValueError):
    """Raised when a would-be native input does not match the observed upstream schema."""


class NativeRunnerError(RuntimeError):
    """Raised when the fixed native stage-two execution contract cannot be honored."""


def _require_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise NativeInputError(f"{name} must be a torch.Tensor; got {type(value).__name__}")
    return value


def _require_floating_tensor(name: str, value: object) -> torch.Tensor:
    tensor = _require_tensor(name, value)
    if not torch.is_floating_point(tensor):
        raise NativeInputError(f"{name} must be floating-point; got {tensor.dtype}")
    return tensor


def _require_numeric_tensor(name: str, value: object) -> torch.Tensor:
    tensor = _require_tensor(name, value)
    if tensor.dtype == torch.bool or torch.is_complex(tensor):
        raise NativeInputError(f"{name} must be a real numeric tensor; got {tensor.dtype}")
    return tensor


def _require_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise NativeInputError(f"{name} must have shape {expected}; got {tuple(tensor.shape)}")


def _freeze_names(values: Sequence[str], batch_size: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise NativeInputError("sample_names must be a sequence of strings")
    frozen = tuple(values)
    if len(frozen) != batch_size:
        raise NativeInputError(f"sample_names must contain {batch_size} values; got {len(frozen)}")
    if not all(isinstance(value, str) for value in frozen):
        raise NativeInputError("sample_names entries must be strings")
    return frozen


def _clone_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Create input-owned storage before the upstream code mutates its data dictionary."""

    return tensor.detach().clone().to(device)


@dataclass(frozen=True, slots=True)
class NativeInferenceInput:
    """Validated upstream-shaped input for one stage-two native inference call.

    Images are the already-normalized upstream values ``[B, 3, H, W]`` in
    ``[-1, 1]``; this class does not rescale, crop, rectify, or otherwise
    normalize them. ``left_mask`` has shape ``[B, 1, H, W]`` and
    ``left_disparity_constant`` is ``[B]`` because the pinned ``disp2depth``
    immediately views it as ``[B, 1, 1, 1]``. The remaining left-view camera
    tensors are required by the unmodified ``pts2render``/rasterizer path.

    The public field names deliberately describe the native dictionary rather
    than a baseline-neutral camera contract: Plan 02 owns any conversion from
    authoritative dataset samples to these values.
    """

    sample_names: Sequence[str]
    left_image: torch.Tensor
    right_image: torch.Tensor
    left_mask: torch.Tensor
    left_disparity_constant: torch.Tensor
    left_intrinsics: torch.Tensor
    right_intrinsics: torch.Tensor
    left_extrinsics: torch.Tensor
    left_fov_x: torch.Tensor
    left_fov_y: torch.Tensor
    left_width: torch.Tensor
    left_height: torch.Tensor
    left_world_view_transform: torch.Tensor
    left_full_projection_transform: torch.Tensor
    left_camera_center: torch.Tensor

    def __post_init__(self) -> None:
        left_image = _require_floating_tensor("left_image", self.left_image)
        right_image = _require_floating_tensor("right_image", self.right_image)
        if left_image.ndim != 4 or left_image.shape[1] != 3:
            raise NativeInputError(
                f"left_image must have shape [B, 3, H, W]; got {tuple(left_image.shape)}"
            )
        if left_image.shape[0] == 0 or left_image.shape[2] == 0 or left_image.shape[3] == 0:
            raise NativeInputError("left_image batch and spatial dimensions must be non-empty")
        _require_shape("right_image", right_image, tuple(left_image.shape))
        if left_image.dtype != right_image.dtype:
            raise NativeInputError(
                "right_image must use the same dtype as left_image; "
                f"got {right_image.dtype} and {left_image.dtype}"
            )

        batch_size, _, height, width = left_image.shape
        left_mask = _require_numeric_tensor("left_mask", self.left_mask)
        _require_shape("left_mask", left_mask, (batch_size, 1, height, width))
        disparity_constant = _require_numeric_tensor(
            "left_disparity_constant", self.left_disparity_constant
        )
        _require_shape("left_disparity_constant", disparity_constant, (batch_size,))

        left_intrinsics = _require_floating_tensor("left_intrinsics", self.left_intrinsics)
        right_intrinsics = _require_floating_tensor("right_intrinsics", self.right_intrinsics)
        left_extrinsics = _require_floating_tensor("left_extrinsics", self.left_extrinsics)
        _require_shape("left_intrinsics", left_intrinsics, (batch_size, 3, 3))
        _require_shape("right_intrinsics", right_intrinsics, (batch_size, 3, 3))
        _require_shape("left_extrinsics", left_extrinsics, (batch_size, 3, 4))

        left_fov_x = _require_floating_tensor("left_fov_x", self.left_fov_x)
        left_fov_y = _require_floating_tensor("left_fov_y", self.left_fov_y)
        left_width = _require_numeric_tensor("left_width", self.left_width)
        left_height = _require_numeric_tensor("left_height", self.left_height)
        world_view_transform = _require_floating_tensor(
            "left_world_view_transform", self.left_world_view_transform
        )
        full_projection_transform = _require_floating_tensor(
            "left_full_projection_transform", self.left_full_projection_transform
        )
        camera_center = _require_floating_tensor("left_camera_center", self.left_camera_center)
        _require_shape("left_fov_x", left_fov_x, (batch_size,))
        _require_shape("left_fov_y", left_fov_y, (batch_size,))
        _require_shape("left_width", left_width, (batch_size,))
        _require_shape("left_height", left_height, (batch_size,))
        _require_shape("left_world_view_transform", world_view_transform, (batch_size, 4, 4))
        _require_shape(
            "left_full_projection_transform", full_projection_transform, (batch_size, 4, 4)
        )
        _require_shape("left_camera_center", camera_center, (batch_size, 3))

        reference_device = left_image.device
        tensors = {
            "right_image": right_image,
            "left_mask": left_mask,
            "left_disparity_constant": disparity_constant,
            "left_intrinsics": left_intrinsics,
            "right_intrinsics": right_intrinsics,
            "left_extrinsics": left_extrinsics,
            "left_fov_x": left_fov_x,
            "left_fov_y": left_fov_y,
            "left_width": left_width,
            "left_height": left_height,
            "left_world_view_transform": world_view_transform,
            "left_full_projection_transform": full_projection_transform,
            "left_camera_center": camera_center,
        }
        for name, tensor in tensors.items():
            if tensor.device != reference_device:
                raise NativeInputError(
                    f"{name} must be on the same device as left_image ({reference_device}); "
                    f"got {tensor.device}"
                )
        object.__setattr__(self, "sample_names", _freeze_names(self.sample_names, batch_size))

    @property
    def batch_size(self) -> int:
        """Return ``B`` for the native ``lmain`` and ``rmain`` dictionaries."""

        return int(self.left_image.shape[0])

    def to_upstream_data(self, device: torch.device) -> dict[str, object]:
        """Return a private mutable native dictionary with tensors copied to ``device``.

        The returned shape is exactly the portion of a collated upstream sample
        read by ``StereoEndoModel.forward(..., is_train=False)`` and
        ``pts2render``. Every tensor is cloned before transfer, so mutations by
        the pinned code (including ``disp_const`` reshaping and output fields)
        cannot alter caller-owned tensor storage.
        """

        return {
            "name": self.sample_names,
            "lmain": {
                "img": _clone_to_device(self.left_image, device),
                "mask": _clone_to_device(self.left_mask, device),
                "disp_const": _clone_to_device(self.left_disparity_constant, device),
                "intr": _clone_to_device(self.left_intrinsics, device),
                "extr": _clone_to_device(self.left_extrinsics, device),
                "FovX": _clone_to_device(self.left_fov_x, device),
                "FovY": _clone_to_device(self.left_fov_y, device),
                "width": _clone_to_device(self.left_width, device),
                "height": _clone_to_device(self.left_height, device),
                "world_view_transform": _clone_to_device(self.left_world_view_transform, device),
                "full_proj_transform": _clone_to_device(
                    self.left_full_projection_transform, device
                ),
                "camera_center": _clone_to_device(self.left_camera_center, device),
            },
            "rmain": {
                "img": _clone_to_device(self.right_image, device),
                "intr": _clone_to_device(self.right_intrinsics, device),
            },
        }


@dataclass(frozen=True, slots=True)
class NativeParityCapture:
    """Native output snapshot and adapter views for one identical execution.

    ``native_output`` is captured immediately after the upstream renderer
    returns and before adapter conversion. ``stereo``, ``gaussians``, and
    ``render_left`` are then created by the canonical adapter without
    modifying that snapshot. This is comparison data, not a claim that native
    runtime or numerical parity has been established on a real checkpoint.
    """

    native_output: Mapping[str, object]
    stereo: StereoPrediction
    gaussians: GaussianField
    render_left: RenderOutput


@dataclass(frozen=True, slots=True)
class NativeInferenceResult:
    """Artifacts from one native stage-two inference call.

    ``renderer_payload`` is the source ``pts2render`` ``temp`` sequence. At
    the pinned revision it contains RGB, XYZ, ``None``, rotation, scale, and
    opacity for only the *last* batch item, so it is preserved for source
    parity inspection and is not treated as a batch-aligned project contract.
    """

    upstream_git_state: UpstreamGitState
    checkpoint: CheckpointLoadResult
    parity: NativeParityCapture
    renderer_payload: tuple[object, ...]

    @property
    def native_output(self) -> Mapping[str, object]:
        """Return the immutable-container native snapshot before adapter conversion."""

        return self.parity.native_output


def _freeze_native_value(value: object) -> object:
    if isinstance(value, Mapping):
        frozen = {key: _freeze_native_value(item) for key, item in value.items()}
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_freeze_native_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_native_value(item) for item in value)
    return value


def _freeze_native_output(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_native_value(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - fixed by the argument type.
        raise NativeRunnerError("internal error while freezing native output")
    return frozen


def _require_config_attribute(value: object, attribute: str, context: str) -> object:
    try:
        return getattr(value, attribute)
    except AttributeError as error:
        raise NativeRunnerError(f"{context} is missing required field {attribute!r}") from error


def _require_callable(value: object, name: str) -> Callable[..., object]:
    if not callable(value):
        raise NativeRunnerError(f"{name} must be callable; got {type(value).__name__}")
    return cast(Callable[..., object], value)


def _resolve_stage_two_config(baseline_config: BaselineConfig) -> tuple[Path, Path]:
    if baseline_config.upstream_commit != PINNED_COMMIT:
        raise NativeRunnerError(
            "native runner only supports the pinned upstream commit "
            f"{PINNED_COMMIT}; got {baseline_config.upstream_commit}"
        )
    if baseline_config.entry_point != "render.py":
        raise NativeRunnerError(
            "native runner only supports the audited upstream entry point 'render.py'; "
            f"got {baseline_config.entry_point!r}"
        )
    if baseline_config.checkpoint_id is None or baseline_config.checkpoint_path is None:
        raise NativeRunnerError(
            "native inference requires explicit checkpoint_id, checkpoint_path, and checkpoint_sha256"
        )
    if baseline_config.checkpoint_sha256 is None:
        raise NativeRunnerError(
            "native inference requires explicit checkpoint_id, checkpoint_path, and checkpoint_sha256"
        )

    upstream_root = baseline_config.upstream_root.resolve()
    config_path = (upstream_root / baseline_config.upstream_config).resolve()
    try:
        relative_config = config_path.relative_to(upstream_root)
    except ValueError as error:
        raise NativeRunnerError("upstream_config must resolve inside upstream_root") from error
    if relative_config.as_posix() != "config/stage2.yaml":
        raise NativeRunnerError(
            "native runner only supports the audited stage-two config 'config/stage2.yaml'; "
            f"got {relative_config.as_posix()!r}"
        )
    if not config_path.is_file():
        raise NativeRunnerError(f"upstream stage-two config is not a readable file: {config_path}")
    entry_point = upstream_root / baseline_config.entry_point
    if not entry_point.is_file():
        raise NativeRunnerError(f"upstream entry point is not a readable file: {entry_point}")
    return upstream_root, config_path


def _resolve_cuda_device(configured_device: str) -> torch.device:
    try:
        device = torch.device(configured_device)
    except (RuntimeError, TypeError) as error:
        raise NativeRunnerError(f"baseline device is invalid: {configured_device!r}") from error
    if device.type != "cuda":
        raise NativeRunnerError(
            "the pinned stage-two renderer hardcodes CUDA allocation; native inference requires a CUDA device"
        )
    if not torch.cuda.is_available():
        raise NativeRunnerError("CUDA is unavailable for pinned native stage-two inference")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise NativeRunnerError(
            f"requested CUDA device index {device_index} is unavailable; "
            f"device count is {torch.cuda.device_count()}"
        )
    return torch.device("cuda", device_index)


def _cuda_context(device: torch.device) -> AbstractContextManager[object]:
    """Keep upstream's literal ``device='cuda'`` allocations on the selected device."""

    if device.type != "cuda":
        return nullcontext()
    return cast(AbstractContextManager[object], torch.cuda.device(device))


def _load_stage_two_components(
    upstream_root: Path,
    *,
    enforce_clean: bool,
    config_path: Path,
) -> tuple[torch.nn.Module, Callable[..., object], object]:
    config_module = import_upstream_module(
        "config.stereo_config", upstream_root, enforce_clean=enforce_clean
    )
    network_module = import_upstream_module(
        "lib.network",
        upstream_root,
        require_corr_sampler=True,
        enforce_clean=enforce_clean,
    )
    renderer_module = import_upstream_module(
        "lib.GaussianRender",
        upstream_root,
        require_rasterizer=True,
        enforce_clean=enforce_clean,
    )

    config_constructor = _require_callable(
        _require_config_attribute(config_module, "ConfigStereo", "config.stereo_config"),
        "config.stereo_config.ConfigStereo",
    )
    config_instance = config_constructor()
    load_config = _require_callable(
        _require_config_attribute(config_instance, "load", "ConfigStereo instance"),
        "ConfigStereo.load",
    )
    get_config = _require_callable(
        _require_config_attribute(config_instance, "get_cfg", "ConfigStereo instance"),
        "ConfigStereo.get_cfg",
    )
    load_config(str(config_path))
    upstream_config = get_config()

    raft_config = _require_config_attribute(upstream_config, "raft", "stage-two config")
    correlation_implementation = _require_config_attribute(
        raft_config, "corr_implementation", "stage-two raft config"
    )
    if correlation_implementation != "reg_cuda":
        raise NativeRunnerError(
            "native runner refuses corr_implementation other than 'reg_cuda'; "
            "the pure-PyTorch 'reg' backend is not a parity fallback"
        )
    dataset_config = _require_config_attribute(upstream_config, "dataset", "stage-two config")
    background_color = _require_config_attribute(
        dataset_config, "bg_color", "stage-two dataset config"
    )
    if (
        isinstance(background_color, (str, bytes))
        or not isinstance(background_color, Sequence)
        or len(background_color) != 3
        or not all(isinstance(value, (int, float)) for value in background_color)
    ):
        raise NativeRunnerError(
            "stage-two dataset.bg_color must be a three-element numeric sequence"
        )

    model_constructor = _require_callable(
        _require_config_attribute(network_module, "StereoEndoModel", "lib.network"),
        "lib.network.StereoEndoModel",
    )
    model = model_constructor(upstream_config, with_gs_render=True)
    if not isinstance(model, torch.nn.Module):
        raise NativeRunnerError(
            "lib.network.StereoEndoModel must construct a torch.nn.Module; "
            f"got {type(model).__name__}"
        )
    pts2render = _require_callable(
        _require_config_attribute(renderer_module, "pts2render", "lib.GaussianRender"),
        "lib.GaussianRender.pts2render",
    )
    return model, pts2render, background_color


class NativeBaselineRunner:
    """Execute one unmodified, checkpoint-identified stage-two native inference.

    Construction, checkpoint loading, model evaluation, and rendering happen
    per :meth:`run` call. The runner therefore retains neither a mutable model
    nor caller-owned native dictionary between executions. It is intentionally
    a bounded Plan-01 parity surface rather than a dataset loader or training
    API.
    """

    def __init__(self, baseline_config: BaselineConfig) -> None:
        self._baseline_config = baseline_config

    @property
    def baseline_config(self) -> BaselineConfig:
        """Return the immutable execution configuration and checkpoint identity."""

        return self._baseline_config

    def run(self, native_input: NativeInferenceInput) -> NativeInferenceResult:
        """Run stage-two eval/test-mode inference and capture adapter parity inputs.

        The pinned model mutates its native dictionary; this method first
        clones all user tensors to the selected CUDA device. It uses
        ``torch.no_grad()``, calls the model with ``is_train=False`` (which
        makes upstream RAFT use ``test_mode=True``), and invokes the real
        ``pts2render`` implementation. No CPU execution or correlation
        fallback is supplied.
        """

        upstream_root, config_path = _resolve_stage_two_config(self._baseline_config)
        execution_device = _resolve_cuda_device(self._baseline_config.device)
        git_state = require_pinned_upstream(
            upstream_root, enforce_clean=self._baseline_config.enforce_clean
        )
        model, pts2render, background_color = _load_stage_two_components(
            upstream_root,
            enforce_clean=self._baseline_config.enforce_clean,
            config_path=config_path,
        )
        checkpoint_path = self._baseline_config.checkpoint_path
        checkpoint_sha256 = self._baseline_config.checkpoint_sha256
        if checkpoint_path is None or checkpoint_sha256 is None:  # guarded by resolution above.
            raise NativeRunnerError("internal checkpoint identity validation failure")

        with _cuda_context(execution_device):
            model.to(execution_device)
            checkpoint = load_upstream_checkpoint(
                model,
                checkpoint_path,
                expected_sha256=checkpoint_sha256,
                map_location=execution_device,
                strict=True,
            )
            model.eval()
            mutable_data = native_input.to_upstream_data(execution_device)
            with torch.no_grad():
                model_result = model(mutable_data, is_train=False)
                if not isinstance(model_result, tuple) or len(model_result) != 3:
                    raise NativeRunnerError(
                        "pinned StereoEndoModel inference must return (data, flow_loss, metrics)"
                    )
                rendered_data, flow_loss, metrics = model_result
                if flow_loss is not None or metrics is not None:
                    raise NativeRunnerError(
                        "pinned StereoEndoModel eval/test-mode inference must return None losses and metrics"
                    )
                if not isinstance(rendered_data, Mapping):
                    raise NativeRunnerError("pinned StereoEndoModel must return a mapping as data")
                render_result = pts2render(rendered_data, bg_color=background_color)
                if not isinstance(render_result, tuple) or len(render_result) != 2:
                    raise NativeRunnerError("pinned pts2render must return (data, temp)")
                rendered_data, renderer_payload = render_result
                if not isinstance(rendered_data, Mapping):
                    raise NativeRunnerError("pinned pts2render must return a mapping as data")
                if isinstance(renderer_payload, (str, bytes)) or not isinstance(
                    renderer_payload, Sequence
                ):
                    raise NativeRunnerError("pinned pts2render temp payload must be a sequence")

        native_output = _freeze_native_output(rendered_data)
        parity = NativeParityCapture(
            native_output=native_output,
            stereo=EndoE2EGSAdapter.stereo_prediction(native_output),
            gaussians=EndoE2EGSAdapter.gaussian_field(native_output),
            render_left=EndoE2EGSAdapter.render_output(native_output),
        )
        return NativeInferenceResult(
            upstream_git_state=git_state,
            checkpoint=checkpoint,
            parity=parity,
            renderer_payload=tuple(renderer_payload),
        )
