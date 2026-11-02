"""Memory and precision policy that lets MVSplat and DepthSplat train at the full 512x640 A00 grid.

Measured on one RTX 5070 Ti (15.9 GiB, cap 0.90): without this policy both models run out of memory in the encoder forward at
batch one (the multi-view transformer attention grows with the square of the token count). With it, a full forward and
backward peaks near 9 GiB (MVSplat) and 10 GiB (DepthSplat). The policy changes memory, not the architecture, the resolution
or the learned layers:

* ``checkpoint_module`` recomputes a module's activations in the backward pass (gradient checkpointing, same values).
* ``fp32_island`` runs geometry and covariance math in float32 inside an autocast region (matrix inverses are not defined for
  half precision, and half covariances of metre-scale Gaussians would underflow).
* ``amp_safe_checkpoint`` replaces the upstream UNet's hand-written checkpoint function, whose backward recomputation ignores
  autocast and fails with mixed dtypes.

Upstream source files are not edited. The patches are module-level assignments made once per process by
:func:`apply_memory_policy`, which must run before the first training forward and returns a record for provenance.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import torch
import torch.utils.checkpoint as torch_checkpoint
from torch import nn

_APPLIED: set[int] = set()


def fp32_island(function: Callable[..., Any]) -> Callable[..., Any]:
    """Run ``function`` with autocast disabled and floating tensor arguments promoted to float32."""

    def promote(value: Any) -> Any:
        return value.float() if torch.is_tensor(value) and value.is_floating_point() else value

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
            return function(
                *[promote(a) for a in args], **{k: promote(v) for k, v in kwargs.items()}
            )

    return wrapped


def checkpoint_module(module: nn.Module) -> None:
    """Make ``module`` recompute its activations in backward (non-reentrant; keeps the autocast state)."""
    original = module.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        if torch.is_grad_enabled():
            return torch_checkpoint.checkpoint(original, *args, use_reentrant=False, **kwargs)
        return original(*args, **kwargs)

    module.forward = forward  # type: ignore[method-assign]


def amp_safe_checkpoint(
    function: Callable[..., Any], inputs: tuple[Any, ...], params: Any, flag: bool
) -> Any:
    """Drop-in for the latent-diffusion ``checkpoint(func, inputs, params, flag)`` helper of the upstream UNets."""
    if flag:
        return torch_checkpoint.checkpoint(function, *inputs, use_reentrant=False)
    return function(*inputs)


def _float_warp(original: Callable[..., Any]) -> Callable[..., Any]:
    """Camera-geometry warp in float32; the warped features return in the feature dtype."""

    def warp(feature: torch.Tensor, intrinsics: Any, pose: Any, depth: Any, **kwargs: Any) -> Any:
        with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
            out = original(
                feature.float(), intrinsics.float(), pose.float(), depth.float(), **kwargs
            )
        return out.to(feature.dtype)

    return warp


def apply_memory_policy(kind: str, predictor: nn.Module) -> dict[str, Any]:
    """Apply the policy to a native MVSplat or DepthSplat predictor (launch from the upstream repository)."""
    if kind not in ("mvsplat", "depthsplat"):
        raise ValueError("memory policy is defined for mvsplat and depthsplat only")
    if id(predictor) in _APPLIED:
        raise RuntimeError("memory policy already applied to this predictor")
    encoder = predictor.encoder  # type: ignore[attr-defined]
    record: dict[str, Any] = {
        "kind": kind,
        "checkpointed": [],
        "fp32_islands": [],
        "patched_upstream": [],
    }
    if kind == "mvsplat":
        for index, layer in enumerate(encoder.backbone.transformer.layers):
            checkpoint_module(layer)
            record["checkpointed"].append(f"backbone.transformer.layers.{index}")
        depth = importlib.import_module("src.model.encoder.costvolume.depth_predictor_multiview")
        unet = importlib.import_module("src.model.encoder.costvolume.ldm_unet.unet")
        depth.warp_with_pose_depth_candidates = _float_warp(depth.warp_with_pose_depth_candidates)
        depth.prepare_feat_proj_data_lists = fp32_island(depth.prepare_feat_proj_data_lists)
        record["fp32_islands"] += [
            "warp_with_pose_depth_candidates",
            "prepare_feat_proj_data_lists",
        ]
    else:
        predictor_net = encoder.depth_predictor
        for index, layer in enumerate(predictor_net.transformer.layers):
            checkpoint_module(layer)
            record["checkpointed"].append(f"depth_predictor.transformer.layers.{index}")
        for index, regressor in enumerate(predictor_net.regressor):
            checkpoint_module(regressor)
            record["checkpointed"].append(f"depth_predictor.regressor.{index}")
        for name, module in (
            ("depth_predictor.upsampler", predictor_net.upsampler),
            ("gaussian_regressor", encoder.gaussian_regressor),
            ("feature_upsampler", encoder.feature_upsampler),
        ):
            checkpoint_module(module)
            record["checkpointed"].append(name)
        matching = importlib.import_module("src.model.encoder.unimatch.matching")
        unet = importlib.import_module("src.model.encoder.unimatch.ldm_unet.unet")
        matching.warp_with_pose_depth_candidates = _float_warp(
            matching.warp_with_pose_depth_candidates
        )
        record["fp32_islands"].append("warp_with_pose_depth_candidates")
    unet.checkpoint = amp_safe_checkpoint
    record["patched_upstream"].append(f"{unet.__name__}.checkpoint -> amp_safe_checkpoint")
    encoder.gaussian_adapter.forward = fp32_island(encoder.gaussian_adapter.forward)
    record["fp32_islands"].append("gaussian_adapter.forward")
    _APPLIED.add(id(predictor))
    return record
