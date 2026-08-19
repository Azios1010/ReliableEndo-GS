"""Lossless structural translations from Endo-E2E-GS outputs to project contracts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from reliable_endo_gs.contracts import (
    GaussianField,
    ReconstructionState,
    RenderOutput,
    StereoBatch,
    StereoPrediction,
)


class BaselineAdapterError(ValueError):
    """Raised when an upstream output cannot be translated without guessing."""


def _mapping_field(values: Mapping[str, object], key: str, context: str) -> Mapping[str, object]:
    try:
        value = values[key]
    except KeyError as error:
        raise BaselineAdapterError(f"{context} is missing required field {key!r}") from error
    if not isinstance(value, Mapping):
        raise BaselineAdapterError(f"{context}.{key} must be a mapping")
    if not all(isinstance(item, str) for item in value):
        raise BaselineAdapterError(f"{context}.{key} keys must be strings")
    return value


def _tensor_field(values: Mapping[str, object], key: str, context: str) -> torch.Tensor:
    try:
        value = values[key]
    except KeyError as error:
        raise BaselineAdapterError(f"{context} is missing required tensor {key!r}") from error
    if not isinstance(value, torch.Tensor):
        raise BaselineAdapterError(
            f"{context}.{key} must be a torch.Tensor; got {type(value).__name__}"
        )
    return value


def _flatten_map(name: str, value: torch.Tensor, channels: int) -> torch.Tensor:
    if value.ndim != 4 or value.shape[1] != channels:
        raise BaselineAdapterError(
            f"upstream lmain.{name} must have shape [B, {channels}, H, W]; got {tuple(value.shape)}"
        )
    return value.permute(0, 2, 3, 1).reshape(value.shape[0], -1, channels)


@dataclass(frozen=True, slots=True)
class TensorParity:
    """Numerical comparison for one direct-upstream versus translated tensor."""

    shape: tuple[int, ...]
    dtype: str
    device: str
    max_abs_difference: float
    mean_abs_difference: float
    matches: bool
    absolute_tolerance: float
    relative_tolerance: float


def compare_tensor_translation(
    upstream: torch.Tensor,
    translated: torch.Tensor,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> TensorParity:
    """Compare values translated by the adapter on identical tensor semantics.

    This explicit harness may synchronize accelerator tensors and is therefore
    intended for parity validation, not model execution or profiling.
    """

    if upstream.shape != translated.shape:
        raise BaselineAdapterError(
            "parity tensors must have identical shapes; "
            f"got {tuple(upstream.shape)} and {tuple(translated.shape)}"
        )
    if upstream.dtype != translated.dtype:
        raise BaselineAdapterError(
            f"parity tensors must have identical dtypes; got {upstream.dtype} and {translated.dtype}"
        )
    if upstream.device != translated.device:
        raise BaselineAdapterError(
            f"parity tensors must share a device; got {upstream.device} and {translated.device}"
        )
    if upstream.numel() == 0:
        raise BaselineAdapterError("parity tensors must be non-empty")

    difference = (upstream.detach() - translated.detach()).abs().float()
    return TensorParity(
        shape=tuple(upstream.shape),
        dtype=str(upstream.dtype),
        device=str(upstream.device),
        max_abs_difference=float(difference.max().item()),
        mean_abs_difference=float(difference.mean().item()),
        matches=bool(
            torch.allclose(
                upstream,
                translated,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
        ),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


class EndoE2EGSAdapter:
    """Translate the pinned upstream model's dictionary outputs.

    The official model mutates a nested ``data['lmain']`` mapping. This class
    is the only project-owned location that understands those keys. It does
    not import upstream modules and does not recreate disparity, geometry, or
    rendering formulas.
    """

    @staticmethod
    def stereo_prediction(
        upstream_output: Mapping[str, object],
        *,
        disparity_iterations: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> StereoPrediction:
        """Map ``flow_pred`` and the upstream supervision mask.

        ``flow_pred`` remains the same tensor. The upstream mask is converted
        from its numeric 0/1 representation with the same ``>= 0.5`` rule used
        by upstream evaluation. Official test-mode inference exposes only the
        final disparity, so iterations are absent unless a direct upstream
        call explicitly supplies them.
        """

        lmain = _mapping_field(upstream_output, "lmain", "upstream output")
        disparity = _tensor_field(lmain, "flow_pred", "upstream output.lmain")
        mask = _tensor_field(lmain, "mask", "upstream output.lmain")
        valid_mask = mask if mask.dtype == torch.bool else mask >= 0.5
        return StereoPrediction(
            disparity=disparity,
            valid_mask=valid_mask,
            disparity_iterations=disparity_iterations,
            diagnostics={
                "baseline": "Intelligent-Imaging-Center/Endo-E2E-GS",
                "upstream_field": "lmain.flow_pred",
                "validity_rule": "lmain.mask >= 0.5",
                "test_mode_iterations_exposed": False,
            },
        )

    @staticmethod
    def gaussian_field(upstream_output: Mapping[str, object]) -> GaussianField:
        """Map the full upstream pixel-aligned Gaussian grid without filtering.

        Upstream stores attributes as image maps and filters them only inside
        ``pts2render``. The contract retains the full grid plus ``pts_valid``.
        RGB is transformed from upstream image space ``[-1, 1]`` to renderer
        color space ``[0, 1]`` using upstream's exact ``x * 0.5 + 0.5`` step.
        """

        lmain = _mapping_field(upstream_output, "lmain", "upstream output")
        means3d = _tensor_field(lmain, "xyz", "upstream output.lmain")
        image = _tensor_field(lmain, "img", "upstream output.lmain")
        rotations_map = _tensor_field(lmain, "rot_maps", "upstream output.lmain")
        scales_map = _tensor_field(lmain, "scale_maps", "upstream output.lmain")
        opacities_map = _tensor_field(lmain, "opacity_maps", "upstream output.lmain")
        valid_mask = _tensor_field(lmain, "pts_valid", "upstream output.lmain")

        colors = _flatten_map("img", image, 3) * 0.5 + 0.5
        rotations = _flatten_map("rot_maps", rotations_map, 4)
        scales = _flatten_map("scale_maps", scales_map, 3)
        opacities = _flatten_map("opacity_maps", opacities_map, 1)
        if means3d.ndim != 3 or means3d.shape[2] != 3:
            raise BaselineAdapterError(
                f"upstream lmain.xyz must have shape [B, N, 3]; got {tuple(means3d.shape)}"
            )
        if valid_mask.dtype != torch.bool:
            raise BaselineAdapterError(
                "upstream lmain.pts_valid must be boolean; refusing to infer validity semantics"
            )

        return GaussianField(
            means3d=means3d,
            colors=colors,
            rotations=rotations,
            scales=scales,
            opacities=opacities,
            valid_mask=valid_mask,
        )

    @staticmethod
    def render_output(upstream_output: Mapping[str, object]) -> RenderOutput:
        """Map the RGB image actually returned by the upstream rasterizer path."""

        lmain = _mapping_field(upstream_output, "lmain", "upstream output")
        image = _tensor_field(lmain, "img_pred", "upstream output.lmain")
        return RenderOutput(image=image)

    @classmethod
    def reconstruction_state(
        cls,
        batch: StereoBatch,
        upstream_output: Mapping[str, object],
        *,
        provenance: Mapping[str, object],
        disparity_iterations: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> ReconstructionState:
        """Assemble mapped official outputs into the Phase-I-neutral state contract."""

        return ReconstructionState(
            batch=batch,
            stereo=cls.stereo_prediction(
                upstream_output,
                disparity_iterations=disparity_iterations,
            ),
            gaussians=cls.gaussian_field(upstream_output),
            render_left=cls.render_output(upstream_output),
            diagnostics={"baseline_mode": "official_unmodified"},
            provenance=provenance,
        )
