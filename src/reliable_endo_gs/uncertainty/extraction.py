"""P1 inference-only extraction from the pinned stereo tensor contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from reliable_endo_gs.contracts import StereoPrediction
from reliable_endo_gs.uncertainty.proxies import (
    FinalUpdateMagnitudeProxy,
    IterationDisagreementProxy,
    left_right_consistency_score,
    photometric_residual_score,
)
from reliable_endo_gs.uncertainty.records import RawUncertaintyScore


P1_PROXY_IDS = (
    "final_update_magnitude",
    "iteration_disagreement",
    "left_right_consistency",
    "photometric_residual",
)


@dataclass(frozen=True, slots=True)
class ProxyOutputs:
    """P1 raw scores and inference-only validity maps for one stereo batch."""

    disparity: torch.Tensor
    disparity_iterations: torch.Tensor
    inference_valid_mask: torch.Tensor
    update_magnitude: RawUncertaintyScore
    iteration_disagreement: RawUncertaintyScore
    lr_consistency: RawUncertaintyScore | None
    photometric_residual: RawUncertaintyScore
    correlation_entropy: RawUncertaintyScore | None
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        expected = self.disparity.shape
        if self.disparity.ndim != 4 or self.disparity.shape[1] != 1:
            raise ValueError("disparity must have shape [B, 1, H, W]")
        if self.disparity_iterations.ndim != 4 or self.disparity_iterations.shape[0] != expected[0]:
            raise ValueError("disparity_iterations must have shape [B, K, H, W]")
        if self.disparity_iterations.shape[1] < 2 or tuple(self.disparity_iterations.shape[2:]) != tuple(expected[2:]):
            raise ValueError("disparity_iterations must contain aligned spatial predictions")
        if self.inference_valid_mask.shape != expected or self.inference_valid_mask.dtype != torch.bool:
            raise ValueError("inference_valid_mask must be boolean and match disparity")
        if self.correlation_entropy is not None:
            raise ValueError("correlation entropy is unavailable for the pinned upstream contract")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def scores(self) -> dict[str, RawUncertaintyScore]:
        """Return only available scores in stable protocol order."""

        values: dict[str, RawUncertaintyScore] = {
            "final_update_magnitude": self.update_magnitude,
            "iteration_disagreement": self.iteration_disagreement,
            "photometric_residual": self.photometric_residual,
        }
        if self.lr_consistency is not None:
            values["left_right_consistency"] = self.lr_consistency
        return values


def _stack_iterations(
    iterations: torch.Tensor | Sequence[torch.Tensor],
) -> torch.Tensor:
    if isinstance(iterations, torch.Tensor):
        if iterations.ndim != 4:
            raise ValueError("iteration tensor must have shape [B, K, H, W]")
        stacked = iterations
    else:
        values = tuple(iterations)
        if len(values) < 2:
            raise ValueError("at least two disparity iterations are required")
        if any(value.ndim != 4 or value.shape[1] != 1 for value in values):
            raise ValueError("iteration sequence values must have shape [B, 1, H, W]")
        stacked = torch.cat(values, dim=1)
    if stacked.shape[1] < 2:
        raise ValueError("at least two disparity iterations are required")
    if not torch.isfinite(stacked).all():
        raise ValueError("disparity iterations must be finite")
    return stacked


def extract_proxy_outputs(
    disparity_iterations: torch.Tensor | Sequence[torch.Tensor],
    left_image: torch.Tensor,
    right_image: torch.Tensor,
    *,
    right_disparity_iterations: torch.Tensor | Sequence[torch.Tensor] | None = None,
    iteration_window: int | None = None,
) -> ProxyOutputs:
    """Compute preregistered P1 proxies without accepting any GT tensor.

    ``right_disparity_iterations`` must come from a separate swapped-image
    forward and uses the native right-reference sign.  If omitted, LR
    consistency is explicitly unavailable rather than approximated.
    """

    left_iterations = _stack_iterations(disparity_iterations)
    disparity = left_iterations[:, -1:]
    inference_valid = torch.isfinite(disparity) & (disparity > 0.0)
    prediction = StereoPrediction(
        disparity=disparity,
        valid_mask=inference_valid,
        disparity_iterations=left_iterations,
    )
    update = FinalUpdateMagnitudeProxy().predict(prediction)
    disagreement = IterationDisagreementProxy(
        window=iteration_window or left_iterations.shape[1]
    ).predict(prediction)
    photometric = photometric_residual_score(
        left_image,
        right_image,
        disparity,
        valid_mask=inference_valid,
    )

    lr_score: RawUncertaintyScore | None = None
    if right_disparity_iterations is not None:
        right_iterations = _stack_iterations(right_disparity_iterations)
        if right_iterations.shape != left_iterations.shape:
            raise ValueError("left and right iteration stacks must have identical shapes")
        right_disparity = right_iterations[:, -1:]
        right_valid = torch.isfinite(right_disparity) & (right_disparity > 0.0)
        lr_score = left_right_consistency_score(
            disparity,
            right_disparity,
            inference_valid,
            right_valid,
        )

    return ProxyOutputs(
        disparity=disparity,
        disparity_iterations=left_iterations,
        inference_valid_mask=inference_valid,
        update_magnitude=update,
        iteration_disagreement=disagreement,
        lr_consistency=lr_score,
        photometric_residual=photometric,
        correlation_entropy=None,
        metadata={
            "proxy_contract": "phase1_p1.v1",
            "image_space": "[-1, 1]",
            "disparity_convention": "positive left-reference pixels",
            "iteration_std_unbiased": False,
            "right_reference_available": right_disparity_iterations is not None,
            "correlation_entropy_available": False,
        },
    )


def extract_model_iterations(
    model: Any,
    left_image: torch.Tensor,
    right_image: torch.Tensor,
    *,
    iterations: int | None = None,
) -> torch.Tensor:
    """Run the pinned encoder/RAFT path and retain every recurrent prediction.

    The model must expose the audited ``img_encoder`` and ``raft_stereo``
    attributes.  This calls RAFT with ``test_mode=False`` solely to preserve
    its returned iteration sequence; it does not modify model parameters or
    invoke the Gaussian renderer.
    """

    if not isinstance(left_image, torch.Tensor) or not isinstance(right_image, torch.Tensor):
        raise TypeError("left_image and right_image must be tensors")
    if left_image.shape != right_image.shape or left_image.ndim != 4 or left_image.shape[1] != 3:
        raise ValueError("left_image and right_image must share shape [B, 3, H, W]")
    if not hasattr(model, "img_encoder") or not hasattr(model, "raft_stereo"):
        raise TypeError("model must expose img_encoder and raft_stereo")
    image_pair = torch.cat((left_image, right_image), dim=0)
    features = model.img_encoder(image_pair)
    count = int(iterations or getattr(model, "val_iters", 0))
    if count < 2:
        raise ValueError("the pinned RAFT contract requires at least two iterations")
    predictions = model.raft_stereo(features[2], iters=count, test_mode=False)
    return _stack_iterations(predictions)


def extract_model_proxies(
    model: Any,
    left_image: torch.Tensor,
    right_image: torch.Tensor,
    *,
    iterations: int | None = None,
    include_left_right: bool = False,
) -> ProxyOutputs:
    """Extract P1 maps, optionally paying for a swapped-image second forward."""

    left_iterations = extract_model_iterations(
        model, left_image, right_image, iterations=iterations
    )
    right_iterations = None
    if include_left_right:
        right_iterations = extract_model_iterations(
            model, right_image, left_image, iterations=iterations
        )
    return extract_proxy_outputs(
        left_iterations,
        left_image,
        right_image,
        right_disparity_iterations=right_iterations,
        iteration_window=iterations,
    )


def availability_table(*, include_left_right: bool = False) -> tuple[dict[str, object], ...]:
    """Return the static P1 capability table without probing model tensors."""

    return (
        {
            "proxy": "final_update_magnitude",
            "available": True,
            "extra_forward": False,
            "units": "disparity pixels",
            "reason": "RAFT returns all three validation iterations with test_mode=False",
        },
        {
            "proxy": "iteration_disagreement",
            "available": True,
            "extra_forward": False,
            "units": "disparity pixels",
            "reason": "aligned recurrent disparity stack is exposed",
        },
        {
            "proxy": "left_right_consistency",
            "available": include_left_right,
            "extra_forward": True,
            "units": "disparity pixels",
            "reason": "requires a valid swapped-image right-reference forward",
        },
        {
            "proxy": "photometric_residual",
            "available": True,
            "extra_forward": False,
            "units": "normalized image-space absolute RGB residual",
            "reason": "rectified right image and final disparity are available",
        },
        {
            "proxy": "correlation_entropy",
            "available": False,
            "extra_forward": False,
            "units": None,
            "reason": "pinned source exposes feature correlations, not a normalized probability distribution",
        },
    )
