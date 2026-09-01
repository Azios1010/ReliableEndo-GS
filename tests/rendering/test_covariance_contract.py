import pytest
import torch

from reliable_endo_gs.probabilistic_gs import build_probabilistic_representation
from reliable_endo_gs.rendering import (
    CovarianceRequest,
    RendererRequest,
    renderer_contract_metadata,
    validate_renderer_request,
)


def _representation():
    means = torch.zeros((1, 2, 3), dtype=torch.float64)
    rotations = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    scales = torch.ones((1, 2, 3), dtype=torch.float64)
    opacity = torch.full((1, 2, 1), 0.5, dtype=torch.float64)
    center = torch.zeros((1, 2, 3, 3), dtype=torch.float64)
    center[..., 2, 2] = 0.25
    return build_probabilistic_representation(
        means,
        rotations,
        scales,
        opacity,
        center,
        torch.ones((1, 2), dtype=torch.bool),
        frame="world",
        length_unit="m",
    )


def test_renderer_request_explicitly_selects_effective_covariance() -> None:
    request = RendererRequest.from_representation(
        _representation(), covariance_source=CovarianceRequest.EFFECTIVE
    )

    result = validate_renderer_request(request)
    metadata = renderer_contract_metadata(request)
    assert result.valid
    assert request.covariance_source is CovarianceRequest.EFFECTIVE
    assert request.covariance is not None
    assert metadata["covariance_source"] == "effective"
    assert metadata["production_parity_validated"] is False


def test_baseline_request_has_no_covariance() -> None:
    representation = _representation()
    baseline = representation.__class__(
        means3d=representation.means3d,
        base_opacity=representation.base_opacity,
        effective_opacity=representation.base_opacity,
        valid_mask=representation.valid_mask,
        frame="world",
        length_unit="m",
        variant="baseline",
        colors=representation.colors,
    )
    request = RendererRequest.from_representation(
        baseline, covariance_source=CovarianceRequest.NONE
    )
    assert request.covariance is None
    assert request.variant.value == "baseline"


def test_renderer_request_rejects_hidden_covariance_broadcast_and_bad_opacity() -> None:
    representation = _representation()
    with pytest.raises(ValueError, match="must have shape"):
        RendererRequest(
            means3d=representation.means3d,
            opacities=representation.effective_opacity,
            valid_mask=representation.valid_mask,
            covariance_source=CovarianceRequest.EFFECTIVE,
            covariance=torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3),
            variant=representation.variant,
            frame="world",
            length_unit="m",
        )


def test_renderer_request_rejects_non_psd_covariance() -> None:
    representation = _representation()
    covariance = representation.cov_effective.clone()
    covariance[0, 0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="positive definite"):
        RendererRequest(
            means3d=representation.means3d,
            opacities=representation.effective_opacity,
            valid_mask=representation.valid_mask,
            covariance_source=CovarianceRequest.EFFECTIVE,
            covariance=covariance,
            variant=representation.variant,
            frame="world",
            length_unit="m",
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        RendererRequest(
            means3d=representation.means3d,
            opacities=torch.full((1, 2, 1), 1.5, dtype=torch.float64),
            valid_mask=representation.valid_mask,
            covariance_source=CovarianceRequest.NONE,
            covariance=None,
            variant=representation.variant,
            frame="world",
            length_unit="m",
        )
