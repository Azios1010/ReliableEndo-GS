import torch

from reliable_endo_gs.probabilistic_gs import (
    RepresentationVariant,
    build_probabilistic_representation,
)
from reliable_endo_gs.rendering import CovarianceRequest, RendererRequest, validate_renderer_request


def test_plan06_center_covariance_reaches_renderer_contract_without_provider_branch() -> None:
    means = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64)
    rotations = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    scales = torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float64)
    opacity = torch.tensor([[[0.8]]], dtype=torch.float64)
    center_covariance = torch.zeros((1, 1, 3, 3), dtype=torch.float64)
    center_covariance[..., 2, 2] = 0.01

    representation = build_probabilistic_representation(
        means,
        rotations,
        scales,
        opacity,
        center_covariance,
        torch.ones((1, 1), dtype=torch.bool),
        frame="world",
        length_unit="m",
        variant=RepresentationVariant.COVARIANCE_ONLY,
    )
    request = RendererRequest.from_representation(
        representation, covariance_source=CovarianceRequest.EFFECTIVE
    )

    assert request.covariance is representation.cov_effective
    assert validate_renderer_request(request).valid
