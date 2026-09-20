"""CPU-only checks for the local Stage2 scale activation and diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_scale_sigmoid_mapping_is_bounded_and_has_cpu_gradients(
    stage2_modules: SimpleNamespace,
) -> None:
    scale_parameterization = stage2_modules.scale_parameterization
    raw_scale = torch.tensor([[-2.0, 0.0, 2.0]], requires_grad=True)

    activated_scale = scale_parameterization.activate_scale(raw_scale)
    activated_scale.sum().backward()

    expected = 0.01 * torch.sigmoid(raw_scale.detach())
    assert torch.allclose(activated_scale.detach(), expected)
    assert torch.all(activated_scale.detach() > 0.0)
    assert torch.all(activated_scale.detach() < 0.01)
    assert raw_scale.grad is not None
    assert torch.all(raw_scale.grad > 0.0)
    assert raw_scale.grad[0, 1].item() == pytest.approx(0.0025)


def test_stage2_diagnostics_report_expected_observability_keys(
    stage2_modules: SimpleNamespace,
) -> None:
    diagnostics = stage2_modules.stage2_diagnostics

    parameter_metrics = diagnostics.gaussian_parameter_metrics(
        torch.tensor([[-5.0, 0.0, 5.0]]),
        torch.tensor([[0.00001, 0.005, 0.00999]]),
        torch.tensor([[0.1, 0.5, 0.9]]),
        quantiles=(0.01, 0.5, 0.99),
        boundary_margin=0.01,
    )
    render_metrics = diagnostics.render_diagnostics(
        torch.tensor([0, 3, 5]),
        torch.zeros(1, 3, 2, 2),
        background=(0.0, 0.0, 0.0),
        quantiles=(0.01, 0.5, 0.99),
    )

    assert parameter_metrics["scale/raw/p50"] == pytest.approx(0.0)
    assert parameter_metrics["scale/activated/p50"] == pytest.approx(0.005)
    assert parameter_metrics["opacity/mean"] == pytest.approx(0.5)
    assert parameter_metrics["scale/boundary_fraction"] == pytest.approx(2.0 / 3.0)
    assert render_metrics["render/submitted_gaussians"] == 3.0
    assert render_metrics["render/visible_gaussians"] == 2.0
    assert render_metrics["render/visible_gaussian_fraction"] == pytest.approx(2.0 / 3.0)
    assert render_metrics["render/nonbackground_pixel_fraction"] == 0.0


def test_scale_gradcheck_and_extreme_numerical_saturation(stage2_modules: SimpleNamespace) -> None:
    activate = stage2_modules.scale_parameterization.activate_scale
    raw = torch.linspace(-5, 5, 6, dtype=torch.double).reshape(1, 3, 1, 2).requires_grad_()
    assert torch.autograd.gradcheck(activate, (raw,))
    extreme = torch.tensor([-100.0, 20.0], requires_grad=True)
    scales = activate(extreme)
    scales.sum().backward()
    assert torch.isfinite(scales).all()
    assert extreme.grad is not None
    assert (extreme.grad == 0).all()  # Floating-point sigmoid can saturate; no false guarantee.


def test_pixel_proxy_is_independent_of_gaussian_visibility(stage2_modules: SimpleNamespace) -> None:
    image = torch.zeros(1, 3, 2, 2)
    image[0, 0, 0, 0] = 1.0
    image[0, 1, 1, 1] = float("nan")
    before = image.clone()
    metrics = stage2_modules.stage2_diagnostics.render_diagnostics(
        torch.tensor([1, 1]),
        image,
        background=(0, 0, 0),
        quantiles=(0.5,),
    )
    assert metrics["render/visible_gaussian_fraction"] == 1.0
    assert metrics["render/nonbackground_pixel_fraction"] == 0.25
    assert metrics["render/nonfinite_pixel_fraction"] == 0.25
    torch.testing.assert_close(image, before, equal_nan=True)
    assert "render/coverage_fraction" not in metrics


def test_background_color_and_empty_radii(stage2_modules: SimpleNamespace) -> None:
    metrics = stage2_modules.stage2_diagnostics.render_diagnostics(
        torch.empty(0),
        torch.full((2, 3, 2, 2), 0.5),
        background=(0.5, 0.5, 0.5),
        quantiles=(0.5,),
    )
    assert metrics["render/submitted_gaussians"] == 0
    assert metrics["render/visible_gaussian_fraction"] == 0
    assert metrics["render/nonbackground_pixel_fraction"] == 0


def test_diagnostics_do_not_change_gradients_or_rng(stage2_modules: SimpleNamespace) -> None:
    diagnostics = stage2_modules.stage2_diagnostics
    head = torch.nn.Linear(2, 1, bias=False)
    head(torch.tensor([[3.0, 4.0]])).sum().backward()
    original = head.weight.grad.clone()
    assert diagnostics.head_gradient_norm(head) == 5.0
    assert torch.equal(head.weight.grad, original)
    raw = torch.zeros(1, 3, 2, 2, requires_grad=True)
    scale = stage2_modules.scale_parameterization.activate_scale(raw)
    opacity = torch.full((1, 1, 2, 2), 0.5, requires_grad=True)
    rng = torch.get_rng_state().clone()
    metrics = diagnostics.gaussian_parameter_metrics(
        raw,
        scale,
        opacity,
        quantiles=(0.01, 0.5, 0.99),
        boundary_margin=0.01,
    )
    assert all(isinstance(value, float) for value in metrics.values())
    assert torch.equal(rng, torch.get_rng_state())
    assert raw.grad is None and opacity.grad is None
    scale.sum().backward()
    assert raw.grad is not None and (raw.grad > 0).all()
