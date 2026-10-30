"""FieldRefiner: Stage A parity, identity at init, bounds, input ablations and gradients."""

import pytest
import torch

from reliable_endo_gs.actions.refinement import RefinedInference, refine_gaussian_field
from reliable_endo_gs.actions.repair import Action
from reliable_endo_gs.gaussians import ReferenceAttributeHead
from reliable_endo_gs.gaussians.refinement import (
    ALL_CONTEXT_CHANNELS,
    NO_ABSOLUTE_XY_CHANNELS,
    FieldRefiner,
    FieldRefinerConfig,
)
from reliable_endo_gs.regions.grid import grid_regions
from tests.routing.test_frozen_a04 import setup  # noqa: F401  (pytest fixture)


def _randomise(module: torch.nn.Module, seed: int = 0, scale: float = 0.8) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * scale)


def test_parameter_counts_pin_the_stage_a_architecture_and_the_ladder():
    assert FieldRefiner().parameter_count == 1735  # 13-32-32-7, the Stage A Gaussian head
    assert FieldRefiner(FieldRefinerConfig(hidden_dims=(128, 128))).parameter_count == 19207
    assert FieldRefiner(FieldRefinerConfig(hidden_dims=(), input_channels=())).parameter_count == 14
    assert (
        FieldRefiner(
            FieldRefinerConfig(hidden_dims=(), input_channels=ALL_CONTEXT_CHANNELS)
        ).parameter_count
        == 98
    )
    assert (
        len(NO_ABSOLUTE_XY_CHANNELS) == 11
        and 3 not in NO_ABSOLUTE_XY_CHANNELS
        and 5 in NO_ABSOLUTE_XY_CHANNELS
    )


def test_identity_at_initialisation_with_and_without_colour():
    context = torch.randn(2, 9, 13)
    for config in (
        FieldRefinerConfig(),
        FieldRefinerConfig(color_residual=True),
        FieldRefinerConfig(hidden_dims=(), input_channels=()),
    ):
        out = FieldRefiner(config)(context)
        for value in (
            out.attributes.delta_rotation,
            out.attributes.delta_log_scale,
            out.attributes.delta_alpha_logit,
        ):
            assert torch.equal(value, torch.zeros_like(value))
        assert (out.delta_color is None) == (not config.color_residual)
        if out.delta_color is not None:
            assert torch.equal(out.delta_color, torch.zeros_like(out.delta_color))


def test_stage_a_head_weights_load_and_reproduce_its_residuals_bitwise():
    head = ReferenceAttributeHead(hidden_dims=(32, 32))
    _randomise(head)
    refiner = FieldRefiner()
    refiner.load_state_dict(head.state_dict(), strict=True)  # same body / output names
    context = torch.randn(1, 40, 13)
    valid = torch.rand(1, 40) > 0.3
    expected = head(context, valid)
    actual = refiner(context, valid).attributes
    for a, b in zip(
        (actual.delta_rotation, actual.delta_log_scale, actual.delta_alpha_logit),
        (expected.delta_rotation, expected.delta_log_scale, expected.delta_alpha_logit),
        strict=True,
    ):
        assert torch.equal(a, b)


def test_outputs_respect_bounds_and_invalid_primitives_are_zero():
    refiner = FieldRefiner(FieldRefinerConfig(color_residual=True))
    _randomise(refiner, scale=30.0)
    valid = torch.zeros(1, 12, dtype=torch.bool)
    valid[0, :6] = True
    out = refiner(torch.randn(1, 12, 13) * 5, valid)
    b = refiner.config.bounds
    assert out.attributes.delta_rotation.abs().max() <= b.max_rotation_rad + 1e-6
    assert out.attributes.delta_log_scale.abs().max() <= b.max_log_scale + 1e-6
    assert out.attributes.delta_alpha_logit.abs().max() <= b.max_alpha_logit + 1e-6
    assert out.delta_color.abs().max() <= refiner.config.max_color_delta + 1e-6
    assert not out.delta_color[0, 6:].any()


def test_input_channel_selection_hides_absolute_position_but_not_depth():
    refiner = FieldRefiner(FieldRefinerConfig(input_channels=NO_ABSOLUTE_XY_CHANNELS))
    _randomise(refiner)
    context = torch.randn(1, 20, 13)
    base = refiner(context).attributes.delta_log_scale
    moved_xy = context.clone()
    moved_xy[..., 3:5] += 7.0
    assert torch.equal(refiner(moved_xy).attributes.delta_log_scale, base)
    moved_z = context.clone()
    moved_z[..., 5] += 7.0
    assert not torch.equal(refiner(moved_z).attributes.delta_log_scale, base)
    constant = FieldRefiner(FieldRefinerConfig(hidden_dims=(), input_channels=()))
    _randomise(constant)
    assert torch.equal(
        constant(context).attributes.delta_log_scale[:, :1].expand(1, 20, 3),
        constant(torch.randn(1, 20, 13)).attributes.delta_log_scale[:, :1].expand(1, 20, 3),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_dims": (0,)},
        {"input_channels": (13,)},
        {"input_channels": (1, 1)},
        {"max_color_delta": 0.0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FieldRefinerConfig(**kwargs)


def test_refine_gaussian_field_reproduces_the_stage_a_gaussian_action(setup):  # noqa: F811
    learner, state = setup
    _randomise(learner.actions.gaussian, scale=0.5)
    region = grid_regions((5, 5), (5, 5), 4)[0]
    expected = learner.actions.execute(state, region, Action.GAUSSIAN, learner.reference)
    refiner = FieldRefiner(
        FieldRefinerConfig(hidden_dims=(8, 8), bounds=learner.reference.config.residual_bounds)
    )
    refiner.load_state_dict(learner.actions.gaussian.state_dict(), strict=True)
    actual = refine_gaussian_field(
        state,
        refiner,
        learner.reference,
        composition_alpha_epsilon=learner.actions.gaussian_bounds.alpha_logit_epsilon,
        region=region,
        exact_zero_identity=learner.actions.exact_zero_identity,
    )
    for name in ("rotations", "scales", "opacities", "colors", "means3d"):
        assert torch.equal(getattr(actual.field, name), getattr(expected.field, name)), name
    assert torch.equal(actual.left.image, expected.left.image)


def test_zero_refiner_is_identity_and_gradients_reach_every_layer(setup):  # noqa: F811
    learner, state = setup
    region = grid_regions((5, 5), (5, 5), 4)[0]
    refiner = FieldRefiner(
        FieldRefinerConfig(
            hidden_dims=(8, 8), color_residual=True, bounds=learner.reference.config.residual_bounds
        )
    )
    kwargs = {
        "composition_alpha_epsilon": learner.actions.gaussian_bounds.alpha_logit_epsilon,
        "region": region,
    }
    identity = refine_gaussian_field(
        state, refiner, learner.reference, exact_zero_identity=True, **kwargs
    )
    assert torch.equal(identity.field.scales, state.base.gaussians.scales)
    assert torch.equal(identity.field.colors, state.base.gaussians.colors)
    _randomise(refiner, scale=0.3)
    out = refine_gaussian_field(
        state, refiner, learner.reference, exact_zero_identity=True, **kwargs
    )
    # The analytic test renderer ignores colour, so the colour path is checked on the field itself.
    (out.left.image.mean() + out.right.image.mean() + out.field.colors.mean()).backward()
    for name, parameter in refiner.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert (
        refiner.body[0].weight.grad.abs().sum() > 0 and refiner.output.weight.grad.abs().sum() > 0
    )
    assert state.base.gaussians.scales.grad is None  # the frozen base is not modified


def test_refined_inference_skips_the_base_renders_and_is_bit_identical(setup):  # noqa: F811
    learner, state = setup
    model = learner.reference
    refiner = FieldRefiner(
        FieldRefinerConfig(hidden_dims=(8, 8), bounds=model.config.residual_bounds)
    )
    _randomise(refiner, scale=0.3)
    eps = learner.actions.gaussian_bounds.alpha_logit_epsilon
    expected = refine_gaussian_field(state, refiner, model, composition_alpha_epsilon=eps)
    original = model.renderer
    calls: list[str] = []
    real_render = original.render

    def counting_render(request):
        calls.append(request.view)
        return real_render(request)

    original.render = counting_render  # type: ignore[method-assign]
    try:
        deployed = RefinedInference(model, refiner, composition_alpha_epsilon=eps)
        assert getattr(model.renderer, "identity", None) == getattr(original, "identity", None)
        actual = deployed(
            state.batch,
            target=state.base.target,
            sample_ids=("dataset_1/keyframe_1/frame_data000000",),
        )
        # only the two refined renders rasterise; the two base renders were skipped
        assert calls == ["left", "right"] and deployed.skipped_base_renders == 2
        for name in ("rotations", "scales", "opacities", "colors", "means3d"):
            assert torch.equal(getattr(actual.field, name), getattr(expected.field, name)), name
        assert torch.equal(actual.left.image, expected.left.image)
        assert torch.equal(actual.right.image, expected.right.image)
        assert torch.equal(actual.disparity, expected.disparity)
        # the support covariance nobody reads is not composed in the deployed path
        assert expected.field.cov_surface is not None and actual.field.cov_surface is None
        deployed.close()
        assert model.renderer is original
    finally:
        original.render = real_render  # type: ignore[method-assign]


def test_refined_inference_resets_the_skip_flag_when_the_forward_fails(setup):  # noqa: F811
    learner, state = setup
    model = learner.reference
    refiner = FieldRefiner(FieldRefinerConfig(hidden_dims=(8, 8)))
    deployed = RefinedInference(
        model,
        refiner,
        composition_alpha_epsilon=learner.actions.gaussian_bounds.alpha_logit_epsilon,
    )
    try:
        with pytest.raises(Exception):  # noqa: B017  (any validation error from the model forward)
            deployed(state.batch, target=state.base.target, sample_ids=("not/a/sample",))
        assert deployed._skipper.skip is False
    finally:
        deployed.close()


def test_skip_unread_surface_covariance_scope_is_exact_and_resets(setup):  # noqa: F811
    from reliable_endo_gs.gaussians import compose_gaussian_field
    from reliable_endo_gs.gaussians.residual_head import AttributeResiduals
    from reliable_endo_gs.utils.inference_mode import (
        skip_unread_surface_covariance,
        surface_covariance_skipped,
    )

    learner, state = setup
    prior = state.base.prior
    zeros = torch.zeros_like(prior.mu)
    residuals = AttributeResiduals(zeros, zeros, torch.zeros_like(prior.mu[..., :1]))
    full = compose_gaussian_field(prior, residuals, bounds=learner.reference.config.residual_bounds)
    assert not surface_covariance_skipped()
    with pytest.raises(RuntimeError):
        with skip_unread_surface_covariance():
            assert surface_covariance_skipped()
            lean = compose_gaussian_field(
                prior, residuals, bounds=learner.reference.config.residual_bounds
            )
            raise RuntimeError("scope must reset after an exception")
    assert not surface_covariance_skipped()
    assert full.cov_surface is not None and lean.cov_surface is None
    for name in ("rotations", "scales", "opacities", "colors", "means3d", "valid_mask"):
        assert torch.equal(getattr(lean, name), getattr(full, name)), name
