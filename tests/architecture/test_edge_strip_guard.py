"""Opt-in right-edge strip guard: the mask formula, the compositing, and that the default path is untouched."""

from dataclasses import replace

import pytest
import torch

from reliable_endo_gs.actions.refinement import (
    RefinedInference,
    guard_right_edge,
    refine_gaussian_field,
    right_edge_strip_mask,
)
from reliable_endo_gs.actions.repair import Action, ActionResult
from reliable_endo_gs.contracts.rendering import RenderOutput
from reliable_endo_gs.gaussians.refinement import FieldRefiner, FieldRefinerConfig
from tests.routing.test_frozen_a04 import setup  # noqa: F401  (pytest fixture)


def _disparity(rows: int = 100, width: int = 64, value: float = 10.0) -> torch.Tensor:
    return torch.full((1, 1, rows, width), value)


def test_mask_marks_the_last_d_hat_columns():
    mask = right_edge_strip_mask(_disparity(value=10.0))
    assert mask.shape == (1, 1, 1, 64)
    assert mask.flatten()[:54].sum() == 0 and mask.flatten()[54:].min() == 1


def test_mask_uses_the_largest_disparity_of_the_last_eight_columns_and_ignores_a_few_outlier_rows():
    disparity = _disparity(value=4.0)
    disparity[..., -3] = 12.0  # inside the last eight columns, every row
    assert int(right_edge_strip_mask(disparity).sum()) == 12
    outliers = _disparity(value=4.0)
    outliers[0, 0, :5, -1] = 60.0  # 5 of 100 rows are below the 90th percentile
    assert int(right_edge_strip_mask(outliers).sum()) == 4
    outside_window = _disparity(value=4.0)
    outside_window[..., 10] = 30.0  # not among the last eight columns
    assert int(right_edge_strip_mask(outside_window).sum()) == 4


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_edge_disparity_falls_back_to_the_unrefined_image(bad):
    disparity = _disparity(value=4.0)
    disparity[..., -1] = bad
    assert right_edge_strip_mask(disparity).min() == 1


@pytest.mark.parametrize("shape", [(2, 1, 8, 16), (1, 3, 8, 16), (8, 16)])
def test_only_a_single_frame_disparity_map_is_accepted(shape):
    with pytest.raises(ValueError, match="expects a"):
        right_edge_strip_mask(torch.ones(shape))


def test_a_strip_wider_than_the_image_is_the_whole_image():
    assert right_edge_strip_mask(_disparity(width=16, value=500.0)).min() == 1


def _result(refined_right: float, width: int = 64) -> ActionResult:
    image = torch.full((1, 3, 6, width), refined_right)
    render = RenderOutput(
        image, depth=torch.ones(1, 1, 6, width), visibility=torch.ones(1, 1, 6, width)
    )
    return ActionResult(
        Action.GAUSSIAN,
        None,
        _disparity(rows=6, width=width, value=10.0),
        render,
        render,
        ("scale",),  # type: ignore[arg-type]
    )


def test_guard_replaces_only_the_right_image_inside_the_strip():
    result = _result(refined_right=0.25)
    stop = torch.full((1, 3, 6, 64), 0.75)
    guarded = guard_right_edge(result, stop)
    assert torch.equal(guarded.right.image[..., :54], result.right.image[..., :54])
    assert torch.equal(guarded.right.image[..., 54:], stop[..., 54:])
    assert (
        guarded.left is result.left
        and guarded.field is result.field
        and guarded.disparity is result.disparity
    )
    assert (
        guarded.right.depth is result.right.depth
        and guarded.right.visibility is result.right.visibility
    )


def _with_disparity(state, value: float):
    target = replace(
        state.base.target, disparity=torch.full_like(state.base.target.disparity, value)
    )
    return replace(state, base=replace(state.base, target=target))


def _refiner(model) -> FieldRefiner:
    refiner = FieldRefiner(
        FieldRefinerConfig(hidden_dims=(8, 8), bounds=model.config.residual_bounds)
    )
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in refiner.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)
    return refiner


def test_refine_gaussian_field_default_is_unchanged_and_the_guard_uses_the_base_right_render(setup):  # noqa: F811
    learner, state = setup
    model = learner.reference
    state = _with_disparity(state, 2.0)  # a 2-column strip on the analytic test grid
    refiner = _refiner(model)
    eps = learner.actions.gaussian_bounds.alpha_logit_epsilon
    plain = refine_gaussian_field(state, refiner, model, composition_alpha_epsilon=eps)
    explicit_off = refine_gaussian_field(
        state, refiner, model, composition_alpha_epsilon=eps, edge_guard=False
    )
    guarded = refine_gaussian_field(
        state, refiner, model, composition_alpha_epsilon=eps, edge_guard=True
    )
    assert torch.equal(plain.right.image, explicit_off.right.image)
    strip = right_edge_strip_mask(state.base.target.disparity).flatten().bool()
    assert 0 < int(strip.sum()) < strip.numel()
    assert torch.equal(guarded.right.image[..., strip], state.base.render_right.image[..., strip])
    assert torch.equal(guarded.right.image[..., ~strip], plain.right.image[..., ~strip])
    assert torch.equal(guarded.left.image, plain.left.image)


def test_refined_inference_guard_matches_the_shared_action_and_costs_one_extra_right_render(
    setup,  # noqa: F811
    monkeypatch,
):
    learner, state = setup
    model = learner.reference
    refiner = _refiner(model)
    eps = learner.actions.gaussian_bounds.alpha_logit_epsilon
    # The fixture disparity (tens of pixels) exceeds its 5-column grid, so pin a partial strip (last 2 columns) for both paths.
    width = int(state.base.target.disparity.shape[-1])
    partial = torch.zeros(1, 1, 1, width)
    partial[..., -2:] = 1.0
    monkeypatch.setattr(
        "reliable_endo_gs.actions.refinement.right_edge_strip_mask", lambda disparity: partial
    )
    expected = refine_gaussian_field(
        state, refiner, model, composition_alpha_epsilon=eps, edge_guard=True
    )
    original = model.renderer
    calls: list[str] = []
    real_render = original.render

    def counting_render(request):
        calls.append(request.view)
        return real_render(request)

    original.render = counting_render  # type: ignore[method-assign]
    try:
        off = RefinedInference(model, refiner, composition_alpha_epsilon=eps)
        plain = off(
            state.batch,
            target=state.base.target,
            sample_ids=("dataset_1/keyframe_1/frame_data000000",),
        )
        assert calls == ["left", "right"]
        off.close()
        calls.clear()
        on = RefinedInference(model, refiner, composition_alpha_epsilon=eps, edge_guard=True)
        actual = on(
            state.batch,
            target=state.base.target,
            sample_ids=("dataset_1/keyframe_1/frame_data000000",),
        )
        assert calls == ["left", "right", "right"]
        on.close()
    finally:
        original.render = real_render  # type: ignore[method-assign]
    assert torch.equal(actual.right.image, expected.right.image)
    assert torch.equal(actual.left.image, expected.left.image)
    assert torch.equal(actual.right.image[..., :-2], plain.right.image[..., :-2])
    assert not torch.equal(
        actual.right.image[..., -2:], plain.right.image[..., -2:]
    )  # the guard changed the strip


def test_guard_is_inert_for_a_stop_instance(setup):  # noqa: F811
    learner, state = setup
    model = learner.reference
    eps = learner.actions.gaussian_bounds.alpha_logit_epsilon
    stop = RefinedInference(model, None, composition_alpha_epsilon=eps, edge_guard=True)
    reference = RefinedInference(model, None, composition_alpha_epsilon=eps)
    kwargs = {"target": state.base.target, "sample_ids": ("dataset_1/keyframe_1/frame_data000000",)}
    guarded_stop = stop(state.batch, **kwargs)
    plain_stop = reference(state.batch, **kwargs)
    reference.close()
    stop.close()
    assert torch.equal(guarded_stop.right.image, plain_stop.right.image)
