"""Full-frame refinement training: loss decreases, frozen weights are bitwise unchanged, arms are separated."""

import pytest
import torch

from reliable_endo_gs.gaussians.refinement import FieldRefiner, FieldRefinerConfig
from reliable_endo_gs.training.field_refinement import (
    FieldRefinementTrainConfig,
    column_error_profile,
    state_digest,
    train_field_refinement,
)
from tests.training.test_student_joint import setup  # noqa: F401  (pytest fixture)


def _loader(sample):
    return lambda sample_id: (sample.batch, dict(sample.calibration_metadata))


def test_refiner_arm_reduces_loss_and_leaves_every_frozen_weight_unchanged(setup):  # noqa: F811
    learner, sample, _ = setup
    model = learner.model
    torch.manual_seed(1)
    refiner = FieldRefiner(
        FieldRefinerConfig(hidden_dims=(8, 8), bounds=model.config.residual_bounds)
    )
    before_model = state_digest(model)
    before_refiner = state_digest(refiner)
    rows: list[dict] = []
    summary = train_field_refinement(
        model,
        refiner,
        _loader(sample),
        [sample.sample_id],
        FieldRefinementTrainConfig(steps=40, learning_rate=3e-2, log_every=10),
        composition_alpha_epsilon=learner.actions.gaussian_bounds.alpha_logit_epsilon,
        record=rows.append,
    )
    assert summary["frozen_unchanged"] is True and summary["arm"] == "refiner"
    assert state_digest(model) == before_model  # student and head untouched
    assert state_digest(refiner) != before_refiner  # the refiner did learn
    assert rows[0]["step"] == 1 and rows[-1]["step"] == 40
    assert rows[-1]["loss_window_mean"] < rows[0]["loss"]
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in rows)


def test_head_arm_finetunes_only_the_attribute_head(setup):  # noqa: F811
    learner, sample, _ = setup
    model = learner.model
    stereo_before = state_digest(model.stereo_student)
    head_before = state_digest(model.head)
    summary = train_field_refinement(
        model,
        None,
        _loader(sample),
        [sample.sample_id],
        FieldRefinementTrainConfig(steps=10, arm="head", learning_rate=1e-2),
        composition_alpha_epsilon=learner.actions.gaussian_bounds.alpha_logit_epsilon,
    )
    assert summary["trainable_parameters"] == sum(p.numel() for p in model.head.parameters())
    assert state_digest(model.stereo_student) == stereo_before
    assert state_digest(model.head) != head_before


def test_arm_and_refiner_must_agree_and_configuration_is_validated(setup):  # noqa: F811
    learner, sample, _ = setup
    eps = learner.actions.gaussian_bounds.alpha_logit_epsilon
    with pytest.raises(ValueError, match="needs a FieldRefiner"):
        train_field_refinement(
            learner.model,
            None,
            _loader(sample),
            ["a"],
            FieldRefinementTrainConfig(steps=1),
            composition_alpha_epsilon=eps,
        )
    with pytest.raises(ValueError, match="takes no refiner"):
        train_field_refinement(
            learner.model,
            FieldRefiner(),
            _loader(sample),
            ["a"],
            FieldRefinementTrainConfig(steps=1, arm="head"),
            composition_alpha_epsilon=eps,
        )
    for kwargs in (
        {"steps": 0},
        {"steps": 1, "arm": "other"},
        {"steps": 1, "learning_rate": 2.0},
        {"steps": 1, "grad_clip": 0.0},
    ):
        with pytest.raises(ValueError):
            FieldRefinementTrainConfig(**kwargs)


def test_digest_detects_any_change_and_metric_helpers_are_consistent():
    module = torch.nn.Linear(3, 2)
    first = state_digest(module)
    assert state_digest(module) == first
    with torch.no_grad():
        module.weight[0, 0] += 1e-7
    assert state_digest(module) != first
    image = torch.rand(1, 3, 4, 64)
    profile = column_error_profile(image, torch.zeros_like(image), bins=8)
    assert len(profile) == 8 and all(v >= 0 for v in profile)
    with pytest.raises(ValueError, match="divisible"):
        column_error_profile(image, image, bins=7)
