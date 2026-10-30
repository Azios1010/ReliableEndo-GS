"""Action checkpoint tradeoffs preserve risk, ranking and historical identity."""

from dataclasses import replace

import pytest

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.config.selector_refit import SelectorRefitConfig
from reliable_endo_gs.training.selector_fitting import (
    joint_checkpoint_improves,
    validation_guard_components,
)


def config() -> InterventionConfig:
    return InterventionConfig(
        schema="intervention_selection.v2",
        selector_architecture="separate_action_heads",
        selector_balance_mode="per_action",
        local_auxiliary_weight=0.25,
        harm_weight=1.0,
    )


def components() -> dict[str, float]:
    return {
        "stereo_global": 0.21,
        "stereo_local": 0.50,
        "stereo_harm": 0.66,
        "gaussian_global": 0.22,
        "gaussian_local": 0.0,
        "gaussian_harm": 0.58,
        "ranking": 0.15,
    }


def test_grouped_guard_allows_gain_tradeoff_when_both_actions_and_risks_improve():
    cfg = config()
    old = components()
    improved = {
        **old,
        "stereo_global": 0.218,
        "stereo_local": 0.48,
        "stereo_harm": 0.56,
        "gaussian_global": 0.15,
        "gaussian_harm": 0.42,
        "ranking": 0.08,
    }
    default = SelectorRefitConfig()
    grouped = replace(default, validation_guard_mode="action_objectives")
    assert validation_guard_components(old, cfg, default) == old
    assert not joint_checkpoint_improves(improved, old, 0.81, 1.05, 0.0001)
    assert joint_checkpoint_improves(
        validation_guard_components(improved, cfg, grouped),
        validation_guard_components(old, cfg, grouped),
        0.81,
        1.05,
        0.0001,
    )


@pytest.mark.parametrize(
    "guard", ["stereo_objective", "gaussian_objective", "stereo_harm", "gaussian_harm", "ranking"]
)
def test_grouped_guard_cannot_hide_action_risk_or_ranking_regression(guard: str):
    policy = SelectorRefitConfig(validation_guard_mode="action_objectives")
    floor = validation_guard_components(components(), config(), policy)
    candidate = {key: value * 0.8 for key, value in floor.items()}
    candidate[guard] = floor[guard] + 0.01
    assert not joint_checkpoint_improves(candidate, floor, 0.5, 1.0, 0.0001)


def test_grouped_mode_rejects_shared_balance_and_unknown_policy():
    grouped = SelectorRefitConfig(validation_guard_mode="action_objectives")
    with pytest.raises(ValueError, match="per-action"):
        validation_guard_components(
            components(), replace(config(), selector_balance_mode="shared"), grouped
        )
    with pytest.raises(ValueError, match="guard mode"):
        SelectorRefitConfig(validation_guard_mode="unknown")


def test_default_guard_keeps_historical_config_hash_and_opt_in_changes_identity():
    default = SelectorRefitConfig()
    assert default.config_hash == "059d44399fa1e9672c1e3d3fd6bc2f5d650d2d7328dc781224ad92868e9b719b"
    assert (
        replace(default, validation_guard_mode="action_objectives").config_hash
        != default.config_hash
    )
