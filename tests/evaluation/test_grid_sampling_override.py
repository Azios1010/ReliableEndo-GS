from __future__ import annotations

from dataclasses import replace

import pytest

from reliable_endo_gs.evaluation.ours import with_grid_sampling
from reliable_endo_gs.gaussians.priors import GeometryPriorPolicy
from reliable_endo_gs.gaussians.sampling import SamplingPolicy
from reliable_endo_gs.reconstruction import ReferenceConfig


def _config(mode: str = "fixed") -> ReferenceConfig:
    sampling = SamplingPolicy(mode=mode, stride=2, cap=100_000, max_samples=100_000)
    return replace(ReferenceConfig(), prior_policy=GeometryPriorPolicy(sampling=sampling))


def test_none_returns_the_same_config() -> None:
    config = _config()
    assert with_grid_sampling(config, None, None) is config


def test_explicit_default_equals_none() -> None:
    config = _config()
    assert with_grid_sampling(config, 2, 100_000) == config
    assert with_grid_sampling(config, 2, 100_000).to_dict() == config.to_dict()


def test_override_changes_only_the_sampling_entry_and_does_not_mutate_the_input() -> None:
    config = _config()
    before = config.to_dict()
    changed = with_grid_sampling(config, 1, 327_680)
    assert config.to_dict() == before
    sampling = changed.prior_policy.sampling
    assert (sampling.stride, sampling.cap, sampling.max_samples, sampling.mode) == (
        1,
        327_680,
        327_680,
        "fixed",
    )
    other = changed.to_dict()
    other["prior_policy"]["sampling"] = before["prior_policy"]["sampling"]
    assert other == before


def test_stride_needs_a_cap_and_a_cap_needs_a_stride() -> None:
    with pytest.raises(ValueError):
        with_grid_sampling(_config(), 1, None)
    with pytest.raises(ValueError):
        with_grid_sampling(_config(), None, 1000)


def test_only_fixed_grids_can_change() -> None:
    with pytest.raises(ValueError):
        with_grid_sampling(_config("dense"), 2, 100_000)
