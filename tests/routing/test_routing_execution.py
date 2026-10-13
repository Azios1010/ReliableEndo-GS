"""Execution optimizations preserve utility, risk, masks and STOP semantics."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from reliable_endo_gs.actions.repair import RepairState, primitive_core
from reliable_endo_gs.geometry.rectified import reconstruct_rectified
from reliable_endo_gs.regions.masks import MaskRegion
from reliable_endo_gs.routing.calibrated_selector import CalibratedSelector
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES, balanced_features
from reliable_endo_gs.routing.repairability import select_repairability
from tests.routing.test_balanced import setup as setup_fixture
from tests.routing.test_selector_refit import fitted_head, policy, table

setup = setup_fixture


def test_calibrated_prediction_reuses_encoder_and_is_identical():
    head = fitted_head()
    model = CalibratedSelector(head, policy(), {}, "mask")
    x = table((1, 2)).features
    with torch.no_grad():
        head.gain_network[-1].weight.fill_(0.1)
        head.harm_network[-1].weight.fill_(0.2)
        gain, _ = head.predict(x)
        logits = head.harm_network(head.normalization(x))
        expected = (
            torch.cat((gain[:, :1], gain[:, 1:] * model.gain_slope + model.gain_bias), 1),
            torch.cat(
                (gain[:, :1], (logits / model.risk_temperature + model.risk_bias).sigmoid()), 1
            ),
        )
    counts = {"normalization": 0, "gain": 0, "risk": 0}
    hooks = []
    for key, module in (
        ("normalization", head.normalization),
        ("gain", head.gain_network),
        ("risk", head.harm_network),
    ):

        def called(_module, _inputs, _output, key=key):
            counts[key] += 1

        hooks.append(module.register_forward_hook(called))
    try:
        measured = model.predict(x)
        assert counts == {"normalization": 1, "gain": 1, "risk": 1}
        for a, b in zip(measured, expected, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        counts.update(normalization=0, gain=0, risk=0)
        data = table((1, 2))
        with torch.no_grad():
            losses, _ = head.objective(data.features, data.global_gain, data.local_gain, data.harms)
        assert counts == {"normalization": 1, "gain": 1, "risk": 1}
        torch.testing.assert_close(
            losses["harm"], F.binary_cross_entropy_with_logits(logits, data.harms[:, 1:])
        )
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize("rows", [0, 1, 4])
def test_allocator_matches_independent_scalar_reference(rows):
    cfg = policy()
    rng = torch.Generator().manual_seed(17)
    for _ in range(30):
        gains = torch.randn(rows, 3, generator=rng)
        gains[:, 0] = 0
        harms = torch.rand(rows, 3, generator=rng)
        costs = torch.rand(rows, 3, generator=rng) * 200
        costs[:, 0] = 0
        valid = torch.rand(rows, 3, generator=rng) > 0.2
        risk_available = torch.rand(rows, 3, generator=rng) > 0.2
        candidates = []
        for i in range(rows):
            for a, cutoff in (
                (1, cfg.maximum_stereo_harm_probability),
                (2, cfg.maximum_gaussian_harm_probability),
            ):
                cutoff = cfg.maximum_harm_probability if cutoff is None else cutoff
                if (
                    valid[i, a]
                    and risk_available[i, a]
                    and 0 < float(costs[i, a]) <= 100
                    and float(harms[i, a]) <= cutoff
                    and float(gains[i, a]) > cfg.minimum_gain
                ):
                    candidates.append((float(gains[i, a]), i, a, float(costs[i, a])))
        decision = select_repairability(
            gains, harms, costs, valid, 100, cfg, risk_available=risk_available
        )
        if not candidates:
            assert decision.region_index is None and int(decision.action) == 0
        else:
            gain, i, a, cost = max(candidates, key=lambda row: row[0])
            assert (
                decision.region_index,
                int(decision.action),
                decision.predicted_utility,
                decision.estimated_ms,
            ) == (i, a, gain, cost)
    if rows:
        gains[0, 1] = float("nan")
        with pytest.raises(ValueError, match="invalid gain"):
            select_repairability(gains, harms, costs, valid, 100, cfg)


def test_fixed_shape_context_matches_independent_selected_statistics(setup):
    learner, sample, _ = setup
    with torch.no_grad():
        state = learner.state(sample)
        regions = (
            MaskRegion.from_indices(0, (5, 5), (6, 7, 11, 12), 1),
            MaskRegion.from_indices(1, (5, 5), (18, 19, 23, 24), 1),
        )
        values = balanced_features(
            state,
            regions,
            learner.intervention_config,
            scale_unit_m=learner.model.config.context_scale_m,
        )
        geometry = reconstruct_rectified(
            state.base.target.disparity,
            state.batch.camera,
            state.support,
            min_effective_disparity=1e-6,
        )
        for i, r in enumerate(regions):
            depth = geometry.depth.depth[
                r.mask(state.batch.left) & geometry.depth.valid_mask & state.support
            ]
            slots = primitive_core(state, r)[0]
            scales = state.base.gaussians.scales[0, slots]
            expected = {
                "depth_mean_m": depth.float().mean(),
                "depth_std_m": depth.float().std(unbiased=False),
                "global_disparity_std": state.base.target.disparity[state.support]
                .float()
                .std(unbiased=False),
            }
            expected.update(
                {
                    f"log_scale_{axis}": scales.clamp_min(1e-8).log().mean(0)[j]
                    for j, axis in enumerate("xyz")
                }
            )
            for name, value in expected.items():
                torch.testing.assert_close(
                    values[i, BALANCED_FEATURE_NAMES.index(name)], value, atol=2e-6, rtol=2e-5
                )
        empty = RepairState(
            replace(state.batch, support=torch.zeros_like(state.batch.support)), state.base
        )
        with pytest.raises(ValueError, match="invalid balanced context"):
            balanced_features(
                empty,
                regions,
                learner.intervention_config,
                scale_unit_m=learner.model.config.context_scale_m,
            )
