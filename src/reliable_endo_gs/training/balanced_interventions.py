"""Same local registered actions, full component supervision on one snapshot."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from reliable_endo_gs.actions.repair import Action, RepairActions, RepairState
from reliable_endo_gs.reconstruction.model import ReferenceReconstructionModel
from reliable_endo_gs.regions.grid import Region
from reliable_endo_gs.training.metric_quality import (
    OfflineMetricTruth,
    metric_action_policy,
    metric_risk_targets,
    metric_targets,
)
from reliable_endo_gs.training.quality_components import component_targets, measure_quality
from reliable_endo_gs.training.repairability import RepairabilityHarness, SelectionIntervention


@dataclass(frozen=True, slots=True)
class BalancedIntervention(SelectionIntervention):
    local_gain: float
    components_before: dict[str, float]
    components_after: dict[str, float]
    metric_changes: list[float] | None = None
    metric_observed: bool = True
    metric_diagnostics: dict[str, Any] | None = None
    risk_local: bool | None = None
    risk_global: bool | None = None


class BalancedHarness(RepairabilityHarness):
    def collect(
        self,
        state: RepairState,
        region: Region,
        actions: RepairActions,
        reference: ReferenceReconstructionModel,
        *,
        metric_truth: OfflineMetricTruth | None = None,
    ) -> tuple[BalancedIntervention, ...]:
        if any(p.requires_grad for p in actions.parameters()):
            raise ValueError("disable action gradients while collecting labels")
        with torch.no_grad():
            base = actions.execute(state, region, Action.STOP, reference)
            before = measure_quality(state, base, region).values()
            physical = self.config.metric_quality is not None
            metric_before = None
            if physical:
                if metric_truth is None:
                    raise ValueError("metric joint labels require separate offline TRAIN truth")
                from reliable_endo_gs.evaluation.mask_metrics import observation_metrics

                metric_before = observation_metrics(
                    base, state, metric_truth.disparity, metric_truth.valid_mask
                )
            rows = []
            for action in (Action.STOP, Action.STEREO, Action.GAUSSIAN):
                if action == Action.STOP:
                    after = before
                    elapsed = 0.0
                else:
                    device = state.batch.left.device
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    outcome = actions.execute(state, region, action, reference)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = (time.perf_counter() - started) * 1000
                    after = measure_quality(state, outcome, region).values()
                gain, local, harm = component_targets(before, after, self.config)
                changes = None
                observed = True
                metric_diagnostics = None
                if physical and action != Action.STOP:
                    from reliable_endo_gs.evaluation.mask_metrics import (
                        intervention_metrics,
                        observation_metrics,
                    )

                    assert metric_truth is not None and metric_before is not None
                    measured = observation_metrics(
                        outcome, state, metric_truth.disparity, metric_truth.valid_mask
                    )
                    label = intervention_metrics(
                        outcome,
                        base,
                        state,
                        region,
                        metric_truth.disparity,
                        metric_truth.valid_mask,
                        metric_action_policy(self.config, int(action)),
                        metric_before,
                        measured,
                    )
                    changes = label["metric_changes"]
                    metric_diagnostics = {
                        key: label[key] for key in ("metric_support", "metric_unavailable_reasons")
                    }
                    observed = changes is not None
                    # Unknown outcomes carry no selector label. The training
                    # caller excludes their entire region from supervision.
                    gain, local, harm = (
                        metric_targets(changes, self.config, int(action))
                        if changes is not None
                        else (0.0, 0.0, False)
                    )
                events = (
                    (False, False, False)
                    if action == Action.STOP
                    else metric_risk_targets(changes, self.config, int(action))
                    if physical and changes is not None
                    else None
                )
                rows.append(
                    BalancedIntervention(
                        action,
                        before["global_rgb"],
                        after["global_rgb"],
                        before["local_rgb"],
                        after["local_rgb"],
                        elapsed,
                        gain,
                        before["global_warp"],
                        after["global_warp"],
                        before["outside_rgb"],
                        after["outside_rgb"],
                        harm,
                        local,
                        before,
                        after,
                        changes,
                        observed,
                        metric_diagnostics,
                        None if events is None else events[0],
                        None if events is None else events[1],
                    )
                )
        return tuple(rows)
