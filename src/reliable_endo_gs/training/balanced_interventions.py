"""Same local registered actions, full component supervision on one snapshot."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from reliable_endo_gs.actions.repair import Action, RepairActions, RepairState
from reliable_endo_gs.reconstruction.model import ReferenceReconstructionModel
from reliable_endo_gs.regions.grid import Region
from reliable_endo_gs.training.quality_components import component_targets, measure_quality
from reliable_endo_gs.training.repairability import RepairabilityHarness, SelectionIntervention


@dataclass(frozen=True, slots=True)
class BalancedIntervention(SelectionIntervention):
    local_gain: float
    components_before: dict[str, float]
    components_after: dict[str, float]


class BalancedHarness(RepairabilityHarness):
    def collect(
        self,
        state: RepairState,
        region: Region,
        actions: RepairActions,
        reference: ReferenceReconstructionModel,
    ) -> tuple[BalancedIntervention, ...]:
        if any(p.requires_grad for p in actions.parameters()):
            raise ValueError("disable action gradients while collecting labels")
        with torch.no_grad():
            base = actions.execute(state, region, Action.STOP, reference)
            before = measure_quality(state, base, region).values()
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
                    )
                )
        return tuple(rows)
