"""Scientific Phase II harness: real interventions, fixed-mask quality and cost.

This is distinct from the operational checkpoint/log harness. All alternatives
start from the same frozen A04 state, execute the same action implementation,
and are judged on the same pre-action masks. Outcomes are detached labels;
they are never concatenated to deployment selector features.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite

import torch

from reliable_endo_gs.actions.repair import (
    Action,
    ActionResult,
    RepairActions,
    RepairState,
    action_eligible,
)
from reliable_endo_gs.reconstruction.model import ReferenceReconstructionModel
from reliable_endo_gs.regions.grid import Region
from reliable_endo_gs.rendering.cross_view import RobustLossConfig, cross_view_loss


def reconstruction_loss(
    state: RepairState, outcome: ActionResult, region: Region | None = None
) -> torch.Tensor:
    """Mean of left/right Charbonnier RGB losses on fixed deployment masks.

    Phase II explicitly supervises BOTH views. A04's old left-only training is
    unchanged. Right support is the sensor mask, not left-disparity GT validity.
    This is a photometric objective, not an assertion of metric GT improvement.
    """
    core = torch.ones_like(state.support) if region is None else region.mask(state.batch.left)
    losses = []
    for render, observation, mask in (
        (outcome.left, state.batch.left, state.support & core),
        (outcome.right, state.batch.right, state.batch.support & core),
    ):
        if not bool(torch.isfinite(render.image).all()):
            raise ValueError("nonfinite intervention render")
        result = cross_view_loss(
            render, observation.to(render.image.dtype), mask, RobustLossConfig(kind="charbonnier")
        )
        if result.loss is None:
            raise ValueError("empty observation mask")
        losses.append(result.loss)
    return torch.stack(losses).mean()


def utility(
    before_loss: float, after_loss: float, cost_ms: float, cost_weight_per_ms: float
) -> float:
    """Canonical U = (loss_before - loss_after) - lambda * measured_ms."""
    if not all(isfinite(x) for x in (before_loss, after_loss, cost_ms, cost_weight_per_ms)):
        raise ValueError("utility inputs must be finite")
    if min(cost_ms, cost_weight_per_ms) < 0:
        raise ValueError("cost and cost weight must be nonnegative")
    return before_loss - after_loss - cost_weight_per_ms * cost_ms


@dataclass(frozen=True, slots=True)
class Intervention:
    action: Action
    before_loss: float
    after_loss: float
    local_before_loss: float
    local_after_loss: float
    cost_ms: float | None
    utility: float
    action_available: bool = field(default=True, kw_only=True)
    unavailable_reason: str | None = field(default=None, kw_only=True)


def intervention_costs(
    rows: Sequence[Intervention], reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return timings and execution mask [3]; skipped timings are NaN, never zero.

    The mask describes execution independently of any offline GT availability.
    Only executed alternatives can be passed to the timing-history recorder.
    """
    if [row.action for row in rows] != [Action.STOP, Action.STEREO, Action.GAUSSIAN]:
        raise ValueError("intervention costs require fixed STOP/stereo/Gaussian order")
    if any(row.action_available != (row.cost_ms is not None) for row in rows):
        raise ValueError("intervention timing does not match action execution availability")
    return (
        reference.new_tensor(
            [float("nan") if row.cost_ms is None else row.cost_ms for row in rows]
        ),
        torch.tensor(
            [row.action_available for row in rows], device=reference.device, dtype=torch.bool
        ),
    )


class InterventionHarness:
    """Counterfactual action evaluation on one region with synchronized timing."""

    def __init__(self, cost_weight_per_ms: float) -> None:
        if not isfinite(cost_weight_per_ms) or cost_weight_per_ms < 0:
            raise ValueError("cost weight must be finite/nonnegative")
        self.cost_weight_per_ms = cost_weight_per_ms

    def collect(
        self,
        state: RepairState,
        region: Region,
        actions: RepairActions,
        reference: ReferenceReconstructionModel,
    ) -> tuple[Intervention, ...]:
        """Evaluate executable alternatives; retain unavailable fixed-table slots."""
        if any(parameter.requires_grad for parameter in actions.parameters()):
            raise ValueError("freeze repair heads before selector label collection")
        with torch.no_grad():
            stop = actions.execute(state, region, Action.STOP, reference)
            before = float(reconstruction_loss(state, stop))
            local_before = float(reconstruction_loss(state, stop, region))
            rows = [Intervention(Action.STOP, before, before, local_before, local_before, 0.0, 0.0)]
            for action in (Action.STEREO, Action.GAUSSIAN):
                if not action_eligible(state, region, action):
                    rows.append(
                        Intervention(
                            action,
                            before,
                            before,
                            local_before,
                            local_before,
                            None,
                            0.0,
                            action_available=False,
                            unavailable_reason="action_precondition",
                        )
                    )
                    continue
                device = state.batch.left.device
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                outcome = actions.execute(state, region, action, reference)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - started) * 1000
                after = float(reconstruction_loss(state, outcome))
                local_after = float(reconstruction_loss(state, outcome, region))
                rows.append(
                    Intervention(
                        action,
                        before,
                        after,
                        local_before,
                        local_after,
                        elapsed_ms,
                        utility(before, after, elapsed_ms, self.cost_weight_per_ms),
                    )
                )
        return tuple(rows)
