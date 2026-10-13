"""Frozen metric selector for the real stereo/Gaussian mask actions.

Uses the existing canonical metric outcome predictor and raw-unit guards.
The exploratory correspondence candidate remains an offline comparison and
cannot be dispatched by this two-repair deployment policy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch
from torch import nn

from reliable_endo_gs.actions.repair import Action
from reliable_endo_gs.routing.outcome_predictor import OutcomePredictor
from reliable_endo_gs.routing.selector import Decision
from reliable_endo_gs.routing.tradeoff import TradeoffConfig, select_tradeoff


def module_state_sha256(module: nn.Module) -> str:
    """Hash named parameters/buffers once when binding a frozen artifact."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class MetricRepairSelector:
    """Pre-action [R,F] -> one STOP/STEREO/GAUSSIAN decision, without GT."""

    def __init__(
        self,
        predictor: OutcomePredictor,
        config: TradeoffConfig,
        action_identity: Mapping[str, object],
        candidate_hash: str,
    ) -> None:
        if not config.schema.endswith(".v2") or not predictor.component_risk:
            raise ValueError("metric mask selection requires calibrated v2 physical risk")
        if not bool(predictor.fitted):
            raise ValueError("metric selector needs TRAIN-fitted normalization")
        self.predictor = predictor.eval().requires_grad_(False)
        self.config = config
        self.action_identity = dict(action_identity)
        self.candidate_hash = candidate_hash

    @torch.no_grad()
    def select(
        self,
        features: torch.Tensor,
        costs_ms: torch.Tensor,
        valid: torch.Tensor,
        remaining_ms: float,
    ) -> Decision:
        """Costs/valid [R,3] are in STOP/STEREO/GAUSSIAN enum order.

        Risk is the joint physical guard event. Unsupported calibration
        produces risk=1 in OutcomePredictor and therefore abstains.
        """
        if costs_ms.shape != (features.shape[0], 3) or valid.shape != costs_ms.shape:
            raise ValueError("metric repair costs/validity require [R,3]")
        changes, risks = self.predictor.predict(features)
        # Predictor order is correspondence/learned_stereo/gaussian. Explicitly
        # disable correspondence instead of manufacturing STOP training labels.
        candidate_valid = valid.clone()
        candidate_valid[:, 0] = False
        decision = select_tradeoff(
            changes.detach().cpu().tolist(),
            risks.detach().cpu().tolist(),
            costs_ms.detach().cpu().tolist(),
            candidate_valid.detach().cpu().tolist(),
            remaining_ms=remaining_ms,
            scales=self.predictor.metric_scale.tolist(),
            config=self.config,
        )
        action = {
            "STOP": Action.STOP,
            "learned_stereo": Action.STEREO,
            "gaussian": Action.GAUSSIAN,
        }[decision.action]
        return Decision(
            decision.region_index, action, decision.utility, decision.estimated_action_ms
        )
