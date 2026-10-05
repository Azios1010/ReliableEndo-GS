"""Frozen-reference repair learning followed by supervised utility selection."""

from __future__ import annotations

import time
from dataclasses import asdict

import torch

from reliable_endo_gs.actions.repair import Action, RepairActions, RepairState, eligible
from reliable_endo_gs.reconstruction.model import ReferenceReconstructionModel
from reliable_endo_gs.regions.grid import Region, grid_regions
from reliable_endo_gs.routing.config import RoutingConfig
from reliable_endo_gs.routing.selector import ActionSelector, region_features, select_action
from reliable_endo_gs.training.interventions import (
    InterventionHarness,
    reconstruction_loss,
    utility,
)


class ActionLearner:
    """Own ONLY Phase II optimizers; reference/teacher remain in eval mode."""

    def __init__(self, reference: ReferenceReconstructionModel, config: RoutingConfig) -> None:
        self.reference = reference.eval().requires_grad_(False)
        self.config = config
        if config.halo < reference.config.prior_policy.surface.neighborhood_radius:
            raise ValueError("repair halo must cover the canonical surface neighbourhood")
        device = next(reference.head.parameters()).device
        self.actions = RepairActions(
            reference,
            width=config.width,
            max_disparity_px=config.max_disparity_px,
            composition_alpha_epsilon=config.composition_alpha_epsilon,
        ).to(device)
        self.selector = ActionSelector(config.width).to(device)
        self.action_optimizer = torch.optim.Adam(self.actions.parameters(), lr=config.learning_rate)
        self.selector_optimizer = torch.optim.Adam(
            self.selector.parameters(), lr=config.learning_rate
        )
        self.harness = InterventionHarness(config.cost_weight_per_ms)
        self.cost_max_ms = torch.zeros(3, dtype=torch.float32, device=device)
        self.cost_observations = 0

    def regions(self, state: RepairState) -> tuple[Region, ...]:
        """Fixed grid on the actual aligned deployment resolution."""
        return grid_regions(
            (int(state.batch.left.shape[-2]), int(state.batch.left.shape[-1])),
            self.config.tile_hw,
            self.config.halo,
        )

    def warmup_step(self, state: RepairState) -> dict[str, object]:
        """Train both real repairs from the same whole-frame state, not STOP."""
        self.actions.train().requires_grad_(True)
        self.reference.eval().requires_grad_(False)
        height, width = state.batch.left.shape[-2:]
        full = grid_regions((height, width), (height, width), self.config.halo)[0]
        self.action_optimizer.zero_grad(set_to_none=True)
        losses: dict[str, float] = {}
        for action in (Action.STEREO, Action.GAUSSIAN):
            outcome = self.actions.execute(state, full, action, self.reference)
            loss = reconstruction_loss(state, outcome)
            if not bool(torch.isfinite(loss)) or not loss.requires_grad:
                raise ValueError("repair loss must be finite and differentiable")
            loss.backward()
            losses[action.name.lower()] = float(loss.detach())
        self._step(self.actions, self.action_optimizer)
        return {"repair_losses": losses}

    def check_zero_repairs(self, state: RepairState) -> dict[str, float]:
        """Require initial zero residuals to reproduce the frozen field/render.

        This diagnostic must run before learning; it catches precision or
        composition differences masquerading as useful repair supervision.
        """
        height, width = state.batch.left.shape[-2:]
        full = grid_regions((height, width), (height, width), self.config.halo)[0]
        errors: dict[str, float] = {}
        with torch.no_grad():
            for action in (Action.STEREO, Action.GAUSSIAN):
                result = self.actions.execute(state, full, action, self.reference)
                for name in ("means3d", "scales", "opacities", "rotations"):
                    torch.testing.assert_close(
                        getattr(result.field, name),
                        getattr(state.base.gaussians, name),
                        rtol=1e-5,
                        atol=1e-7,
                    )
                torch.testing.assert_close(
                    result.left.image, state.base.render_left.image, rtol=1e-5, atol=2e-6
                )
                if state.base.render_right is None:
                    raise ValueError("zero-repair check requires both renders")
                torch.testing.assert_close(
                    result.right.image, state.base.render_right.image, rtol=1e-5, atol=2e-6
                )
                errors[action.name.lower()] = float(
                    (result.left.image - state.base.render_left.image).abs().max()
                )
        return errors

    def freeze_actions(self) -> None:
        """Keep label-generating action weights stable during selector learning."""
        self.actions.eval().requires_grad_(False)
        self.actions.zero_grad(set_to_none=True)

    def selector_step(self, state: RepairState, sample_position: int) -> dict[str, object]:
        """Observe a rotating eligible core; train MLP on measured utility gains."""
        self.freeze_actions()
        self.selector.train()
        regions = self.regions(state)
        candidates = [i for i, region in enumerate(regions) if eligible(state, region)]
        if not candidates:
            raise ValueError("sample has no eligible intervention regions")
        index = candidates[sample_position % len(candidates)]
        region = regions[index]
        features = region_features(state, (region,))  # BEFORE actions/outcomes
        rows = self.harness.collect(state, region, self.actions, self.reference)
        labels = features.new_tensor([[row.utility for row in rows]])
        self.selector_optimizer.zero_grad(set_to_none=True)
        prediction = self.selector(features)
        # Only the non-STOP outputs are learned; STOP is the exact zero anchor.
        loss = torch.nn.functional.mse_loss(prediction[:, 1:], labels[:, 1:])
        loss.backward()
        self._step(self.selector, self.selector_optimizer)
        self.cost_max_ms = torch.maximum(
            self.cost_max_ms, features.new_tensor([row.cost_ms for row in rows])
        )
        self.cost_observations += 1
        return {
            "selector_loss": float(loss.detach()),
            "region": index,
            "features": features.detach().cpu().tolist(),
            "interventions": [asdict(row) for row in rows],
            "oracle_action": int(labels.argmax(dim=1).item()),
            "action_parameters_frozen": True,
        }

    @staticmethod
    def _step(module: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Fail before Adam on NaN/Inf; no silent overflow or partial update."""
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(grad).all()) for grad in gradients):
            raise ValueError("Phase II gradients are missing or nonfinite")
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in module.parameters()):
            raise ValueError("Phase II optimizer produced nonfinite parameters")

    def infer(self, state: RepairState) -> dict[str, object]:
        """Select and execute one action; no oracle enumeration at deployment."""
        if self.cost_observations == 0:
            raise ValueError("selector has no measured cost calibration")
        self.freeze_actions()
        self.selector.eval()
        with torch.no_grad():
            device = state.batch.left.device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            regions = self.regions(state)
            features = region_features(state, regions)
            utilities = self.selector(features)
            valid = torch.tensor(
                [eligible(state, region) for region in regions],
                dtype=torch.bool,
                device=features.device,
            )
            costs = self.cost_max_ms.expand(len(regions), -1)
            decision = select_action(utilities, costs, valid, self.config.budget_ms)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            selection_ms = (time.perf_counter() - started) * 1000
            region = regions[0 if decision.region_index is None else decision.region_index]
            started = time.perf_counter()
            outcome = self.actions.execute(state, region, decision.action, self.reference)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            repair_ms = (
                (time.perf_counter() - started) * 1000 if decision.action != Action.STOP else 0.0
            )
            before = self.actions.execute(state, region, Action.STOP, self.reference)
            before_loss = float(reconstruction_loss(state, before))
            after_loss = float(reconstruction_loss(state, outcome))
            return {
                "decision": asdict(decision),
                "base_loss": before_loss,
                "selected_loss": after_loss,
                "selection_ms": selection_ms,
                "repair_ms": repair_ms,
                "budget_overrun": repair_ms > self.config.budget_ms,
                "measured_utility": utility(
                    before_loss, after_loss, repair_ms, self.config.cost_weight_per_ms
                ),
                "features": features.cpu().tolist(),
            }
