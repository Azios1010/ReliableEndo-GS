"""Our student (unrefined output) and student + refiner as model-agnostic predictors for the comparison runner.

The student is rebuilt exactly as the development and confirmatory evaluations do: the construction, routing and learning
configs come from the Stage A checkpoint (pinned by SHA-256, which also supplies the frozen action bounds); the student weights
are either those of that checkpoint or those of another base-only student checkpoint (strict, finite, same shapes). The
refined path is ``refine_gaussian_field``: the single implementation shared with training, evaluation and the deployed path
(the graph-captured deployment is bit-identical to it). No edge guard.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from reliable_endo_gs.actions.refinement import RefinedInference
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.evaluation.multi_model import Prediction
from reliable_endo_gs.gaussians.refinement import (
    CONTEXT_CHANNEL_NAMES,
    FieldRefiner,
    FieldRefinerConfig,
)
from reliable_endo_gs.reconstruction import ReferenceConfig
from reliable_endo_gs.reconstruction.deployment import use_parallel_group_norm
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.rendering.production import ProductionFieldRenderer
from reliable_endo_gs.routing.config import RoutingConfig
from reliable_endo_gs.training.student_checkpoint import (
    import_frozen_student_actions,
    read_student_checkpoint,
)
from reliable_endo_gs.training.student_config import StudentLearningConfig
from reliable_endo_gs.training.student_learning import StudentLearner


def build_student(
    stage_a_checkpoint: Path,
    stage_a_sha256: str,
    device: torch.device,
    *,
    weights_from: Path | None = None,
    weights_from_sha256: str | None = None,
) -> tuple[StudentReconstructionModel, float]:
    """Frozen eval-mode student and the composition alpha epsilon. ``weights_from`` replaces the student weights."""
    raw = read_student_checkpoint(stage_a_checkpoint)
    construction = ReferenceConfig.from_dict(raw["construction_config"])
    learning = StudentLearningConfig(**raw["student_config"])
    routing = RoutingConfig(**raw["routing_config"])
    model = StudentReconstructionModel(
        renderer=ProductionFieldRenderer(),
        config=construction,
        width=learning.width,
        bins=learning.bins,
        minimum_effective_px=learning.minimum_effective_px,
        stereo_architecture=learning.stereo_architecture,
    ).to(device)
    learner = StudentLearner(
        model,
        learning,
        routing,
        intervention_config=InterventionConfig(**raw["intervention_config"]),
    )
    import_frozen_student_actions(stage_a_checkpoint, learner, expected_sha256=stage_a_sha256)
    epsilon = learner.actions.gaussian_bounds.alpha_logit_epsilon
    if weights_from is not None:
        if hashlib.sha256(weights_from.read_bytes()).hexdigest() != weights_from_sha256:
            raise ValueError("student weights file hash differs from the frozen one")
        other = read_student_checkpoint(weights_from)
        if other.get("intervention_config") is not None or other.get("mask_config") is not None:
            raise ValueError("expected a base-only student checkpoint")
        if (
            StudentLearningConfig(**other["student_config"]).stereo_architecture
            != learning.stereo_architecture
        ):
            raise ValueError("student architecture differs from the Stage A student")
        expected = model.state_dict()
        state = other["model"]
        if set(state) != set(expected):
            raise ValueError("checkpoint parameter/buffer keys differ")
        for key, tensor in state.items():
            if tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype:
                raise ValueError("checkpoint shape/dtype mismatch: " + key)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError("non-finite checkpoint tensor: " + key)
        model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model, float(epsilon)


def load_refiner(
    directory: Path, model: StudentReconstructionModel, device: torch.device
) -> FieldRefiner:
    """Rebuild the refiner saved as ``refiner.pt`` + ``spec.json`` (bounds scaled as in the ladder)."""
    blob = torch.load(directory / "refiner.pt", map_location=device)
    spec = json.loads((directory / "spec.json").read_text())
    bounds = model.config.residual_bounds
    scale = spec.get("bounds_scale", 1.0)
    bounds = replace(
        bounds,
        max_rotation_rad=bounds.max_rotation_rad * scale,
        max_log_scale=bounds.max_log_scale * scale,
        max_alpha_logit=bounds.max_alpha_logit * scale,
    )
    config = FieldRefinerConfig(
        hidden_dims=tuple(blob["config"]["hidden_dims"]),
        input_channels=tuple(
            CONTEXT_CHANNEL_NAMES.index(n) for n in blob["config"]["input_channels"]
        ),
        color_residual=blob["config"]["color_residual"],
        max_color_delta=blob["config"]["max_color_delta"],
        bounds=bounds,
    )
    refiner = FieldRefiner(config).to(device)
    refiner.load_state_dict(blob["state_dict"], strict=True)
    return refiner.eval().requires_grad_(False)


class StudentPredictor(nn.Module):
    """Unrefined output (``refiner=None``) or student + refiner, on the deployed configuration of the confirmatory run.

    The deployed configuration is a parallel group norm in the stereo network, a static grid layout and one CUDA graph from the
    stereo network to the refined field (``RefinedInference``, bit-identical to the eager ``refine_gaussian_field`` path on every
    valid slot). Using it makes the render numbers of this table identical to those of the held-out refinement study. The graph is
    captured on the first frame. The source-pixel index of the Gaussians is not returned by the deployed path, so Gaussian-centre
    cloud metrics are unavailable for these rows; the headline metrics do not use them.
    """

    def __init__(
        self, model: StudentReconstructionModel, refiner: FieldRefiner | None, epsilon: float
    ) -> None:
        super().__init__()
        use_parallel_group_norm(model.stereo_student)
        self.model = model
        self.runner = RefinedInference(
            model, refiner, composition_alpha_epsilon=epsilon, static_layout=True
        )
        self._captured = False

    def forward(self, sample: ScaredA00Sample) -> Prediction:
        metadata = dict(sample.calibration_metadata)
        if not self._captured:
            self.runner.capture([(sample.batch, metadata)])
            self._captured = True
        result = self.runner(sample.batch, calibration_metadata=metadata)
        return Prediction(result.disparity, result.left, result.right, result.field, None)
