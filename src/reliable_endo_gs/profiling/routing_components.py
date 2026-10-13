"""Paired routing-component timings on a GT-free state and immutable references."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.actions.repair import eligible
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.calibrated_selector import CalibratedSelector
from reliable_endo_gs.routing.context import AVAILABILITY_COLUMNS, balanced_features
from reliable_endo_gs.routing.cost import capture_runtime_identity
from reliable_endo_gs.routing.metric_selection import module_state_sha256
from reliable_endo_gs.training.harness import capture_rng, restore_rng
from reliable_endo_gs.training.student_learning import StudentLearner, evaluation_mode


def _reference(path: Path) -> Any:
    """Load explicitly supplied local historical code, record its byte identity."""
    spec = importlib.util.spec_from_file_location("routing_benchmark_reference", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load archived routing reference")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile_routing_components(
    learner: StudentLearner, sample: ScaredA00Sample, reference_dir: Path, selector_checkpoint: Path
) -> dict[str, Any]:
    """Preserve RNG while profiling one GT-free native deployment sample."""
    device = next(learner.model.parameters()).device
    rng = capture_rng(device)
    try:
        return _profile_components(learner, sample, reference_dir, selector_checkpoint)
    finally:
        restore_rng(rng, device)


def _profile_components(
    learner: StudentLearner, sample: ScaredA00Sample, reference_dir: Path, selector_checkpoint: Path
) -> dict[str, Any]:
    """One native frame, three warmups and ten paired repeats; no model updates.

    Wall times include host stalls and CUDA synchronization. CUDA event times
    are reported separately. This is not full-pipeline FPS, training throughput,
    or a cost calibration used by the allocator.
    """
    if sample.supervision is not None or learner.intervention_config is None:
        raise ValueError("routing profiling requires GT-free physical deployment input")
    feature_config = learner.intervention_config
    if learner.mask_config is not None and learner.mask_config.feature_evidence != "combined":
        raise ValueError("this archived paired profile requires combined source features")
    device = next(learner.model.parameters()).device
    if device.type != "cuda":
        raise ValueError("CUDA component measurements require a CUDA state")
    paths = [reference_dir / "context.py", reference_dir / "calibrated_selector.py"]
    context_reference, head_reference = (_reference(p) for p in paths)
    raw = torch.load(selector_checkpoint, map_location="cpu", weights_only=True)
    if raw["action_identity"] != learner.frozen_action_identity():
        raise ValueError("profiling selector must match the frozen model/repairs")
    from reliable_endo_gs.config.intervention import InterventionConfig

    cfg = InterventionConfig(**raw["decision_config"])
    head = BalancedSelector(raw["width"], InterventionConfig(**raw["head_config"]))
    current = CalibratedSelector(head, cfg, raw["action_identity"], raw["target_mask_index_sha256"])
    current.load_state_dict(raw["state"])
    current.validate_state()
    current = current.eval().requires_grad_(False).to(device)
    old = head_reference.CalibratedSelector(
        copy.deepcopy(current.head), cfg, current.action_identity, current.mask_identity
    ).to(device)
    old.load_state_dict(current.state_dict())
    before = {"actions": learner.frozen_action_identity(), "selector": module_state_sha256(current)}
    with evaluation_mode(learner.model, learner.actions, learner.selector, current, old):
        state = learner.state(sample)
        regions = tuple(
            r for r in learner.candidate_regions(state, sample.sample_id) if eligible(state, r)
        )
        if not regions:
            raise ValueError("profile frame has no eligible regions")
        scale = learner.model.config.context_scale_m

        # Use the feature owner directly; the archived experiment uses combined
        # source features. No GT is loaded to choose or alter these masks.
        def old_features() -> torch.Tensor:
            result = context_reference.balanced_features(
                state, regions, feature_config, scale_unit_m=scale
            )
            if not isinstance(result, torch.Tensor):
                raise TypeError("archived context must return a tensor")
            return result

        def new_features() -> torch.Tensor:
            return balanced_features(state, regions, feature_config, scale_unit_m=scale)

        x0, x1 = old_features(), new_features()
        torch.testing.assert_close(x1, x0, atol=2e-5, rtol=2e-5)
        assert torch.equal(x1[:, AVAILABILITY_COLUMNS], x0[:, AVAILABILITY_COLUMNS])
        p0, p1 = old.predict(x0), current.predict(x0)
        for a, b in zip(p0, p1, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        rows: dict[str, Any] = {}

        def measure_pair(
            name: str, baseline: Callable[[], Any], optimized: Callable[[], Any]
        ) -> None:
            for _ in range(3):
                baseline()
                optimized()
            observations: dict[str, list[dict[str, float]]] = {"before": [], "after": []}
            for repeat in range(10):
                ordered: tuple[tuple[str, Callable[[], Any]], ...] = (
                    ("before", baseline),
                    ("after", optimized),
                )
                if repeat % 2:
                    ordered = tuple(reversed(ordered))
                for label, operation in ordered:
                    torch.cuda.synchronize(device)
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    wall = time.perf_counter()
                    start.record()
                    operation()
                    end.record()
                    torch.cuda.synchronize(device)
                    observations[label].append(
                        {
                            "wall_ms": (time.perf_counter() - wall) * 1000,
                            "cuda_event_ms": start.elapsed_time(end),
                        }
                    )
            rows[name] = {
                label: {
                    "raw": values,
                    "median_wall_ms": statistics.median(v["wall_ms"] for v in values),
                }
                for label, values in observations.items()
            }

        measure_pair("context", old_features, new_features)
        measure_pair("calibrated_head", lambda: old.predict(x0), lambda: current.predict(x0))
    if before != {
        "actions": learner.frozen_action_identity(),
        "selector": module_state_sha256(current),
    }:
        raise ValueError("profiling mutated scientific model state")
    return {
        "schema": "routing_component_profile.v1",
        "sample_id": sample.sample_id,
        "runtime": capture_runtime_identity(device),
        "regions": len(regions),
        "measurements": rows,
        "reference_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "selector_checkpoint_sha256": hashlib.sha256(selector_checkpoint.read_bytes()).hexdigest(),
        "feature_max_abs_difference": float((x1 - x0).abs().max()),
        "prediction_bitwise_equal": True,
        "weights_unchanged": True,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "ground_truth_loaded": False,
        "teacher_inference_executed": False,
        "training_executed": False,
        "full_pipeline_FPS_measured": False,
    }
