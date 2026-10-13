"""Dedicated frozen TRAIN inference timing; training maxima stay diagnostic."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from functools import partial

import torch

from reliable_endo_gs.actions.repair import Action, eligible
from reliable_endo_gs.routing.cost import (
    CostCalibrationPolicy,
    DeploymentCostCalibration,
    calibrate_action_costs,
    capture_runtime_identity,
)
from reliable_endo_gs.training.harness import capture_rng, restore_rng
from reliable_endo_gs.training.student_learning import StudentLearner, evaluation_mode
from reliable_endo_gs.training.student_run import StudentSamples


def calibrate_student_costs(
    learner: StudentLearner,
    samples: StudentSamples,
    training_identity: Mapping[str, object],
    policy: CostCalibrationPolicy,
) -> DeploymentCostCalibration:
    """Measure complete shared action+render calls for every eligible TRAIN mask.

    The deployment proposal is fixed before each timed call. GT is not loaded.
    Warmup/repeats are explicit. Runtime, grid, worklist and exact action state
    bind the artifact; no development timings enter the estimator.
    """
    if not len(samples):
        raise ValueError("deployment calibration requires TRAIN samples")
    device = next(learner.model.parameters()).device
    runtime = capture_runtime_identity(device)
    identity = learner.frozen_action_identity()
    rng = capture_rng(device)
    timings = []
    shapes: set[tuple[int, int]] = set()
    ids = []
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
    try:
        with torch.no_grad(), evaluation_mode(learner.model, learner.actions, learner.selector):
            for index in range(len(samples)):
                sample = samples[index]
                if (
                    sample.supervision is not None
                    or sample.data_role != "final_train"
                    or sample.sample_id.split("/")[0] not in {f"dataset_{i}" for i in range(1, 8)}
                ):
                    raise ValueError("deployment cost calibration accepts GT-free TRAIN only")
                ids.append(sample.sample_id)
                state = learner.state(sample)
                shape = (int(state.batch.left.shape[-2]), int(state.batch.left.shape[-1]))
                shapes.add(shape)
                if len(shapes) != 1:
                    raise ValueError("deployment calibration requires one output grid")
                for region in learner.candidate_regions(state, sample.sample_id):
                    if not eligible(state, region):
                        continue
                    artifact = calibrate_action_costs(
                        {
                            int(action): partial(
                                learner.actions.execute, state, region, action, learner.model
                            )
                            for action in (Action.STOP, Action.STEREO, Action.GAUSSIAN)
                        },
                        binding={},
                        shape_hw=shape,
                        policy=policy,
                        runtime=runtime,
                        synchronize=sync,
                    )
                    timings.append(artifact.timings_ms)
                print(
                    json.dumps(
                        {
                            "event": "deployment_cost_progress",
                            "samples": index + 1,
                            "total": len(samples),
                            "regions": len(timings),
                        }
                    ),
                    flush=True,
                )
    finally:
        restore_rng(rng, device)
    if learner.frozen_action_identity() != identity:
        raise ValueError("deployment calibration mutated model/action state")
    if not timings:
        raise ValueError("no eligible TRAIN masks for deployment calibration")
    values = torch.cat(timings, dim=1)
    aggregate_policy = replace(policy, measure_repeats=int(values.shape[1]))
    if policy.tail_policy == "max":
        estimate = values.amax(1)
    else:
        assert policy.quantile is not None
        estimate = values.quantile(policy.quantile, dim=1)
    return DeploymentCostCalibration(
        {
            "collection_role": "train",
            "frozen_action_identity": identity,
            "training_identity": dict(training_identity),
            "training_manifest_hash": training_identity["manifest_hash"],
            "sample_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "target_mask_index_sha256": None
            if learner.target_masks is None
            else learner.target_masks.identity,
            "timing_scope": "shared_action_and_render",
            "regions_measured": len(timings),
        },
        aggregate_policy,
        runtime,
        next(iter(shapes)),
        values,
        estimate,
    )
