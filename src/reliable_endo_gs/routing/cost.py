"""Explicit inference-time action-cost calibration.

Training-time timings remain diagnostics. Allocation may consume only an
artifact bound to the exact training identity, action/model state, runtime,
configuration, and output shape.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

HISTORY_SIZE = 64
CALIBRATION_SCHEMA = "deployment_action_cost_calibration.v1"
TailPolicy = Literal["max", "quantile"]


@dataclass(frozen=True, slots=True)
class CostCalibrationPolicy:
    """Declared timing protocol; every value is recorded in the artifact."""

    tail_policy: TailPolicy
    quantile: float | None
    warmup_repeats: int
    measure_repeats: int

    def __post_init__(self) -> None:
        if self.tail_policy not in ("max", "quantile"):
            raise ValueError("tail_policy must be max or quantile")
        if self.tail_policy == "quantile" and (self.quantile is None or not 0 < self.quantile <= 1):
            raise ValueError("quantile policy requires a value in (0,1]")
        if self.tail_policy == "max" and self.quantile is not None:
            raise ValueError("max policy cannot carry a quantile")
        if type(self.warmup_repeats) is not int or self.warmup_repeats < 0:
            raise ValueError("warmup_repeats must be nonnegative")
        if type(self.measure_repeats) is not int or self.measure_repeats < 1:
            raise ValueError("measure_repeats must be positive")

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def capture_runtime_identity(device: torch.device | str | None = None) -> dict[str, object]:
    """Capture hardware, software, and CUDA launch settings affecting timing."""

    resolved = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    result: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(resolved),
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
    }
    if resolved.type == "cuda" and torch.cuda.is_available():
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        result.update(
            {
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "device_name": props.name,
                "device_index": index,
                "compute_capability": [props.major, props.minor],
            }
        )
    return result


@dataclass(frozen=True, slots=True)
class DeploymentCostCalibration:
    """Measured STOP/STEREO/GAUSSIAN timings and their binding provenance."""

    binding: Mapping[str, object]
    policy: CostCalibrationPolicy
    runtime: Mapping[str, object]
    shape_hw: tuple[int, int]
    timings_ms: torch.Tensor
    estimated_ms: torch.Tensor
    schema: str = CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CALIBRATION_SCHEMA:
            raise ValueError("unsupported deployment cost calibration schema")
        if self.timings_ms.shape != (3, self.policy.measure_repeats) or not bool(
            torch.isfinite(self.timings_ms).all() & (self.timings_ms >= 0).all()
        ):
            raise ValueError("invalid calibration timing matrix")
        if (
            not bool((self.timings_ms[0] == 0).all())
            or self.estimated_ms.shape != (3,)
            or not bool(torch.isfinite(self.estimated_ms).all() & (self.estimated_ms >= 0).all())
            or float(self.estimated_ms[0]) != 0
        ):
            raise ValueError("invalid calibration estimates")
        if len(self.shape_hw) != 2 or any(type(x) is not int or x < 1 for x in self.shape_hw):
            raise ValueError("shape_hw must be positive integers")

    @property
    def binding_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(dict(self.binding), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def matches(
        self,
        binding: Mapping[str, object],
        runtime: Mapping[str, object],
        shape_hw: tuple[int, int],
    ) -> bool:
        return (
            dict(self.binding) == dict(binding)
            and dict(self.runtime) == dict(runtime)
            and tuple(shape_hw) == self.shape_hw
        )

    def matches_training_identity(self, identity: Mapping[str, object]) -> bool:
        """Check the optional serialized training identity binding."""

        expected = self.binding.get("training_identity")
        return expected is None or expected == dict(identity)

    def to_payload(self) -> dict[str, object]:
        values = self.timings_ms.detach().cpu()
        distribution = {
            str(action): {
                "min_ms": float(values[action].min()),
                "median_ms": float(values[action].median()),
                "mean_ms": float(values[action].mean()),
                "max_ms": float(values[action].max()),
                "tail_ms": float(self.estimated_ms[action]),
            }
            for action in range(3)
        }
        return {
            "schema": self.schema,
            "binding": dict(self.binding),
            "policy": asdict(self.policy),
            "runtime": dict(self.runtime),
            "shape_hw": list(self.shape_hw),
            "timings_ms": values,
            "estimated_ms": self.estimated_ms.detach().cpu(),
            "distribution": distribution,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> DeploymentCostCalibration:
        policy_payload = payload.get("policy")
        binding_payload = payload.get("binding")
        runtime_payload = payload.get("runtime")
        if payload.get("schema") != CALIBRATION_SCHEMA or not isinstance(policy_payload, Mapping):
            raise ValueError("invalid deployment cost calibration payload")
        shape, timings, estimate = (
            payload.get("shape_hw"),
            payload.get("timings_ms"),
            payload.get("estimated_ms"),
        )
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 2
            or not isinstance(timings, torch.Tensor)
            or not isinstance(estimate, torch.Tensor)
        ):
            raise ValueError("invalid deployment calibration tensors")
        return cls(
            binding=binding_payload if isinstance(binding_payload, Mapping) else {},
            policy=CostCalibrationPolicy(**dict(policy_payload)),
            runtime=runtime_payload if isinstance(runtime_payload, Mapping) else {},
            shape_hw=(int(shape[0]), int(shape[1])),
            timings_ms=timings.float(),
            estimated_ms=estimate.float(),
        )


def _tail_estimate(timings_ms: torch.Tensor, policy: CostCalibrationPolicy) -> torch.Tensor:
    if policy.tail_policy == "max":
        estimate = timings_ms.amax(1)
    else:
        if policy.quantile is None:
            raise ValueError("quantile policy requires a quantile")
        estimate = torch.quantile(timings_ms, policy.quantile, dim=1)
    estimate = estimate.clone()
    estimate[0] = 0
    return estimate


def calibrate_action_costs(
    actions: Mapping[int, Callable[[], object]],
    *,
    binding: Mapping[str, object],
    shape_hw: tuple[int, int],
    policy: CostCalibrationPolicy,
    runtime: Mapping[str, object] | None = None,
    synchronize: Callable[[], None] | None = None,
) -> DeploymentCostCalibration:
    """Measure full-frame action callables using an explicit warmup/repeat protocol."""

    if set(actions) != {0, 1, 2}:
        raise ValueError("cost calibration needs action ids 0, 1, and 2")
    if (
        type(shape_hw) is not tuple
        or len(shape_hw) != 2
        or any(type(x) is not int or x < 1 for x in shape_hw)
    ):
        raise ValueError("shape_hw must be positive integer pair")
    values = torch.zeros((3, policy.measure_repeats), dtype=torch.float64)
    with torch.inference_mode():
        for action_id in (0, 1, 2):
            for _ in range(policy.warmup_repeats):
                actions[action_id]()
            for repeat in range(policy.measure_repeats):
                if synchronize is not None:
                    synchronize()
                started = time.perf_counter()
                actions[action_id]()
                if synchronize is not None:
                    synchronize()
                values[action_id, repeat] = (time.perf_counter() - started) * 1000
    values[0].zero_()
    timings = values.float()
    return DeploymentCostCalibration(
        dict(binding),
        policy,
        dict(runtime or capture_runtime_identity()),
        shape_hw,
        timings,
        _tail_estimate(timings, policy),
    )


def save_cost_calibration(path: Path, calibration: DeploymentCostCalibration) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibration.to_payload(), path)


def load_cost_calibration(path: Path) -> DeploymentCostCalibration:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("invalid deployment calibration artifact")
    return DeploymentCostCalibration.from_payload(payload)


def new_cost_history(device: torch.device | str) -> torch.Tensor:
    """Training diagnostic history; never a deployment calibration."""

    return torch.zeros((3, HISTORY_SIZE), dtype=torch.float32, device=device)


def record_costs(history: torch.Tensor, costs_ms: torch.Tensor, observations: int) -> int:
    if (
        history.shape != (3, HISTORY_SIZE)
        or costs_ms.ndim not in (1, 2)
        or not isinstance(observations, int)
        or observations < 0
        or not bool(
            torch.isfinite(history).all()
            & torch.isfinite(costs_ms).all()
            & (history >= 0).all()
            & (costs_ms >= 0).all()
        )
    ):
        raise ValueError("invalid action-cost diagnostic record")
    rows = costs_ms.reshape(-1, 3)
    if rows.shape[0] == 0 or not bool((rows[:, 0] == 0).all()):
        raise ValueError("STOP cost must remain exactly zero")
    for index, row in enumerate(rows):
        history[:, (observations + index) % HISTORY_SIZE].copy_(row.detach().to(history))
    return observations + int(rows.shape[0])


def estimated_costs(
    history: torch.Tensor | None,
    maximum_ms: torch.Tensor,
    observations: int,
    *,
    history_observations: int = 0,
    calibration: DeploymentCostCalibration | None = None,
    binding: Mapping[str, object] | None = None,
    runtime: Mapping[str, object] | None = None,
    shape_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Use an exact deployment artifact; otherwise return training max diagnostics."""

    del history, observations, history_observations
    if maximum_ms.shape != (3,) or not bool(
        torch.isfinite(maximum_ms).all() & (maximum_ms >= 0).all()
    ):
        raise ValueError("maximum_ms must be finite [3]")
    if calibration is None:
        return maximum_ms
    if (
        binding is None
        or runtime is None
        or shape_hw is None
        or not calibration.matches(binding, runtime, shape_hw)
    ):
        raise ValueError("deployment calibration identity/runtime/grid mismatch")
    return calibration.estimated_ms.to(maximum_ms)


def calibration_metadata(
    history: torch.Tensor | None,
    maximum_ms: torch.Tensor,
    observations: int,
    *,
    history_observations: int = 0,
    calibration: DeploymentCostCalibration | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "method": "deployment_artifact" if calibration is not None else "training_max_diagnostic",
        "history_size": HISTORY_SIZE if history is not None else 0,
        "history_observations": int(history_observations),
        "observations": int(observations),
        "training_max_ms": [float(x) for x in maximum_ms.detach().cpu()],
        "full_frame_action_cost": True,
    }
    if calibration is not None:
        payload = calibration.to_payload()
        result.update(
            calibration_schema=calibration.schema,
            calibration_binding_hash=calibration.binding_hash,
            calibration_policy=asdict(calibration.policy),
            calibrated_estimated_ms=[float(x) for x in calibration.estimated_ms.detach().cpu()],
            calibration_runtime=dict(calibration.runtime),
            calibration_shape_hw=list(calibration.shape_hw),
        )
        result["calibration_distribution"] = payload["distribution"]
    return result
