"""Configurable deployment measurement protocol, not a measured speed claim."""

from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class DeploymentRuntimeProfile:
    """Bounded RTX 5070 Ti batch-one deployment protocol.

    The values are an engineering safety envelope, not a measured performance
    claim.  A caller may replace them explicitly before collecting evidence.
    """

    hardware: str = "NVIDIA RTX 5070 Ti 16 GB"
    device: str = "cuda:0"
    memory_gib: int = 16
    batch_size: int = 1
    precision: str = "fp16"
    geometry_precision: str = "fp32"
    resolution_hw: tuple[int, int] | None = (512, 640)
    deadline_ms: float | None = 100.0
    warmup: int = 20
    repetitions: int = 1000
    synchronize: bool = True
    compilation: str = "eager"

    def __post_init__(self) -> None:
        if not self.hardware or not self.device or self.compilation not in ("eager", "compiled"):
            raise ValueError("explicit hardware/device and supported compilation policy required")
        for name in ("memory_gib", "batch_size", "warmup", "repetitions"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "warmup" else 1):
                raise ValueError(f"invalid {name}")
        if self.precision not in ("fp16", "bf16") or self.geometry_precision != "fp32":
            raise ValueError("deployment target fp16/bf16 with explicit fp32 geometry")
        if self.synchronize is not True:
            raise ValueError("synchronized timing required")
        if self.resolution_hw is not None:
            if len(self.resolution_hw) != 2 or any(
                type(x) is not int or x < 1 for x in self.resolution_hw
            ):
                raise ValueError("resolution_hw requires positive integer H,W")
        if self.deadline_ms is not None:
            if (
                isinstance(self.deadline_ms, bool)
                or not isfinite(self.deadline_ms)
                or self.deadline_ms <= 0
            ):
                raise ValueError("deadline_ms must be finite and positive")

    def require_benchmark_ready(self) -> None:
        """Reject unresolved protocol choices, not test hardware availability."""
        if self.resolution_hw is None or self.deadline_ms is None:
            raise ValueError("freeze resolution_hw and deadline_ms before benchmarking")


def load_deployment_profile(path: Path) -> DeploymentRuntimeProfile:
    """Load a strict standalone profile; historical experiment configs stay intact."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) - {f.name for f in fields(DeploymentRuntimeProfile)}:
        raise ValueError("profile must be a mapping with known runtime fields")
    if data.get("resolution_hw") is not None:
        data["resolution_hw"] = tuple(data["resolution_hw"])
    return DeploymentRuntimeProfile(**data)
