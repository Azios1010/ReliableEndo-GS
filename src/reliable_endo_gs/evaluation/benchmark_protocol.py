"""Explicit evaluation protocol, independent of saved training configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Freeze metric, sampling and timing choices before looking at test results."""

    schema: str = "a04_benchmark.v1"
    data_role: str = "final_test"
    resolution_hw: tuple[int, int] | None = None
    warmup: int = 3
    profile_repetitions: int = 5
    cloud_max_points: int = 2048
    cloud_chunk_size: int = 256
    log_every: int = 25
    dedicated_gpu: bool = False

    def __post_init__(self) -> None:
        if self.schema != "a04_benchmark.v1" or self.data_role != "final_test":
            raise ValueError("benchmark requires a04_benchmark.v1 and final_test")
        for name in (
            "warmup",
            "profile_repetitions",
            "cloud_max_points",
            "cloud_chunk_size",
            "log_every",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "warmup" else 1):
                raise ValueError(f"invalid benchmark {name}")
        if type(self.dedicated_gpu) is not bool:
            raise ValueError("dedicated_gpu must explicitly be boolean")
        if self.resolution_hw is not None and (
            len(self.resolution_hw) != 2
            or any(type(v) is not int or v < 32 for v in self.resolution_hw)
        ):
            raise ValueError("benchmark resolution_hw requires H/W >=32")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def identity_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> BenchmarkProtocol:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(cls)}:
            raise ValueError("invalid/unknown benchmark protocol fields")
        if raw.get("resolution_hw") is not None:
            raw["resolution_hw"] = tuple(raw["resolution_hw"])
        return cls(**raw)
