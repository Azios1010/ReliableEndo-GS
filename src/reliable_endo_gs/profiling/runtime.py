"""Synchronized, callable-injected runtime profiling for development runs."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


class ProfilingError(ValueError):
    """Raised for an invalid profiling protocol."""


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Latency and memory measurements for one fixed callable protocol."""

    warmup: int
    repetitions: int
    batch_size: int
    device: str
    synchronized: bool
    latencies_ms: tuple[float, ...]
    peak_memory_bytes: int | None = None

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.latencies_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.latencies_ms)

    @property
    def p90_ms(self) -> float:
        ordered = sorted(self.latencies_ms)
        index = min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1)
        return ordered[index]

    @property
    def fps(self) -> float:
        return self.batch_size * 1000.0 / self.mean_ms if self.mean_ms > 0 else float("inf")

    @property
    def latency_mean_ms(self) -> float:
        return self.mean_ms

    @property
    def latency_median_ms(self) -> float:
        return self.median_ms

    @property
    def latency_p90_ms(self) -> float:
        return self.p90_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup": self.warmup,
            "repetitions": self.repetitions,
            "batch_size": self.batch_size,
            "device": self.device,
            "synchronized": self.synchronized,
            "latencies_ms": list(self.latencies_ms),
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p90_ms": self.p90_ms,
            "fps": self.fps,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


def _synchronizer(device: torch.device | str | None) -> tuple[str, Callable[[], None] | None]:
    try:
        resolved = torch.device(device) if device is not None else torch.device("cpu")
    except (RuntimeError, TypeError) as error:
        raise ProfilingError(f"invalid profiling device: {device!r}") from error
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ProfilingError("CUDA profiling requested but CUDA is unavailable")
        return str(resolved), lambda: torch.cuda.synchronize(resolved)
    return str(resolved), None


def profile_callable(
    function: Callable[[], object],
    *,
    warmup: int = 3,
    repetitions: int = 10,
    batch_size: int = 1,
    device: torch.device | str | None = None,
    synchronize: Callable[[], None] | None = None,
    reset_peak_memory: Callable[[], None] | None = None,
    peak_memory: Callable[[], int] | None = None,
) -> tuple[RuntimeProfile, object]:
    """Profile an injected end-to-end callable.

    Synchronization is injected so tests and non-CUDA backends can provide an
    observable barrier. The returned output is the final timed-call output;
    warm-up outputs are deliberately discarded.
    """

    if not callable(function):
        raise ProfilingError("function must be callable")
    if not isinstance(warmup, int) or warmup < 0:
        raise ProfilingError("warmup must be a non-negative integer")
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ProfilingError("repetitions must be a positive integer")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ProfilingError("batch_size must be a positive integer")
    device_name, default_sync = _synchronizer(device)
    barrier = synchronize if synchronize is not None else default_sync
    if reset_peak_memory is not None:
        reset_peak_memory()
    for _ in range(warmup):
        function()
        if barrier is not None:
            barrier()
    latencies: list[float] = []
    output: object = None
    for _ in range(repetitions):
        if barrier is not None:
            barrier()
        start = time.perf_counter()
        output = function()
        if barrier is not None:
            barrier()
        latencies.append((time.perf_counter() - start) * 1000.0)
    memory = peak_memory() if peak_memory is not None else None
    return RuntimeProfile(
        warmup=warmup,
        repetitions=repetitions,
        batch_size=batch_size,
        device=device_name,
        synchronized=barrier is not None,
        latencies_ms=tuple(latencies),
        peak_memory_bytes=memory,
    ), output


def profile_inference(
    function: Callable[[], object],
    *,
    warmup: int = 3,
    repetitions: int = 10,
    batch_size: int = 1,
    device: torch.device | str | None = None,
    synchronize: Callable[[], None] | None = None,
) -> RuntimeProfile:
    """Profile an injected callable and return only its measurement record."""

    profile, _ = profile_callable(
        function,
        warmup=warmup,
        repetitions=repetitions,
        batch_size=batch_size,
        device=device,
        synchronize=synchronize,
    )
    return profile


profile_runtime = profile_inference
