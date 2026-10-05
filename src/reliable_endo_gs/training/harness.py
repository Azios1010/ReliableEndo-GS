"""Sequential data cursors, safe RNG snapshots and streaming training logs."""

from __future__ import annotations

import json
import random
import signal
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np
import torch
from torch.utils.data import Sampler


class CursorSampler(Sampler[int]):
    """Visit sequential indices starting at the next unconsumed sample."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.start = 0

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, self.size))

    def __len__(self) -> int:
        return self.size - self.start


def capture_rng(device: torch.device) -> dict[str, object]:
    """Use tensors/primitives only, compatible with weights_only checkpoint loading."""

    numpy_state = cast(tuple[Any, ...], np.random.get_state(legacy=True))
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    """Restore the process RNG and the one CUDA device owned by this trainer."""

    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        if not isinstance(state.get("torch_cuda"), torch.Tensor):
            raise ValueError("training checkpoint lacks the CUDA RNG state")
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


class TrainingLog:
    """Flush JSONL to stdout and optionally a new, never-overwritten log file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._handle: TextIO | None = None
        self._stdout_available = True

    def __enter__(self) -> TrainingLog:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("x", encoding="utf-8")
        return self

    def __exit__(self, *args: object) -> None:
        if self._handle is not None:
            self._handle.close()

    def emit(self, event: Mapping[str, object]) -> None:
        """Flush each event; retain the disk log when stdout disconnects."""

        line = json.dumps(dict(event), sort_keys=True, allow_nan=False)
        if self._handle is not None:
            self._handle.write(line + "\n")
            self._handle.flush()
        if self._stdout_available:
            try:
                print(line, flush=True)
            except BrokenPipeError:
                if self._handle is None:
                    raise
                self._stdout_available = False
                self._handle.write(
                    json.dumps(
                        {
                            "event": "console_disconnected",
                            "training_log": str(self.path),
                        }
                    )
                    + "\n"
                )
                self._handle.flush()


class TrainingSignals:
    """Drain SIGINT/SIGTERM/SIGHUP to the next committed sample boundary."""

    def __init__(self) -> None:
        self.reason: str | None = None
        self._previous: dict[signal.Signals, Any] = {}

    def __enter__(self) -> TrainingSignals:
        supported = [signal.SIGINT, signal.SIGTERM]
        hangup = getattr(signal, "SIGHUP", None)
        if hangup is not None:
            supported.append(hangup)
        for signum in supported:
            self._previous[signum] = signal.signal(signum, self._request_stop)
        return self

    def __exit__(self, *args: object) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)

    def _request_stop(self, signum: int, frame: object) -> None:
        self.reason = signal.Signals(signum).name

    def requested(self) -> str | None:
        """Return the received termination signal without interrupting Adam."""

        return self.reason


def checkpoint_rng(state: object) -> Mapping[str, Any]:
    """Reject incomplete RNG snapshots before modifying a resumed model."""

    if not isinstance(state, Mapping) or not all(
        key in state for key in ("python", "numpy", "torch_cpu", "torch_cuda")
    ):
        raise ValueError("training checkpoint lacks a complete RNG state")
    if not isinstance(state["torch_cpu"], torch.Tensor):
        raise ValueError("training checkpoint has an invalid CPU RNG state")
    return cast(Mapping[str, Any], state)
