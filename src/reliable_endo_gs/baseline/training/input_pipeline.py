"""Bounded byte staging ahead of the unchanged supervised stereo tensor loader."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Generic, TypeVar

from reliable_endo_gs.data.staging import FileStager, deployment_file_index

T = TypeVar("T")


def supervised_training_file_index(
    sample_ids: Sequence[str], inventory: Path
) -> dict[str, tuple[str, ...]]:
    """Index TRAIN RGB/calibration and separate GT bytes without decoding tensors.

    Deployment indexing remains GT-free. Only this supervised baseline path adds
    the disparity TIFF; the canonical loader still returns GT separately.
    """
    deployment = deployment_file_index(sample_ids, inventory)
    available = set(inventory.read_text().splitlines())
    result = {}
    for sample_id, members in zip(sample_ids, deployment, strict=True):
        base = members[0].split("/left_finalpass/", 1)[0]
        frame = sample_id.rsplit("/", 1)[1]
        disparity = f"{base}/disparity/{frame}.tiff"
        if disparity not in available:
            raise ValueError(f"Missing supervised disparity in pinned inventory: {sample_id}")
        result[sample_id] = (*members, disparity)
    return result


class StagedInputPrefetch(Generic[T]):
    """Overlap byte-copy lookahead, canonical tensor loading and model execution.

    Byte workers stage a bounded window on Linux. Tensor workers call the supplied
    loader unchanged, preserving its tensor shapes, dtypes and device. Futures are
    consumed in the original absolute update order, including across epochs and
    resume. No model, optimizer, RNG or precision setting is owned here.
    """

    def __init__(
        self,
        *,
        sample_id_at: Callable[[int], str],
        load: Callable[[str], T],
        members: Mapping[str, Sequence[str]],
        stager: FileStager,
        start: int,
        tensor_ahead: int = 32,
        tensor_workers: int = 16,
        file_ahead: int = 256,
        io_workers: int = 2,
        initial_samples: int = 64,
        event: Callable[..., None] | None = None,
    ) -> None:
        if (
            start < 0
            or min(tensor_ahead, tensor_workers, file_ahead, io_workers) < 1
            or file_ahead < tensor_ahead
            or not 0 <= initial_samples <= file_ahead
        ):
            raise ValueError("Invalid staging/prefetch window or worker limit")
        self.sample_id_at, self.load, self.members = sample_id_at, load, members
        self.stager, self.event = stager, event
        self.expected = self._next_load = start
        self.tensor_ahead, self.file_ahead = tensor_ahead, file_ahead
        self.initial_samples = initial_samples
        self._io_pool = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="baseline-io")
        self._load_pool = ThreadPoolExecutor(
            max_workers=tensor_workers, thread_name_prefix="baseline-load"
        )
        self._staged: dict[int, tuple[str, Future[None]]] = {}
        self._loaded: deque[tuple[int, Future[T]]] = deque()
        self._first = True
        self._closed = False
        self.wait_seconds = 0.0
        self.initial_fill_seconds = 0.0
        self.consumed = self.peak_files_pending = self.peak_tensors_pending = 0

    def _load_after_staging(self, sample_id: str, ready: Future[None]) -> T:
        ready.result()  # No source fallback and no update before checksum verification.
        return self.load(sample_id)

    def _fill(self) -> None:
        for index in range(self.expected, self.expected + self.file_ahead):
            if index not in self._staged:
                sample_id = self.sample_id_at(index)
                self._staged[index] = (
                    sample_id,
                    self._io_pool.submit(self.stager.prepare, self.members[sample_id]),
                )
        self.peak_files_pending = max(self.peak_files_pending, len(self._staged))
        if self._first:
            began = time.perf_counter()
            for index in range(self.expected, self.expected + self.initial_samples):
                self._staged[index][1].result()
            self.initial_fill_seconds = time.perf_counter() - began
            self._first = False
        while len(self._loaded) < self.tensor_ahead:
            index = self._next_load
            sample_id, ready = self._staged[index]
            self._loaded.append(
                (index, self._load_pool.submit(self._load_after_staging, sample_id, ready))
            )
            self._next_load += 1
        self.peak_tensors_pending = max(self.peak_tensors_pending, len(self._loaded))

    def get(self) -> T:
        """Return exactly the next original sample/GT pair, or propagate its error."""
        if self._closed:
            raise RuntimeError("Staged input pipeline is closed")
        self._fill()
        index, ready = self._loaded[0]
        if index != self.expected:
            raise RuntimeError("Staged input order changed")
        began = time.perf_counter()
        try:
            value = ready.result()
        finally:
            self.wait_seconds += time.perf_counter() - began
        sample_id = self._staged[index][0]
        self.stager.release(self.members[sample_id])
        self._loaded.popleft()
        del self._staged[index]
        self.expected += 1
        self.consumed += 1
        if self.event is not None and (self.consumed == 1 or self.consumed % 32 == 0):
            self.event(
                "staged_input_ready",
                consumed=self.consumed,
                next_update_index=self.expected,
                initial_fill_seconds=self.initial_fill_seconds,
                input_wait_seconds=self.wait_seconds,
                pending_files=len(self._staged),
                pending_tensors=len(self._loaded),
            )
        return value

    def close(self) -> None:
        """Drain active byte copies and cancel unused lookahead without deleting data."""
        if not self._closed:
            for _, ready in self._loaded:
                ready.cancel()
            self._io_pool.shutdown(wait=True, cancel_futures=True)
            self._load_pool.shutdown(wait=True, cancel_futures=True)
            self._loaded.clear()
            self._staged.clear()
            self._closed = True
