"""Bounded asynchronous I/O ahead of an unchanged, ordered tensor loader."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Generic, Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class IndexedSamples(Protocol[T_co]):
    """Dataset values retain their original type, tensor shape, dtype and device."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> T_co: ...


class OrderedInputPrefetch(Generic[T]):
    """Prepare up to ``ahead`` samples while main-thread GPU work continues.

    Workers perform only byte I/O via ``prepare``. The canonical tensor loader
    executes on its original thread, preserving RNG, CUDA stream and rounding.
    No model reload or optimizer reset occurs at queue boundaries.
    """

    def __init__(
        self,
        samples: IndexedSamples[T],
        prepare: Callable[[int], None],
        *,
        start: int = 0,
        ahead: int = 64,
        workers: int = 1,
        release: Callable[[int], None] | None = None,
        event: Callable[..., None] | None = None,
    ) -> None:
        if not len(samples) or not 0 <= start < len(samples) or ahead < 1 or workers < 1:
            raise ValueError("Invalid prefetch size/start/worker limit")
        self.samples, self.prepare = samples, prepare
        self.release, self.event = release, event
        self.expected = start
        self.ahead, self.workers = min(ahead, len(samples)), workers
        self._pool: ThreadPoolExecutor | None = None
        self._pending: dict[int, Future[None]] = {}
        self.wait_seconds = 0.0
        self.peak_pending = 0
        self.consumed = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __enter__(self) -> OrderedInputPrefetch[T]:
        if self._pool is not None:
            raise RuntimeError("Prefetch already started")
        self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="input-io")
        self._fill()
        return self

    def _fill(self) -> None:
        assert self._pool is not None
        for offset in range(self.ahead):
            index = (self.expected + offset) % len(self)
            if index not in self._pending:
                self._pending[index] = self._pool.submit(self.prepare, index)
        self.peak_pending = max(self.peak_pending, len(self._pending))

    def __getitem__(self, index: int) -> T:
        if self._pool is None:
            raise RuntimeError("Use prefetch as a context manager")
        if index != self.expected:
            raise ValueError("Prefetch requires unchanged sequential sample order")
        began = time.perf_counter()
        self._pending[index].result()  # propagate errors before any sample update
        self.wait_seconds += time.perf_counter() - began
        value = self.samples[index]
        if self.release is not None:
            self.release(index)
        del self._pending[index]
        self.expected = (index + 1) % len(self)
        self.consumed += 1
        self._fill()
        if self.event is not None and (self.consumed == 1 or self.consumed % 32 == 0):
            self.event(
                "sample_input_ready",
                index=index,
                consumed=self.consumed,
                input_wait_seconds=self.wait_seconds,
                pending=len(self._pending),
            )
        return value

    def __exit__(self, *args: object) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._pending.clear()
