"""One device-to-host transfer for several scalars.

Each ``tensor.item()``/``bool(tensor)`` synchronises the stream and stalls the host; reading K diagnostics
that way costs K synchronisation points. These helpers stack the scalars on their device and copy them once.
Counts and flags are exactly representable in float64 (counts below 2**53), so the returned Python values equal
the per-scalar reads bit for bit.

CUDA graph capture: a host read synchronises, which is illegal while a stream is capturing. ``graph_capture_diagnostics`` lets ``read_ints``
(used only for diagnostic counts) return placeholders during a capture, whose Python results are discarded after capture because only the
captured tensors are replayed; validation checks are never placeholders: they are collected (``collected_checks``), stacked on the device
into one :class:`PackedChecks` and registered again at every replay.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import NamedTuple

import torch


def read_scalars(values: Sequence[torch.Tensor]) -> list[float]:
    """Return one Python float per single-element tensor, with a single host transfer.

    All values must live on the same device. An empty sequence returns an empty list.
    """
    if not values:
        return []
    if any(value.numel() != 1 for value in values):
        raise ValueError("read_scalars requires single-element tensors")
    return [
        float(x) for x in torch.stack([v.reshape(()).to(torch.float64) for v in values]).tolist()
    ]


_CAPTURE_DIAGNOSTICS: ContextVar[bool] = ContextVar("graph_capture_diagnostics", default=False)


@contextmanager
def graph_capture_diagnostics() -> Iterator[None]:
    """Within a CUDA graph capture, ``read_ints`` returns zero placeholders instead of reading (see the module docstring)."""
    token = _CAPTURE_DIAGNOSTICS.set(True)
    try:
        yield
    finally:
        _CAPTURE_DIAGNOSTICS.reset(token)


def read_diagnostic_scalars(values: Sequence[torch.Tensor]) -> list[float]:
    """:func:`read_scalars` for reported diagnostics, never for values that steer computation.

    Inside ``graph_capture_diagnostics()`` nothing is read and every value is a placeholder (0.0): the objects holding them are
    discarded after the capture, only the captured tensors are replayed.
    """
    if _CAPTURE_DIAGNOSTICS.get():
        if not torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "graph_capture_diagnostics is only valid while a CUDA graph is being captured"
            )
        return [0.0] * len(values)
    return read_scalars(values)


def read_ints(*values: torch.Tensor) -> list[int]:
    """Diagnostic counts (e.g. ``mask.sum()``) as Python ints with one host transfer; placeholders (0) in a graph capture."""
    return [int(x) for x in read_diagnostic_scalars(values)]


def read_bools(*values: torch.Tensor) -> list[bool]:
    """Flags (e.g. ``torch.isfinite(x).all()``) as Python bools with one host transfer."""
    return [bool(x) for x in read_scalars(values)]


_Check = tuple[torch.Tensor, Callable[[], BaseException]]


class PackedChecks(NamedTuple):
    """``flags[k]`` must be true, else ``factories[k]()`` is raised; K checks stacked on the device (one element each)."""

    flags: torch.Tensor
    factories: tuple[Callable[[], BaseException], ...]


_DEFERRED: ContextVar[list[_Check | PackedChecks] | None] = ContextVar(
    "deferred_validation", default=None
)


def pack_checks(checks: Sequence[_Check]) -> PackedChecks:
    """Stack single-element boolean conditions into one tensor, preserving their order."""
    return PackedChecks(
        torch.stack([condition.reshape(()) for condition, _ in checks]),
        tuple(make_exception for _, make_exception in checks),
    )


def raise_first_failure(checks: Sequence[_Check | PackedChecks]) -> None:
    """Evaluate device conditions with ONE host transfer; raise the first failing check's exception.

    ``checks`` holds ``(condition, make_exception)`` pairs (or :class:`PackedChecks`) in the order the caller used to test them one by
    one; ``condition`` is a single-element boolean tensor that must be true. The exception factory runs only
    on failure, so messages are built lazily. Inside ``deferred_validation()`` the checks are collected and
    evaluated together when the scope ends instead.
    """
    if not checks:
        return
    pending = _DEFERRED.get()
    if pending is not None:
        pending.extend(checks)
        return
    conditions: list[torch.Tensor] = []
    factories: list[Callable[[], BaseException]] = []
    for entry in checks:
        if isinstance(entry, PackedChecks):
            conditions.append(entry.flags.reshape(-1).to(torch.float64))
            factories.extend(entry.factories)
        else:
            conditions.append(entry[0].reshape(1).to(torch.float64))
            factories.append(entry[1])
    passed = torch.cat(conditions).tolist()
    for ok, make_exception in zip(passed, factories, strict=True):
        if not ok:
            raise make_exception()


@contextmanager
def collected_checks() -> Iterator[list[_Check | PackedChecks]]:
    """Collect checks into the yielded list WITHOUT evaluating them (a CUDA graph capture re-registers them at every replay)."""
    if _DEFERRED.get() is not None:
        raise RuntimeError("collected_checks cannot be nested in a deferred_validation scope")
    pending: list[_Check | PackedChecks] = []
    token = _DEFERRED.set(pending)
    try:
        yield pending
    finally:
        _DEFERRED.reset(token)


@contextmanager
def deferred_validation() -> Iterator[None]:
    """Collect ``raise_first_failure`` checks and evaluate them with one host transfer on exit.

    Nothing escapes the block unchecked: the first failing check (in registration order) is raised when the
    block ends, before the caller can use the result. If the body itself raises, the pending checks are still
    evaluated and a failed validation is raised in its place (chained from the body's exception), so invalid
    input is reported as invalid input rather than as a downstream symptom. Nested scopes join the outermost.
    """
    if _DEFERRED.get() is not None:
        yield
        return
    pending: list[_Check | PackedChecks] = []
    token = _DEFERRED.set(pending)
    try:
        yield
    except BaseException as error:
        _DEFERRED.reset(token)
        try:
            raise_first_failure(pending)
        except BaseException as validation_error:
            raise validation_error from error
        raise
    _DEFERRED.reset(token)
    raise_first_failure(pending)
