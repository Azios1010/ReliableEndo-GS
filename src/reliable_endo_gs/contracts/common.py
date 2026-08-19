"""Lightweight structural validation shared by scientific contracts."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeVar

import torch

_ValueT = TypeVar("_ValueT")


def require_tensor(name: str, value: object) -> torch.Tensor:
    """Return ``value`` as a tensor or raise a field-specific error."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(value).__name__}")
    return value


def require_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    """Require an exact tensor rank without inspecting tensor values."""

    if tensor.ndim != rank:
        raise ValueError(
            f"{name} must have rank {rank}; expected shape semantics for rank {rank}, "
            f"got {tuple(tensor.shape)}"
        )


def require_min_rank(name: str, tensor: torch.Tensor, minimum_rank: int) -> None:
    """Require at least ``minimum_rank`` dimensions."""

    if tensor.ndim < minimum_rank:
        raise ValueError(
            f"{name} must have rank >= {minimum_rank}; got shape {tuple(tensor.shape)}"
        )


def require_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    """Require an exact shape."""

    actual = tuple(tensor.shape)
    if actual != expected:
        raise ValueError(f"{name} must have shape {expected}; got {actual}")


def require_trailing_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    """Require exact trailing dimensions while leaving leading dimensions variable."""

    actual = tuple(tensor.shape)
    if tensor.ndim < len(expected) or actual[-len(expected) :] != expected:
        raise ValueError(f"{name} must end with shape {expected}; got {actual}")


def require_batch_size(name: str, tensor: torch.Tensor, expected: int) -> None:
    """Require a tensor's leading batch dimension."""

    if tensor.ndim == 0 or tensor.shape[0] != expected:
        raise ValueError(
            f"{name} must have batch dimension {expected}; got shape {tuple(tensor.shape)}"
        )


def require_floating(name: str, tensor: torch.Tensor) -> None:
    """Require any PyTorch floating-point dtype without forcing precision."""

    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must have a floating-point dtype; got {tensor.dtype}")


def require_bool(name: str, tensor: torch.Tensor) -> None:
    """Require a boolean mask dtype."""

    if tensor.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool; got {tensor.dtype}")


def require_same_device(
    name: str,
    tensor: torch.Tensor,
    reference_name: str,
    reference: torch.Tensor,
) -> None:
    """Require coupled tensors to share a device without moving either tensor."""

    if tensor.device != reference.device:
        raise ValueError(
            f"{name} must be on the same device as {reference_name} "
            f"({reference.device}); got {tensor.device}"
        )


def require_same_dtype(
    name: str,
    tensor: torch.Tensor,
    reference_name: str,
    reference: torch.Tensor,
) -> None:
    """Require coupled floating tensors to use one explicit dtype."""

    if tensor.dtype != reference.dtype:
        raise TypeError(
            f"{name} must have the same dtype as {reference_name} "
            f"({reference.dtype}); got {tensor.dtype}"
        )


def freeze_mapping(values: Mapping[str, _ValueT], *, name: str) -> Mapping[str, _ValueT]:
    """Copy a string-keyed mapping into a read-only view."""

    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping; got {type(values).__name__}")
    copied: dict[str, _ValueT] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings; got {type(key).__name__}")
        if not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        copied[key] = value
    return MappingProxyType(copied)
