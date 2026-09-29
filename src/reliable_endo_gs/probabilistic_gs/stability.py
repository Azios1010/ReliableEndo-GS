"""Compatibility alias for the canonical Gaussian stability module."""

from importlib import import_module

_IMPLEMENTATION = import_module("reliable_endo_gs.gaussians.stability")


def __getattr__(name: str) -> object:
    return getattr(_IMPLEMENTATION, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_IMPLEMENTATION)))
