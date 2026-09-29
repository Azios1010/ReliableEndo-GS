"""Compatibility alias for the historical SCARED-C Stage-2 adapter."""

from importlib import import_module

_IMPLEMENTATION = import_module("reliable_endo_gs.baseline.data.scared_c_stage2")


def __getattr__(name: str) -> object:
    return getattr(_IMPLEMENTATION, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_IMPLEMENTATION)))
