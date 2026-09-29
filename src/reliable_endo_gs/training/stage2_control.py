"""Compatibility alias for the historical Stage-2 control trainer."""

from importlib import import_module

_IMPLEMENTATION = import_module("reliable_endo_gs.baseline.training.stage2_control")


def __getattr__(name: str) -> object:
    return getattr(_IMPLEMENTATION, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_IMPLEMENTATION)))


if __name__ == "__main__":
    raise SystemExit(_IMPLEMENTATION.main())
