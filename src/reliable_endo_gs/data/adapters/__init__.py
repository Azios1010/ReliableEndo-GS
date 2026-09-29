"""Explicit dataset adapter implementations.

Active adapters are imported eagerly; the historical SCARED-C adapter is
available only through an explicit lazy attribute access so importing active
SCARED modules cannot pull the historical module graph into ``sys.modules``.
"""

from importlib import import_module
from typing import Any

from reliable_endo_gs.data.adapters.c3vd import C3vdAdapter
from reliable_endo_gs.data.adapters.endonerf import EndoNerfAdapter
from reliable_endo_gs.data.adapters.scared import ScaredAdapter

__all__ = ["C3vdAdapter", "EndoNerfAdapter", "ScaredAdapter", "ScaredCAdapter"]


def __getattr__(name: str) -> Any:
    if name == "ScaredCAdapter":
        return getattr(import_module("reliable_endo_gs.data.adapters.scared_c"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
