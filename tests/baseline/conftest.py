"""Synthetic, non-scientific upstream-shaped fixtures for adapter tests."""

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def stage2_modules(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Load real CPU/stdlib-only helpers without importing the native trainer."""
    root = Path(__file__).resolve().parents[2] / "third_party" / "endo_e2e_gs" / "lib"
    package = ModuleType("_stage2_test_lib")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    modules = {}
    for name in ("scale_parameterization", "stage2_diagnostics", "stage2_run"):
        spec = importlib.util.spec_from_file_location(
            f"{package.__name__}.{name}", root / f"{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return SimpleNamespace(**modules)


@pytest.fixture
def upstream_output() -> Mapping[str, object]:
    batch, height, width = 2, 3, 4
    count = height * width
    return {
        "lmain": {
            "img": torch.linspace(-1.0, 1.0, batch * 3 * height * width).reshape(
                batch, 3, height, width
            ),
            "mask": torch.tensor(
                [0.0, 0.49, 0.5, 1.0] * (batch * height), dtype=torch.float32
            ).reshape(batch, 1, height, width),
            "flow_pred": torch.arange(batch * height * width, dtype=torch.float32).reshape(
                batch, 1, height, width
            ),
            "xyz": torch.arange(batch * count * 3, dtype=torch.float32).reshape(batch, count, 3),
            "pts_valid": torch.ones(batch, count, dtype=torch.bool),
            "rot_maps": torch.nn.functional.normalize(torch.ones(batch, 4, height, width), dim=1),
            "scale_maps": torch.full((batch, 3, height, width), 0.01),
            "opacity_maps": torch.full((batch, 1, height, width), 0.75),
            "img_pred": torch.full((batch, 3, height, width), 0.25),
        }
    }
