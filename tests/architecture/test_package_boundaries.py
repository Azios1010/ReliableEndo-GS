"""Executable boundaries for independent reconstruction and historical controls."""

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def test_core_imports_work_with_baseline_and_optional_research_blocked() -> None:
    code = """
import importlib.abc
import sys

blocked = (
    "reliable_endo_gs.baseline",
    "reliable_endo_gs.probabilistic_gs",
    "reliable_endo_gs.uncertainty",
    "reliable_endo_gs.reliability",
    "lib", "core", "diff_gaussian_rasterization", "corr_sampler",
)
class RejectHistoricalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in blocked):
            raise AssertionError("Core imported historical dependency: " + fullname)
        return None
sys.meta_path.insert(0, RejectHistoricalImports())
import reliable_endo_gs.contracts
import reliable_endo_gs.data
import reliable_endo_gs.geometry
import reliable_endo_gs.gaussians
import reliable_endo_gs.training
"""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("historical", "canonical", "symbol"),
    [
        ("training.stage1_scared_c", "baseline.training.stage1_scared_c", "Stage1ScaredCConfig"),
        ("training.stage2_control", "baseline.training.stage2_control", "Protocol"),
        ("training.stage2_scared_c", "baseline.training.stage2_scared_c", "Stage2ScaredCProtocol"),
        ("data.stage2", "baseline.data.stage2", "compute_safe_depth"),
        (
            "data.scared_c_stage2",
            "baseline.data.scared_c_stage2",
            "cached_stage1_to_scared_c_stage2_sample",
        ),
        ("evaluation.p1_runner", "baseline.evaluation.p1_runner", "_verify_no_final_access"),
        (
            "probabilistic_gs.support_covariance",
            "gaussians.support_covariance",
            "build_surface_covariance",
        ),
        ("probabilistic_gs.stability", "gaussians.stability", "CovarianceStabilityPolicy"),
    ],
)
def test_legacy_imports_resolve_to_one_implementation(
    historical: str, canonical: str, symbol: str
) -> None:
    old = importlib.import_module("reliable_endo_gs." + historical)
    new = importlib.import_module("reliable_endo_gs." + canonical)
    assert getattr(old, symbol) is getattr(new, symbol)


def test_legacy_pickle_global_and_repository_paths_still_resolve() -> None:
    from reliable_endo_gs.baseline.training.stage1_scared_c import (
        DEFAULT_CONFIG_PATH,
        REPOSITORY_ROOT,
        Stage1ScaredCConfig,
    )
    from reliable_endo_gs.gaussians.stability import CovarianceStabilityPolicy

    # Protocol-0 global references emulate historical module names without
    # mutating class metadata or loading any checkpoint.
    assert (
        pickle.loads(b"creliable_endo_gs.training.stage1_scared_c\nStage1ScaredCConfig\n.")
        is Stage1ScaredCConfig
    )
    assert (
        pickle.loads(b"creliable_endo_gs.probabilistic_gs.stability\nCovarianceStabilityPolicy\n.")
        is CovarianceStabilityPolicy
    )
    assert REPOSITORY_ROOT == Path(__file__).resolve().parents[2]
    assert DEFAULT_CONFIG_PATH.is_file()


def test_surface_math_has_gradients_without_uncertainty() -> None:
    from reliable_endo_gs.gaussians import build_surface_covariance

    rotations = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    scales = torch.tensor([[[0.1, 0.2, 0.3]]], requires_grad=True)
    result = build_surface_covariance(rotations, scales)
    assert result.valid_mask.all()
    torch.testing.assert_close(result.cov_surface, torch.diag_embed(scales.square()))
    result.cov_surface.sum().backward()
    torch.testing.assert_close(scales.grad, 2 * scales)
