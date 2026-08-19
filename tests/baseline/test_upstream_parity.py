"""Parity harness and optional pinned-upstream import smoke tests."""

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.baseline.adapter import (
    EndoE2EGSAdapter,
    compare_tensor_translation,
)
from reliable_endo_gs.baseline.upstream import (
    import_upstream_module,
    inspect_capabilities,
)


def test_cpu_translation_parity_is_exact_for_unmodified_fields(
    upstream_output: Mapping[str, object],
) -> None:
    left = upstream_output["lmain"]
    assert isinstance(left, Mapping)
    disparity = left["flow_pred"]
    rendered = left["img_pred"]
    assert isinstance(disparity, torch.Tensor)
    assert isinstance(rendered, torch.Tensor)

    prediction = EndoE2EGSAdapter.stereo_prediction(upstream_output)
    render_output = EndoE2EGSAdapter.render_output(upstream_output)
    disparity_parity = compare_tensor_translation(disparity, prediction.disparity)
    render_parity = compare_tensor_translation(rendered, render_output.image)

    assert disparity_parity.matches
    assert disparity_parity.max_abs_difference == 0.0
    assert render_parity.matches
    assert render_parity.mean_abs_difference == 0.0


def test_color_layout_parity_matches_upstream_pts2render_step(
    upstream_output: Mapping[str, object],
) -> None:
    left = upstream_output["lmain"]
    assert isinstance(left, Mapping)
    image = left["img"]
    assert isinstance(image, torch.Tensor)
    direct = image.permute(0, 2, 3, 1).reshape(2, 12, 3) * 0.5 + 0.5
    translated = EndoE2EGSAdapter.gaussian_field(upstream_output).colors

    parity = compare_tensor_translation(direct, translated)
    assert parity.matches
    assert parity.max_abs_difference == 0.0


def test_pinned_upstream_network_import_without_rasterizer() -> None:
    root = Path("third_party/endo_e2e_gs")
    capabilities = inspect_capabilities(root)
    if not capabilities.upstream_source or not capabilities.python_dependencies:
        pytest.skip(
            "pinned upstream import unavailable: "
            f"source={capabilities.upstream_source}, "
            f"missing_python_dependencies={capabilities.missing_python_dependencies}"
        )
    code = (
        "from pathlib import Path; "
        "from reliable_endo_gs.baseline.upstream import import_upstream_module; "
        "module = import_upstream_module('lib.network', Path('third_party/endo_e2e_gs')); "
        "assert hasattr(module, 'StereoEndoModel')"
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.upstream_gpu
def test_optional_pinned_upstream_renderer_import() -> None:
    root = Path("third_party/endo_e2e_gs")
    capabilities = inspect_capabilities(root)
    if not capabilities.cuda_baseline_runnable:
        pytest.skip(
            "pinned upstream renderer unavailable: "
            f"cuda={capabilities.cuda}, rasterizer={capabilities.rasterizer}, "
            f"missing_python_dependencies={capabilities.missing_python_dependencies}"
        )
    module = import_upstream_module("gaussian_renderer", root, require_rasterizer=True)
    assert hasattr(module, "render")
