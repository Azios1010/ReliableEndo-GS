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
    if not (capabilities.renderer_ready and capabilities.cuda):
        pytest.skip(
            "pinned upstream renderer unavailable: "
            f"cuda={capabilities.cuda}, rasterizer={capabilities.rasterizer}, "
            f"renderer_ready={capabilities.renderer_ready}, "
            f"missing_python_dependencies={capabilities.missing_python_dependencies}"
        )
    module = import_upstream_module("gaussian_renderer", root, require_rasterizer=True)
    assert hasattr(module, "render")


def test_capabilities_report_absent_corr_sampler_and_independent_renderer_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("third_party/endo_e2e_gs")
    capabilities = inspect_capabilities(root)

    assert isinstance(capabilities.upstream_source, bool)
    assert isinstance(capabilities.python_dependencies, bool)
    assert isinstance(capabilities.rasterizer, bool)
    assert isinstance(capabilities.corr_sampler, bool)
    assert isinstance(capabilities.cuda, bool)
    assert isinstance(capabilities.renderer_ready, bool)
    assert isinstance(capabilities.native_inference_ready, bool)
    assert isinstance(capabilities.cuda_baseline_runnable, bool)
    assert capabilities.renderer_import_ready == capabilities.renderer_ready
    assert capabilities.native_inference_dependency_ready == capabilities.native_inference_ready

    # Simulated case 1: rasterizer present, corr_sampler absent
    # Renderer import is ready, but native inference dependency is NOT ready.
    available_map_1 = {
        "cv2": True,
        "scipy": True,
        "torchvision": True,
        "yacs": True,
        "diff_gaussian_rasterization": True,
        "corr_sampler": False,
    }
    monkeypatch.setattr(
        "reliable_endo_gs.baseline.upstream._module_available",
        lambda name: available_map_1.get(name, False),
    )
    caps_1 = inspect_capabilities(root)
    assert caps_1.python_dependencies is True
    assert caps_1.rasterizer is True
    assert caps_1.corr_sampler is False
    assert caps_1.renderer_ready is True
    assert caps_1.renderer_import_ready is True
    assert caps_1.native_inference_ready is False
    assert caps_1.native_inference_dependency_ready is False
    assert caps_1.cuda_baseline_runnable is False

    # Simulated case 2: rasterizer absent, corr_sampler present
    available_map_2 = {
        "cv2": True,
        "scipy": True,
        "torchvision": True,
        "yacs": True,
        "diff_gaussian_rasterization": False,
        "corr_sampler": True,
    }
    monkeypatch.setattr(
        "reliable_endo_gs.baseline.upstream._module_available",
        lambda name: available_map_2.get(name, False),
    )
    caps_2 = inspect_capabilities(root)
    assert caps_2.renderer_ready is False
    assert caps_2.native_inference_ready is False
    assert caps_2.cuda_baseline_runnable is False

    # Simulated case 3: both rasterizer and corr_sampler present
    available_map_3 = {
        "cv2": True,
        "scipy": True,
        "torchvision": True,
        "yacs": True,
        "diff_gaussian_rasterization": True,
        "corr_sampler": True,
    }
    monkeypatch.setattr(
        "reliable_endo_gs.baseline.upstream._module_available",
        lambda name: available_map_3.get(name, False),
    )
    caps_3 = inspect_capabilities(root)
    assert caps_3.renderer_ready is True
    assert caps_3.native_inference_ready is True
    assert caps_3.cuda_baseline_runnable == torch.cuda.is_available()


def test_import_upstream_module_explicit_missing_corr_sampler_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reliable_endo_gs.baseline.upstream import (
        PINNED_COMMIT,
        UpstreamGitState,
        UpstreamUnavailableError,
    )

    root = Path("third_party/endo_e2e_gs")
    monkeypatch.setattr(
        "reliable_endo_gs.baseline.upstream.require_pinned_upstream",
        lambda _root, **_kwargs: UpstreamGitState(commit=PINNED_COMMIT, dirty=False),
    )
    available_map = {
        "cv2": True,
        "scipy": True,
        "torchvision": True,
        "yacs": True,
        "diff_gaussian_rasterization": True,
        "corr_sampler": False,
    }
    monkeypatch.setattr(
        "reliable_endo_gs.baseline.upstream._module_available",
        lambda name: available_map.get(name, False),
    )

    with pytest.raises(UpstreamUnavailableError, match="corr_sampler is unavailable"):
        import_upstream_module("lib.network", root, require_corr_sampler=True)


def test_cpu_ci_safety_for_capability_probe() -> None:
    code = (
        "import sys; "
        "from pathlib import Path; "
        "from reliable_endo_gs.baseline.upstream import inspect_capabilities; "
        "caps = inspect_capabilities(Path('third_party/endo_e2e_gs')); "
        "assert 'diff_gaussian_rasterization' not in sys.modules; "
        "assert 'corr_sampler' not in sys.modules; "
        "assert isinstance(caps.corr_sampler, bool); "
        "assert isinstance(caps.renderer_ready, bool)"
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


def test_isolated_upstream_import_path_scopes_and_restores_bytecode_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reliable_endo_gs.baseline.upstream import _isolated_upstream_import_path

    fake_root = tmp_path / "fake_upstream"
    fake_root.mkdir()

    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    assert sys.dont_write_bytecode is False

    with _isolated_upstream_import_path(fake_root):
        assert sys.dont_write_bytecode is True
        assert str(fake_root.resolve()) in sys.path

    assert sys.dont_write_bytecode is False
    assert str(fake_root.resolve()) not in sys.path


def test_isolated_upstream_import_path_restores_bytecode_flag_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reliable_endo_gs.baseline.upstream import _isolated_upstream_import_path

    fake_root = tmp_path / "fake_upstream"
    fake_root.mkdir()

    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    assert sys.dont_write_bytecode is False

    with pytest.raises(RuntimeError, match="simulated failure"):
        with _isolated_upstream_import_path(fake_root):
            assert sys.dont_write_bytecode is True
            raise RuntimeError("simulated failure")

    assert sys.dont_write_bytecode is False
    assert str(fake_root.resolve()) not in sys.path


def test_isolated_upstream_import_path_preserves_initial_true_bytecode_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reliable_endo_gs.baseline.upstream import _isolated_upstream_import_path

    fake_root = tmp_path / "fake_upstream"
    fake_root.mkdir()

    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    assert sys.dont_write_bytecode is True

    with _isolated_upstream_import_path(fake_root):
        assert sys.dont_write_bytecode is True

    assert sys.dont_write_bytecode is True
