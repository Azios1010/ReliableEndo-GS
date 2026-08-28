"""Optional capability detection and isolated loading for pinned upstream code."""

import importlib
import importlib.util
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

UPSTREAM_PROJECT = "Intelligent-Imaging-Center/Endo-E2E-GS"
UPSTREAM_REPOSITORY = "https://github.com/Intelligent-Imaging-Center/Endo-E2E-GS.git"
PINNED_COMMIT = "186fa2b4a2159b28393492f6df1aa444b54391a8"


class UpstreamUnavailableError(RuntimeError):
    """Raised only when an explicitly requested upstream capability is unavailable."""


@dataclass(frozen=True, slots=True)
class UpstreamGitState:
    """Observed immutable identity and dirty state of the submodule."""

    commit: str
    dirty: bool


@dataclass(frozen=True, slots=True)
class BaselineCapabilities:
    """Read-only detection result; it never imports or initializes the rasterizer or CUDA kernels."""

    upstream_source: bool
    python_dependencies: bool
    rasterizer: bool
    corr_sampler: bool
    cuda: bool
    renderer_ready: bool
    native_inference_ready: bool
    cuda_baseline_runnable: bool
    missing_python_dependencies: tuple[str, ...]

    @property
    def renderer_import_ready(self) -> bool:
        """Alias for renderer import readiness."""
        return self.renderer_ready

    @property
    def native_inference_dependency_ready(self) -> bool:
        """Alias for native inference dependency readiness."""
        return self.native_inference_ready


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return False


def inspect_capabilities(upstream_root: Path) -> BaselineCapabilities:
    """Inspect source, dependencies, CUDA, rasterizer, and corr_sampler availability."""

    required_source = (
        "LICENSE",
        "lib/network.py",
        "core/raft_stereo.py",
        "gaussian_renderer/__init__.py",
        "render.py",
    )
    source_available = all((upstream_root / relative).is_file() for relative in required_source)
    dependency_modules = ("cv2", "scipy", "torchvision", "yacs")
    missing = tuple(name for name in dependency_modules if not _module_available(name))
    python_deps_ok = len(missing) == 0
    rasterizer_available = _module_available("diff_gaussian_rasterization")
    corr_sampler_available = _module_available("corr_sampler")
    cuda_available = torch.cuda.is_available()

    renderer_ready = source_available and python_deps_ok and rasterizer_available
    native_inference_ready = (
        source_available and python_deps_ok and rasterizer_available and corr_sampler_available
    )
    cuda_baseline_runnable = native_inference_ready and cuda_available

    return BaselineCapabilities(
        upstream_source=source_available,
        python_dependencies=python_deps_ok,
        rasterizer=rasterizer_available,
        corr_sampler=corr_sampler_available,
        cuda=cuda_available,
        renderer_ready=renderer_ready,
        native_inference_ready=native_inference_ready,
        cuda_baseline_runnable=cuda_baseline_runnable,
        missing_python_dependencies=missing,
    )


def inspect_git_state(upstream_root: Path) -> UpstreamGitState:
    """Read the submodule revision and dirty state, failing closed on Git errors."""

    resolved_root = upstream_root.resolve()
    git_prefix = ["git", "-c", f"safe.directory={resolved_root.as_posix()}"]
    try:
        commit_result = subprocess.run(
            [*git_prefix, "rev-parse", "HEAD"],
            cwd=resolved_root,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            [*git_prefix, "status", "--porcelain"],
            cwd=resolved_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError) as error:
        raise UpstreamUnavailableError(
            f"unable to inspect upstream Git state at {upstream_root}: {error}"
        ) from error
    commit = commit_result.stdout.strip()
    if len(commit) != 40:
        raise UpstreamUnavailableError(f"upstream Git returned an invalid commit: {commit!r}")
    return UpstreamGitState(commit=commit, dirty=bool(status_result.stdout.strip()))


def require_pinned_upstream(upstream_root: Path, *, enforce_clean: bool = True) -> UpstreamGitState:
    """Reject a missing, changed, or optionally dirty upstream tree."""

    state = inspect_git_state(upstream_root)
    if state.commit != PINNED_COMMIT:
        raise UpstreamUnavailableError(
            f"upstream revision mismatch: expected {PINNED_COMMIT}, got {state.commit}"
        )
    if enforce_clean and state.dirty:
        raise UpstreamUnavailableError("upstream submodule is dirty")
    return state


@contextmanager
def _isolated_upstream_import_path(upstream_root: Path) -> Iterator[None]:
    """Temporarily satisfy upstream's repository-root absolute imports.

    The pinned project is not packaged and imports names such as ``core`` and
    ``lib`` from its repository root. The path change is local to this adapter
    boundary and is always reversed; no project scientific module mutates the
    import path.
    """

    root = str(upstream_root.resolve())
    already_present = root in sys.path
    if not already_present:
        sys.path.insert(0, root)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if not already_present:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


def import_upstream_module(
    module_name: str,
    upstream_root: Path,
    *,
    require_rasterizer: bool = False,
    require_corr_sampler: bool = False,
) -> ModuleType:
    """Import one verified upstream module only when baseline execution asks."""

    require_pinned_upstream(upstream_root)
    capabilities = inspect_capabilities(upstream_root)
    if not capabilities.python_dependencies:
        raise UpstreamUnavailableError(
            "upstream Python dependencies are unavailable: "
            + ", ".join(capabilities.missing_python_dependencies)
        )
    if require_rasterizer and not capabilities.rasterizer:
        raise UpstreamUnavailableError(
            "diff_gaussian_rasterization is unavailable; install the pinned baseline GPU environment"
        )
    if require_corr_sampler and not capabilities.corr_sampler:
        raise UpstreamUnavailableError(
            "corr_sampler is unavailable; install the compiled RAFT stereo sampler extension"
        )
    try:
        with _isolated_upstream_import_path(upstream_root):
            return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as error:
        raise UpstreamUnavailableError(
            f"unable to import upstream module {module_name!r}: {error}"
        ) from error
