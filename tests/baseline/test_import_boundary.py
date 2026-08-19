"""Enforce third-party isolation and lightweight CPU import behavior."""

import ast
import subprocess
import sys
from pathlib import Path

from reliable_endo_gs.baseline.upstream import PINNED_COMMIT, inspect_git_state


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def test_no_project_module_outside_baseline_imports_upstream_internals() -> None:
    source_root = Path("src/reliable_endo_gs")
    forbidden_roots = {"core", "lib", "gaussian_renderer", "diff_gaussian_rasterization"}
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        if "baseline" in path.relative_to(source_root).parts:
            continue
        for module in _imported_modules(path):
            if module.split(".", maxsplit=1)[0] in forbidden_roots:
                violations.append(f"{path}:{module}")
    assert violations == []


def test_core_import_does_not_load_cuda_rasterizer() -> None:
    code = (
        "import sys; import reliable_endo_gs; import reliable_endo_gs.contracts; "
        "import reliable_endo_gs.data; "
        "assert 'diff_gaussian_rasterization' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_submodule_is_at_exact_clean_revision() -> None:
    state = inspect_git_state(Path("third_party/endo_e2e_gs"))
    assert state.commit == PINNED_COMMIT
    assert state.dirty is False
