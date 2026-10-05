"""Keep CPU CI reproducible without changing the scientific runtime."""

from pathlib import Path

import yaml


def _workflow() -> dict:
    root = Path(__file__).resolve().parents[2]
    return yaml.safe_load((root / ".github/workflows/ci-cpu.yml").read_text())


def _constraints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    result = {}
    for line in (root / "environments/ci-constraints.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, separator, version = line.partition("==")
            assert separator and name and version
            assert name not in result
            result[name] = version
    return result


def test_cpu_install_and_constraints_use_the_same_torch_pair() -> None:
    steps = _workflow()["jobs"]["test"]["steps"]
    install = next(step["run"] for step in steps if step["name"] == "Install project")
    constraints = _constraints()
    assert f"torch=={constraints['torch']}" in install
    assert f"torchvision=={constraints['torchvision']}" in install
    assert "--index-url https://download.pytorch.org/whl/cpu" in install
    assert "--constraint environments/ci-constraints.txt" in install
    assert install.index("/whl/cpu") < install.index('install -e ".[dev]"')


def test_ci_checks_cpu_build_and_preserves_pinned_upstream() -> None:
    workflow = _workflow()
    assert workflow["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    steps = workflow["jobs"]["test"]["steps"]
    verification = next(
        step["run"] for step in steps if step["name"] == "Record and verify CPU environment"
    )
    assert "python -m pip check" in verification
    assert "python -m pip freeze" in verification
    assert "assert torch.version.cuda is None" in verification


def test_ci_keeps_all_existing_quality_checks() -> None:
    steps = _workflow()["jobs"]["test"]["steps"]
    commands = {step.get("run", "") for step in steps}
    assert {"ruff check .", "ruff format --check .", "mypy src", "pytest -q"} <= commands
    assert not any(step.get("continue-on-error", False) for step in steps)
    assert {"ruff", "mypy", "pytest", "torch", "torchvision", "omegaconf"} <= set(_constraints())
