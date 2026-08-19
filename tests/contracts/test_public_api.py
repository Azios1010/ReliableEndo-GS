"""Public export and dependency-boundary tests for contracts."""

import ast
from pathlib import Path

from reliable_endo_gs.contracts import (
    CameraBatch,
    GaussianField,
    ReconstructionState,
    RenderOutput,
    StereoBatch,
    StereoPrediction,
)


def test_public_contract_exports_are_available() -> None:
    assert [
        contract.SCHEMA_VERSION
        for contract in (
            CameraBatch,
            StereoBatch,
            StereoPrediction,
            GaussianField,
            RenderOutput,
            ReconstructionState,
        )
    ] == ["1.0"] * 6


def test_contract_modules_do_not_import_future_or_data_packages() -> None:
    contracts_root = Path("src/reliable_endo_gs/contracts")
    forbidden = {
        f"reliable_endo_gs.{package}"
        for package in (
            "data",
            "baseline",
            "uncertainty",
            "geometry",
            "rendering",
            "actions",
            "oracle",
            "routing",
        )
    }
    violations: list[str] = []

    for path in contracts_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module: str | None = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == prefix or alias.name.startswith(f"{prefix}.")
                        for prefix in forbidden
                    ):
                        violations.append(f"{path}:{alias.name}")
            if module is not None:
                if any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden):
                    violations.append(f"{path}:{module}")

    assert violations == []
