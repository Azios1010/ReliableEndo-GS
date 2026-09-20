"""Shared fixtures for infrastructure tests."""

import pytest


@pytest.fixture(autouse=True)
def stage1_runtime_roots(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide portable runtime roots for tests that parse Stage-1 configs."""

    monkeypatch.setenv("RELIABLE_ENDO_DATA_ROOT", str(tmp_path / "datasets"))
    monkeypatch.setenv("RELIABLE_ENDO_STAGE1_FRAME_SPLIT_ROOT", str(tmp_path / "stage1"))


@pytest.fixture
def valid_config_data() -> dict[str, object]:
    """Return a minimal valid infrastructure configuration mapping."""

    return {
        "experiment": {
            "name": "infrastructure_smoke",
            "seed": 42,
            "output_root": "outputs",
        },
        "runtime": {"device": "cpu"},
    }
