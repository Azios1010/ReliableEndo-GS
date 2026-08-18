"""Shared fixtures for infrastructure tests."""

import pytest


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
