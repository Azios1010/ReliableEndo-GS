"""Repair bounds and latency penalties remain part of portable config identity."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from reliable_endo_gs.config.intervention import InterventionConfig, load_intervention_config


@pytest.mark.parametrize("schema", ["intervention_selection.v1", "intervention_selection.v2"])
@pytest.mark.parametrize(
    "settings",
    [
        {"repair_max_disparity_px": 2.0},
        {"cost_weight_per_ms": 0.01},
        {"repair_max_disparity_px": 2.0, "cost_weight_per_ms": 0.01},
    ],
)
def test_repair_and_cost_settings_roundtrip_and_change_identity(schema, settings, tmp_path):
    neutral = InterventionConfig(schema=schema)
    configured = replace(neutral, **settings)
    serialized = configured.to_dict()
    assert settings.items() <= serialized.items()
    assert InterventionConfig(**serialized) == configured
    assert configured.config_hash != neutral.config_hash

    path = tmp_path / "intervention.yaml"
    path.write_text(yaml.safe_dump(serialized), encoding="utf-8")
    loaded = load_intervention_config(path)
    assert loaded == configured
    assert loaded.config_hash == configured.config_hash


def test_neutral_repair_and_cost_settings_keep_historical_v1_identity():
    config = load_intervention_config(Path("configs/routing/intervention_selection.yaml"))
    assert config.config_hash == "3db4be52b6f227b6cbbaf79564510c2f6162f4db7109b231842c75e7a543964f"
    assert "repair_max_disparity_px" not in config.to_dict()
    assert "cost_weight_per_ms" not in config.to_dict()


@pytest.mark.parametrize("schema", ["intervention_selection.v1", "intervention_selection.v2"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("repair_max_disparity_px", True),
        ("repair_max_disparity_px", "2"),
        ("repair_max_disparity_px", 0.0),
        ("repair_max_disparity_px", -1.0),
        ("repair_max_disparity_px", float("nan")),
        ("repair_max_disparity_px", float("inf")),
        ("cost_weight_per_ms", None),
        ("cost_weight_per_ms", True),
        ("cost_weight_per_ms", "0.01"),
        ("cost_weight_per_ms", -0.01),
        ("cost_weight_per_ms", float("nan")),
        ("cost_weight_per_ms", float("inf")),
    ],
)
def test_invalid_repair_and_cost_settings_are_rejected(schema, field, value, tmp_path):
    with pytest.raises(ValueError, match=field):
        InterventionConfig(schema=schema, **{field: value})
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump({"schema": schema, field: value}), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        load_intervention_config(path)


def test_v1_still_rejects_balanced_only_settings(tmp_path):
    path = tmp_path / "balanced.yaml"
    path.write_text(
        "schema: intervention_selection.v1\nnormalization_frames: 64\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="require intervention_selection.v2"):
        load_intervention_config(path)
