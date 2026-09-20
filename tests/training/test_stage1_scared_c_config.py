"""Focused Stage1 SCARED-C config and scratch-initialization checks."""

import pytest
import torch

from reliable_endo_gs.training.stage1_scared_c import (
    DEFAULT_CONFIG_PATH,
    REPOSITORY_ROOT,
    _parse_optional_checkpoint,
    build_stage1_scratch_model,
    load_stage1_scared_c_config,
)


def test_stage1_roles_and_null_checkpoint_are_explicit() -> None:
    config = load_stage1_scared_c_config(DEFAULT_CONFIG_PATH)

    assert config.dataset == "scared_c"
    assert config.mode == "corrected_video"
    assert config.train_sequences == ("1_1", "2_2", "3_2")
    assert config.validation_sequences == ("1_2", "3_1")
    assert config.from_scratch is True
    assert config.initialization == "scratch"
    assert config.restore_ckpt is None
    assert _parse_optional_checkpoint("None") is None
    assert _parse_optional_checkpoint("null") is None
    with pytest.raises(ValueError):
        config.require_safe_mode(steps=60000)


def test_scratch_model_construction_never_loads_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_stage1_scared_c_config(DEFAULT_CONFIG_PATH)
    load_calls: list[object] = []

    def fail_if_called(*args: object, **kwargs: object) -> object:
        load_calls.append((args, kwargs))
        raise AssertionError("scratch initialization attempted torch.load")

    monkeypatch.setattr(torch, "load", fail_if_called)
    model = build_stage1_scratch_model(config)

    assert isinstance(model, torch.nn.Module)
    assert load_calls == []
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_full_data_config_has_explicit_allowlist_and_temporal_split_contract() -> None:
    config = load_stage1_scared_c_config(
        REPOSITORY_ROOT / "configs" / "training" / "stage1_scared_c_full_v1.yaml"
    )

    assert config.is_full_data is True
    assert config.train_dataset_ids == ("dataset_1", "dataset_2", "dataset_3")
    assert config.train_keyframe_entries == ()
    assert config.train_sample_ids == ()
    assert config.validation_fraction == 0.05
    assert config.validation_block_policy == "tail"
    assert config.validation_rounding == "nearest_half_up"
    assert config.batch_size == 1
    assert config.num_steps == 60000
    assert config.expected_total_samples == 8416
    assert config.expected_train_samples == 7994
    assert config.expected_validation_samples == 422
    config.require_safe_mode(steps=60000)
