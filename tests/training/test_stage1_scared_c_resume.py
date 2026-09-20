"""Contract checks for explicit Stage1 checkpoint resume."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from reliable_endo_gs.training.stage1_scared_c import (
    REPOSITORY_ROOT,
    _load_stage1_resume_checkpoint,
    _resolved_config_payload,
    _restore_stage1_resume_state,
    load_stage1_scared_c_config,
)


def _resume_fixture(tmp_path):
    config = load_stage1_scared_c_config(
        REPOSITORY_ROOT / "configs/training/stage1_scared_c_full_v1.yaml"
    )
    config = replace(config, num_steps=4, validation_steps=(2, 4), scheduler_pct_start=0.5)
    split = SimpleNamespace(manifest_sha256="split-sha", train_count=2, validation_count=1)
    quality = SimpleNamespace(
        manifest_sha256="quality-sha",
        train_sample_ids=("train-1", "train-2"),
        validation_sample_ids=("validation-1",),
    )

    torch.manual_seed(7)
    source_model = torch.nn.Linear(2, 1)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=config.learning_rate)
    source_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        source_optimizer,
        config.learning_rate,
        config.num_steps,
        pct_start=config.scheduler_pct_start,
        cycle_momentum=False,
        anneal_strategy=config.scheduler_anneal_strategy,
    )
    for _ in range(2):
        source_optimizer.zero_grad(set_to_none=True)
        source_model(torch.ones(1, 2)).square().mean().backward()
        source_optimizer.step()
        source_scheduler.step()

    checkpoint = tmp_path / "checkpoints" / "best_current.pth"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "total_steps": 2,
            "network": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "scheduler": source_scheduler.state_dict(),
            "stage1_scared_c": {
                "source_sha": "source-sha",
                "config": _resolved_config_payload(config),
                "frame_split_manifest_sha256": "split-sha",
                "gt_quality_manifest_sha256": "quality-sha",
                "train_sample_count": 2,
                "validation_sample_count": 1,
                "selection_row": {
                    "step": 2,
                    "sequence": "MACRO",
                    "epe": 1.5,
                    "median_ae": 1.0,
                    "bias": -0.2,
                    "bad3": 0.1,
                    "bad5": 0.05,
                    "frames": 1,
                    "valid_pixels": 4,
                },
            },
        },
        checkpoint,
    )
    return config, split, quality, checkpoint, source_model


def test_resume_checkpoint_restores_model_optimizer_and_scheduler(tmp_path) -> None:
    config, split, quality, checkpoint, source_model = _resume_fixture(tmp_path)
    state, payload = _load_stage1_resume_checkpoint(
        checkpoint,
        config=config,
        split=split,
        quality=quality,
    )

    target_model = torch.nn.Linear(2, 1)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=config.learning_rate)
    target_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        target_optimizer,
        config.learning_rate,
        config.num_steps,
        pct_start=config.scheduler_pct_start,
        cycle_momentum=False,
        anneal_strategy=config.scheduler_anneal_strategy,
    )
    _restore_stage1_resume_state(
        payload,
        model=target_model,
        optimizer=target_optimizer,
        scheduler=target_scheduler,
        expected_step=state.start_step,
        device=torch.device("cpu"),
    )

    assert state.start_step == state.best_step == 2
    assert state.best_validation_epe == pytest.approx(1.5)
    assert state.best_abs_bias == pytest.approx(0.2)
    assert target_scheduler.last_epoch == 2
    assert target_optimizer.state
    for expected, actual in zip(source_model.parameters(), target_model.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_resume_checkpoint_rejects_a_different_split(tmp_path) -> None:
    config, split, quality, checkpoint, _source_model = _resume_fixture(tmp_path)
    mismatched_split = SimpleNamespace(
        manifest_sha256="different-split",
        train_count=split.train_count,
        validation_count=split.validation_count,
    )

    with pytest.raises(ValueError, match="frame split manifest"):
        _load_stage1_resume_checkpoint(
            checkpoint,
            config=config,
            split=mismatched_split,
            quality=quality,
        )
