"""Regression checks for usable-GT selection in the actual smoke entry point."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from reliable_endo_gs.training import stage1_scared_c as training


@pytest.mark.parametrize("steps", [3, 12])
def test_full_data_smoke_uses_audited_frames_and_runs_every_update(
    monkeypatch: pytest.MonkeyPatch, steps: int
) -> None:
    config = training.load_stage1_scared_c_config(
        training.REPOSITORY_ROOT / "configs/training/stage1_scared_c_full_v1.yaml"
    )
    original_ids = ("bad", "good_1", "good_2", "good_3", "good_4")
    retained_ids = original_ids[1:]
    split = SimpleNamespace(
        train_sample_ids=original_ids, source_sample_ids=original_ids, manifest_sha256="source"
    )
    quality = SimpleNamespace(train_sample_ids=retained_ids)
    monkeypatch.setattr(training, "load_stage1_frame_split", lambda _path: split)
    monkeypatch.setattr(training, "_load_gt_quality", lambda _config, _split: quality)
    seen: list[str] = []
    closed: list[bool] = []

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, _config: object, *, sample_ids: tuple[str, ...], **kwargs: object):
            assert sample_ids == retained_ids
            assert kwargs["cache_expected_sample_ids"] == original_ids
            self.ids = sample_ids

        def __len__(self) -> int:
            return len(self.ids)

        def __getitem__(self, index: int) -> dict[str, object]:
            return {
                "left": torch.ones(3, 2, 2),
                "right": torch.ones(3, 2, 2),
                "disparity": torch.ones(1, 2, 2),
                "mask": torch.ones(1, 2, 2).bool(),
                "disp_const": torch.tensor(1.0),
                "sample_id": self.ids[index],
            }

        def close(self) -> None:
            closed.append(True)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))
            self.raft_stereo = SimpleNamespace(freeze_bn=lambda: None)

        def forward(self, batch: dict[str, object], is_train: bool):
            assert is_train
            seen.extend(batch["sample_id"])
            return batch, (self.weight - 1.0).square(), {}

    monkeypatch.setattr(training, "Stage1ScaredCCachedDataset", Dataset)
    monkeypatch.setattr(training, "build_stage1_scratch_model", lambda _config: Model())
    result = training.run_stage1_scratch_smoke(config, device=torch.device("cpu"), steps=steps)
    assert result.steps == len(result.losses) == len(seen) == steps
    assert set(seen) <= set(retained_ids)
    assert result.losses[-1] < result.losses[0]
    assert closed == [True]
    assert result.checkpoint_loaded is False


@pytest.mark.parametrize("steps", [0, 101, 60000])
def test_full_data_smoke_remains_bounded(steps: int) -> None:
    config = training.load_stage1_scared_c_config(
        training.REPOSITORY_ROOT / "configs/training/stage1_scared_c_full_v1.yaml"
    )
    with pytest.raises(training.Stage1ScaredCConfigError, match="1..100"):
        training.run_stage1_scratch_smoke(config, device=torch.device("cpu"), steps=steps)


def test_reject_policy_does_not_load_an_exclusion_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(training.load_stage1_scared_c_config(), gt_quality_policy="reject")

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("reject policy read an exclusion manifest")

    monkeypatch.setattr(training, "load_stage1_gt_quality_manifest", unexpected)
    assert training._load_gt_quality(config, object()) is None
