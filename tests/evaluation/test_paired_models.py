"""Shared benchmark contracts on synthetic CPU data, not performance evidence."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from reliable_endo_gs.contracts import RenderOutput
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample, ScaredSupervision
from reliable_endo_gs.evaluation.paired_models import (
    FrozenModelPrediction,
    benchmark_frozen_predictor,
    compare_frozen_benchmarks,
    validate_development_worklist,
)
from tests.baseline.test_a00_model import fixture


def test_development_worklist_rejects_test_overlap_and_duplicates():
    sid = "dataset_7/keyframe_1/frame_data000000"
    validate_development_worklist((sid,), {"dataset_1/keyframe_1/frame_data000000"}, {sid})
    for ids, train in (
        ((sid, sid), set()),
        ((sid,), {sid}),
        (("dataset_8/keyframe_1/frame_data000000",), set()),
    ):
        with pytest.raises(ValueError):
            validate_development_worklist(ids, train, {sid, *ids})


def test_shared_metrics_gt_order_frozen_state_rng_and_paired_summary(tmp_path: Path):
    base, batch = fixture()
    ids = tuple(f"dataset_7/keyframe_{k}/frame_data000000" for k in (1, 2))
    events = []

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = base

        def forward(self, sample):
            assert sample.supervision is None
            events.append("prediction")
            out = self.model(sample.batch, with_gaussians=True)
            image = RenderOutput(sample.batch.left + 0.01)
            return FrozenModelPrediction(
                out.disparity, out.field, torch.arange(25)[None], image, image
            )

    class Adapter:
        manifest = SimpleNamespace(
            manifest_hash="synthetic",
            entries=[SimpleNamespace(sample_id=sid, data_role="final_train") for sid in ids],
        )

        def load(self, sid, *, include_supervision, **kwargs):
            assert not include_supervision
            return ScaredA00Sample(sid, "final_train", batch, {})

        def load_supervision(self, sample):
            assert events[-1] == "prediction"
            events.append("GT")
            return ScaredSupervision(torch.ones(1, 1, 5, 5) * 8, batch.support)

    predictor = Predictor()
    rng = torch.get_rng_state().clone()
    for name in ("baseline", "kd"):
        result = benchmark_frozen_predictor(
            predictor=predictor,
            adapter=Adapter(),
            sample_ids=ids,
            training_ids=set(),
            output_dir=tmp_path / name,
            output_hw=(5, 5),
            device=torch.device("cpu"),
            provenance={"checkpoint": name, "checkpoint_sha256": name},
            warmup=1,
            repetitions=3,
        )
        assert result["weights_unchanged"] and result["timing"]["repetitions"] == 3
        assert result["timing"]["unique_frames"] == 2
        assert result["perceptual"]["status"] == "unavailable"
        assert predictor.training
        assert torch.equal(rng, torch.get_rng_state())
    result = compare_frozen_benchmarks(
        tmp_path / "baseline", tmp_path / "kd", tmp_path / "comparison", repetitions=50
    )
    assert result["final_test_executed"] is False and result["superiority_claim"] is False
    assert result["metrics"]["left_psnr"]["KD_minus_baseline"] == 0
    assert result["metrics"]["left_psnr"]["ci95_cluster_keyframe"] == [0, 0]
    assert (tmp_path / "comparison/comparison.md").is_file()
    metadata = tmp_path / "kd/result.json"
    raw = json.loads(metadata.read_text())
    raw["sample_ids"].reverse()
    metadata.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="worklist"):
        compare_frozen_benchmarks(tmp_path / "baseline", tmp_path / "kd", tmp_path / "invalid")
