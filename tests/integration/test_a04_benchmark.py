"""CPU fake-teacher benchmark, GT isolation, failures and strict resume."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.config.deployment import DeploymentRuntimeProfile
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, build_a01_manifest
from reliable_endo_gs.evaluation import a04_benchmark
from reliable_endo_gs.evaluation.stereo import MetricResult
from reliable_endo_gs.reconstruction import (
    ReferenceConfig,
    ReferenceReconstructionModel,
    save_reference_checkpoint,
)
from reliable_endo_gs.rendering.cross_view import ReferenceRenderer
from reliable_endo_gs.rendering.field import ExistingRendererBridge
from tests.integration.test_a01_a04_server_smoke import _fixture, _teacher


class FakePerceptual(torch.nn.Module):
    """Synthetic metric only; the production LPIPS constructor is separately verified."""

    def __init__(self, *_: object) -> None:
        super().__init__()
        self.identity = {"backend": "synthetic_test_only"}
        self.calls = 0

    def measure(self, x: torch.Tensor, y: torch.Tensor) -> MetricResult:
        self.calls += 1
        return MetricResult("lpips_alex", float((x - y).abs().mean()), 1, 1, "synthetic")


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    root = _fixture(tmp_path / "scared")
    (root / "dataset_1").rename(root / "dataset_8")
    data = root / "dataset_8/keyframe_1/data"
    for directory in (
        "left_finalpass",
        "right_finalpass",
        "frame_data",
        "newpram_data",
        "disparity",
    ):
        old = next((data / directory).iterdir())
        (old.parent / old.name.replace("000000", "000001")).write_bytes(old.read_bytes())
    manifest_path = tmp_path / "manifest.json"
    manifest = build_a01_manifest(root, manifest_path, require_all_folds=False)
    adapter = ScaredA00Adapter(root, manifest)
    teacher = _teacher()
    renderer = ExistingRendererBridge(ReferenceRenderer())
    config = replace(
        ReferenceConfig(),
        runtime=replace(
            DeploymentRuntimeProfile(), device="cpu", resolution_hw=(32, 40), precision="bf16"
        ),
        teacher_identity=teacher.identity.to_dict(),
        renderer_identity=renderer.identity,
        head_hidden_dims=(4, 4),
        assessment_views=("left", "right"),
    )
    model = ReferenceReconstructionModel(config=config, teacher=teacher, renderer=renderer)
    sample = adapter.load(
        manifest.entries[0].sample_id,
        data_role="final_test",
        include_supervision=False,
        output_hw=(32, 40),
    )
    with torch.inference_mode():
        result = model(
            sample.batch,
            data_role="final_test",
            sample_ids=(sample.sample_id,),
            calibration_metadata=sample.calibration_metadata,
        )
    checkpoint = tmp_path / "head.pt"
    save_reference_checkpoint(checkpoint, model, provenance=result.provenance_record)
    monkeypatch.setattr(a04_benchmark, "build_foundation_stereo", lambda: teacher)
    monkeypatch.setattr(a04_benchmark, "ProductionFieldRenderer", lambda: renderer)
    monkeypatch.setattr(a04_benchmark, "LPIPSAlexNet", FakePerceptual)
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        "warmup: 0\nprofile_repetitions: 1\ncloud_max_points: 16\ncloud_chunk_size: 8\n"
    )
    repo = Path(__file__).resolve().parents[2]
    return dict(
        checkpoint=checkpoint,
        data_root=root,
        manifest=manifest_path,
        roles_path=repo / "configs/teachers/a01_scared.yaml",
        protocol_path=protocol,
        output_dir=tmp_path / "results",
        lpips_backbone=tmp_path / "unused",
        lpips_calibration=tmp_path / "unused2",
    )


def test_benchmark_frozen_state_separate_gt_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _setup(tmp_path, monkeypatch)
    loads = []
    original = ScaredA00Adapter.load

    def spy(self, *pos, **kwargs):
        sample = original(self, *pos, **kwargs)
        loads.append((kwargs.get("include_supervision"), sample.supervision is not None))
        return sample

    monkeypatch.setattr(ScaredA00Adapter, "load", spy)
    checkpoint_before = args["checkpoint"].read_bytes()
    summary = a04_benchmark.run_a04_benchmark(**args)
    assert summary["sample_count"] == 2 and summary["complete_final_test"]
    assert summary["model_state_unchanged"] and summary["checkpoint_unchanged"]
    assert summary["optimizer_updates"] == 0
    assert checkpoint_before == args["checkpoint"].read_bytes()
    assert loads == [(False, False), (False, False), (True, True), (False, False), (True, True)]
    assert summary["overall"]["left_ssim_windowed"]["available_frames"] == 2
    # This synthetic renderer exports metric depth, unlike the production backend.
    assert summary["overall"]["rendered_depth_mae_mm"]["available_frames"] == 2
    assert (args["output_dir"] / "identity.json").exists()
    assert (args["output_dir"] / "profile.json").exists()
    assert (args["output_dir"] / "metrics.csv").exists()
    with pytest.raises(ValueError, match="overwrite"):
        a04_benchmark.run_a04_benchmark(**args)
    with pytest.raises(ValueError, match="incomplete"):
        a04_benchmark.run_a04_benchmark(**args, resume=True)


def test_failure_preserved_and_resume_consumes_only_remaining_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _setup(tmp_path, monkeypatch)

    class FailSecondFrame(FakePerceptual):
        def measure(self, x, y):
            if self.calls >= 2:
                raise RuntimeError("injected evaluation failure")
            return super().measure(x, y)

    monkeypatch.setattr(a04_benchmark, "LPIPSAlexNet", FailSecondFrame)
    with pytest.raises(RuntimeError, match="injected evaluation failure"):
        a04_benchmark.run_a04_benchmark(**args)
    out = args["output_dir"]
    first_rows = (out / "samples.jsonl").read_text()
    assert len(first_rows.splitlines()) == 1 and not (out / "summary.json").exists()
    assert len(list(out.glob("error-*.json"))) == 1
    monkeypatch.setattr(a04_benchmark, "LPIPSAlexNet", FakePerceptual)
    summary = a04_benchmark.run_a04_benchmark(**args, resume=True)
    assert summary["sample_count"] == 2
    assert (out / "samples.jsonl").read_text().startswith(first_rows)
    assert len((out / "samples.jsonl").read_text().splitlines()) == 2
    assert len(list(out.glob("error-*.json"))) == 1
