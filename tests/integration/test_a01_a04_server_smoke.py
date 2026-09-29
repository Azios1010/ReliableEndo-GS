"""Bounded fake-teacher A01 cache and A04 CPU end-to-end smoke."""

from dataclasses import replace
import json
from pathlib import Path

import torch
import numpy as np
import tifffile
from PIL import Image

from reliable_endo_gs.config.deployment import DeploymentRuntimeProfile
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, build_a01_manifest
from reliable_endo_gs.reconstruction import ReferenceConfig
from reliable_endo_gs.rendering.cross_view import ReferenceRenderer
from reliable_endo_gs.rendering.field import ExistingRendererBridge
from reliable_endo_gs.teachers import (
    TargetCache,
    TeacherPrediction,
    make_foundation_stereo_candidate,
)
from reliable_endo_gs.teachers.workflow import collect_targets, validate_cache
from reliable_endo_gs.training.a04_reference import A04ReferenceTrainer



def _fixture(root: Path) -> Path:
    data = root / "dataset_1" / "keyframe_1" / "data"
    for name in ("frame_data", "left_finalpass", "right_finalpass", "disparity", "newpram_data"):
        (data / name).mkdir(parents=True, exist_ok=True)
    frame_id = "frame_data000000"
    Image.fromarray(np.full((4, 5, 3), 128, dtype=np.uint8)).save(data / "left_finalpass" / f"{frame_id}.png")
    Image.fromarray(np.full((4, 5, 3), 64, dtype=np.uint8)).save(data / "right_finalpass" / f"{frame_id}.png")
    tifffile.imwrite(data / "disparity" / f"{frame_id}.tiff", np.full((4, 5), 8.0, dtype=np.float32))
    (data / "frame_data" / f"{frame_id}.json").write_text("{}", encoding="utf-8")
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = -0.1
    (data / "newpram_data" / f"{frame_id}.json").write_text(
        json.dumps(
            {
                "intr0": [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
                "intr1": [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
                "T_R_from_L": transform.tolist(),
            }
        ),
        encoding="utf-8",
    )
    (data / "a00_protocol.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_range": "uint8_0_255",
                "rectified": True,
                "distortion_removed": True,
                "pixel_center": "integer",
                "disparity": {"units": "pixels", "sign": "u_left_minus_right", "reference": "left"},
                "extrinsics": {"source": "newpram.T_R_from_L", "direction": "right_from_left", "units": "m"},
                "preprocessing": {"images_grid": "processed"},
            }
        ),
        encoding="utf-8",
    )
    return root


def _teacher():
    def infer(_model: object, left: torch.Tensor, right: torch.Tensor) -> TeacherPrediction:
        disparity = torch.full(
            (left.shape[0], 1, left.shape[2], left.shape[3]),
            8.0,
            dtype=left.dtype,
            device=left.device,
        )
        return TeacherPrediction(disparity)

    return make_foundation_stereo_candidate(
        loader=lambda: object(),
        infer_fn=infer,
        api_version="fake-a01-v1",
        checkpoint_id="fake-checkpoint",
        checkpoint_sha256="a" * 64,
        license_name="test-license",
        preprocessing={"status": "verified", "image_range": "0..1"},
        output={
            "status": "verified",
            "disparity_units": "teacher_grid_pixels",
            "disparity_sign": "u_left_minus_right",
        },
    )


def test_fake_teacher_cache_resume_and_a04_cpu_smoke(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "scared")
    manifest = build_a01_manifest(root, tmp_path / "a01.json", require_all_folds=False)
    adapter = ScaredA00Adapter(root, manifest)
    teacher = _teacher()
    cache = TargetCache(tmp_path / "cache")
    first = collect_targets(adapter, teacher, cache, split_identity="test-a01-v1")
    second = collect_targets(adapter, teacher, TargetCache(tmp_path / "cache"), split_identity="test-a01-v1")
    assert first.computed == 1 and first.cache_hits == 0
    assert second.computed == 0 and second.cache_hits == 1
    assert validate_cache(TargetCache(tmp_path / "cache"))["valid"] == 1

    runtime = replace(DeploymentRuntimeProfile(), device="cpu", resolution_hw=(4, 5), deadline_ms=100.0)
    config = replace(
        ReferenceConfig(),
        teacher_identity=teacher.identity.to_dict(),
        runtime=runtime,
        head_hidden_dims=(4, 4),
        assessment_views=("left", "right"),
    )
    renderer = ExistingRendererBridge(ReferenceRenderer())
    trainer = A04ReferenceTrainer(
        config=config,
        adapter=adapter,
        teacher=teacher,
        renderer=renderer,
        cache=TargetCache(tmp_path / "cache"),
        device="cpu",
    )
    smoke = trainer.smoke()
    assert smoke.completed_steps == 1
    train = trainer.train(max_steps=1)
    assert train.completed_steps == 1
    checkpoint = trainer.save(tmp_path / "a04.pt")
    assert checkpoint.is_file()
    resumed = A04ReferenceTrainer(
        config=config,
        adapter=adapter,
        teacher=teacher,
        renderer=ExistingRendererBridge(ReferenceRenderer()),
        cache=TargetCache(tmp_path / "cache"),
        device="cpu",
    )
    resumed.resume(checkpoint)
    assert resumed.optimizer.state_dict()["state"]
