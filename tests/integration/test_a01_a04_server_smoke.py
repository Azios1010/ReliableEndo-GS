"""Bounded fake-teacher A01 cache and A04 CPU end-to-end smoke."""

import json
import random
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from reliable_endo_gs.config.deployment import DeploymentRuntimeProfile
from reliable_endo_gs.contracts.rendering import RenderOutput
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
from reliable_endo_gs.training.a04_reference import A04ReferenceTrainer, A04TrainingError
from reliable_endo_gs.training.harness import TrainingLog, capture_rng


def _fixture(root: Path) -> Path:
    data = root / "dataset_1" / "keyframe_1" / "data"
    for name in ("frame_data", "left_finalpass", "right_finalpass", "disparity", "newpram_data"):
        (data / name).mkdir(parents=True, exist_ok=True)
    frame_id = "frame_data000000"
    Image.fromarray(np.full((4, 5, 3), 128, dtype=np.uint8)).save(
        data / "left_finalpass" / f"{frame_id}.png"
    )
    Image.fromarray(np.full((4, 5, 3), 64, dtype=np.uint8)).save(
        data / "right_finalpass" / f"{frame_id}.png"
    )
    tifffile.imwrite(
        data / "disparity" / f"{frame_id}.tiff", np.full((4, 5), 8.0, dtype=np.float32)
    )
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
                "extrinsics": {
                    "source": "newpram.T_R_from_L",
                    "direction": "right_from_left",
                    "units": "m",
                },
                "preprocessing": {"images_grid": "processed"},
            }
        ),
        encoding="utf-8",
    )
    return root


def _teacher(loader=None, infer_fn=None):
    def infer(_model: object, left: torch.Tensor, right: torch.Tensor) -> TeacherPrediction:
        disparity = torch.full(
            (left.shape[0], 1, left.shape[2], left.shape[3]),
            8.0,
            dtype=left.dtype,
            device=left.device,
        )
        return TeacherPrediction(disparity)

    return make_foundation_stereo_candidate(
        loader=loader or (lambda: object()),
        infer_fn=infer_fn or infer,
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
    second = collect_targets(
        adapter, teacher, TargetCache(tmp_path / "cache"), split_identity="test-a01-v1"
    )
    assert first.computed == 1 and first.cache_hits == 0
    assert second.computed == 0 and second.cache_hits == 1
    assert validate_cache(TargetCache(tmp_path / "cache"))["valid"] == 1

    runtime = replace(
        DeploymentRuntimeProfile(), device="cpu", resolution_hw=(4, 5), deadline_ms=100.0
    )
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
    with pytest.raises(A04TrainingError, match="restricted to the final_test role"):
        trainer.evaluate()
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

    final_test_trainer = A04ReferenceTrainer(
        config=replace(config, data_role="final_test"),
        adapter=adapter,
        teacher=teacher,
        renderer=ExistingRendererBridge(ReferenceRenderer()),
        cache=TargetCache(tmp_path / "cache"),
        device="cpu",
    )
    with pytest.raises(A04TrainingError, match="restricted to the final_train role"):
        final_test_trainer.train(max_steps=1)


@pytest.mark.parametrize("visible", [False, True])
def test_a04_rejects_empty_visibility_but_accepts_visible_black_scene(
    tmp_path: Path, visible: bool
) -> None:
    class BlackRenderer:
        identity = "test-black-renderer"

        def render(self, request):
            field = request.field
            height, width = request.image_hw
            return RenderOutput(
                image=torch.zeros(field.means3d.shape[0], 3, height, width),
                visibility=torch.full_like(field.valid_mask, visible),
            )

    root = _fixture(tmp_path / "scared")
    manifest = build_a01_manifest(root, tmp_path / "a01.json", require_all_folds=False)
    runtime = replace(DeploymentRuntimeProfile(), device="cpu", resolution_hw=(4, 5))
    trainer = A04ReferenceTrainer(
        config=replace(ReferenceConfig(), runtime=runtime, head_hidden_dims=(4, 4)),
        adapter=ScaredA00Adapter(root, manifest),
        teacher=_teacher(),
        renderer=BlackRenderer(),
        device="cpu",
    )
    if visible:
        assert trainer.smoke().completed_steps == 1
    else:
        with pytest.raises(A04TrainingError, match="no visible primitives"):
            trainer.smoke()
        with pytest.raises(A04TrainingError, match="no visible primitives"):
            trainer.train(max_steps=1)
        assert not trainer.optimizer.state_dict()["state"]


def _multi_trainer(root: Path, count: int = 3, teacher=None, renderer=None) -> A04ReferenceTrainer:
    _fixture(root)
    data = root / "dataset_1/keyframe_1/data"
    for index in range(1, count):
        for directory in ("frame_data", "left_finalpass", "right_finalpass", "newpram_data"):
            source = next((data / directory).glob("frame_data000000.*"))
            shutil.copyfile(source, source.with_name(f"frame_data{index:06d}{source.suffix}"))
        Image.fromarray(np.full((4, 5, 3), 128 + index, dtype=np.uint8)).save(
            data / "left_finalpass" / f"frame_data{index:06d}.png"
        )
    manifest = build_a01_manifest(root, root.parent / f"{root.name}.json", require_all_folds=False)
    runtime = replace(DeploymentRuntimeProfile(), device="cpu", resolution_hw=(4, 5))
    return A04ReferenceTrainer(
        config=replace(ReferenceConfig(), runtime=runtime, head_hidden_dims=(4, 4)),
        adapter=ScaredA00Adapter(root, manifest),
        teacher=teacher or _teacher(),
        renderer=renderer or ExistingRendererBridge(ReferenceRenderer()),
        device="cpu",
    )


def _stochastic_forward(trainer: A04ReferenceTrainer, monkeypatch) -> None:
    forward = trainer._forward_step

    def wrapped(sample):
        result, loss, count, source = forward(sample)
        factor = torch.rand(()) + random.random() + float(np.random.random())
        return result, None if loss is None else loss * factor, count, source

    monkeypatch.setattr(trainer, "_forward_step", wrapped)


def _assert_tree_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def test_a04_resume_matches_uninterrupted_order_weights_optimizer_rng(
    tmp_path, monkeypatch
) -> None:
    def cold_teacher():
        def load():
            torch.rand(5)
            np.random.random(5)
            random.random()
            return object()

        return _teacher(load)

    trainer = _multi_trainer(tmp_path / "data", teacher=cold_teacher())
    _stochastic_forward(trainer, monkeypatch)
    full = trainer.train(epochs=2)
    expected_weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    expected_optimizer = trainer.optimizer.state_dict()
    expected_rng = capture_rng(trainer.device)

    first = _multi_trainer(tmp_path / "data", teacher=cold_teacher())
    _stochastic_forward(first, monkeypatch)
    events = []
    partial = first.train(
        epochs=2,
        max_steps=2,
        log_every=1,
        checkpoint_every=1,
        checkpoint_dir=tmp_path / "periodic",
        progress=events.append,
    )
    assert first.training_progress["next_sample_index"] == 2
    assert len(list((tmp_path / "periodic").glob("*.pt"))) == 2
    checkpoint = first.save(tmp_path / "stop.pt")
    assert first.model.training
    assert [event["event"] for event in events].count("checkpoint_saved") == 2
    _assert_tree_equal(
        capture_rng(first.device),
        torch.load(checkpoint, weights_only=True)["training_state"]["rng"],
    )

    resumed = _multi_trainer(tmp_path / "data", teacher=cold_teacher())
    _stochastic_forward(resumed, monkeypatch)
    resumed.resume(checkpoint)
    loaded_ids = []
    load = resumed.adapter.load

    def checked_load(sample_id, **kwargs):
        assert kwargs["include_supervision"] is False
        loaded_ids.append(sample_id)
        return load(sample_id, **kwargs)

    monkeypatch.setattr(resumed.adapter, "load", checked_load)
    remaining = resumed.train(epochs=2)
    assert [s.sample_id for s in partial.steps + remaining.steps] == [
        s.sample_id for s in full.steps
    ]
    assert loaded_ids[0].endswith("000002")
    assert remaining.completed_steps == 4
    assert resumed.training_progress["epoch"] == 2
    assert resumed.training_progress["next_sample_index"] == 0
    _assert_tree_equal(resumed.model.state_dict(), expected_weights)
    _assert_tree_equal(resumed.optimizer.state_dict(), expected_optimizer)
    _assert_tree_equal(capture_rng(resumed.device), expected_rng)
    with pytest.raises(A04TrainingError, match="epoch budget already completed"):
        resumed.train(epochs=2)


def test_a04_epoch_boundary_resume_and_save_do_not_reinfer(tmp_path, monkeypatch) -> None:
    trainer = _multi_trainer(tmp_path / "data")
    trainer.train(epochs=2, max_steps=3)
    assert trainer.training_progress["epoch"] == 1
    checkpoint = trainer.save(tmp_path / "boundary.pt")
    resumed = _multi_trainer(tmp_path / "data")
    resumed.resume(checkpoint)
    step = resumed.train(epochs=2, max_steps=1)
    assert step.steps[0].sample_id.endswith("000000")
    assert resumed.training_progress["optimizer_steps"] == 4

    def forbidden(*args):
        raise AssertionError("save must use stored provenance without teacher inference")

    monkeypatch.setattr(resumed, "_forward_step", forbidden)
    rng = capture_rng(resumed.device)
    resumed.save(tmp_path / "no-inference.pt")
    _assert_tree_equal(capture_rng(resumed.device), rng)
    assert resumed.model.training
    with pytest.raises(A04TrainingError, match="overwrite"):
        resumed.save(tmp_path / "no-inference.pt")


def test_a04_resume_rejects_manifest_and_legacy_cursor(tmp_path) -> None:
    trainer = _multi_trainer(tmp_path / "data")
    trainer.train(max_steps=1)
    checkpoint = trainer.save(tmp_path / "modern.pt")
    changed = _multi_trainer(tmp_path / "data")
    changed.adapter.manifest = replace(
        changed.adapter.manifest, entries=tuple(reversed(changed.adapter.manifest.entries))
    )
    with pytest.raises(A04TrainingError, match="manifest/order/role"):
        changed.resume(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    payload["training_state"] = {"optimizer_steps": 1}
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    resumed = _multi_trainer(tmp_path / "data")
    with pytest.raises(A04TrainingError, match="legacy checkpoints"):
        resumed.resume(legacy)
    resumed.resume(legacy, restore_training=False)
    _assert_tree_equal(trainer.model.state_dict(), resumed.model.state_dict())


def test_a04_empty_loss_advances_cursor_and_failure_keeps_checkpoint(tmp_path, monkeypatch) -> None:
    trainer = _multi_trainer(tmp_path / "data")
    forward = trainer._forward_step

    def wrapped(sample):
        if sample.sample_id.endswith("000002"):
            raise A04TrainingError("injected failure")
        result, loss, count, source = forward(sample)
        if sample.sample_id.endswith("000001"):
            return result, None, 0, source
        return result, loss, count, source

    monkeypatch.setattr(trainer, "_forward_step", wrapped)
    with pytest.raises(A04TrainingError, match="injected failure"):
        trainer.train(checkpoint_every=1, checkpoint_dir=tmp_path / "periodic")
    assert trainer.training_progress["next_sample_index"] == 2
    assert trainer.training_progress["optimizer_steps"] == 1
    checkpoint = trainer.save(tmp_path / "after-skip.pt")
    resumed = _multi_trainer(tmp_path / "data")
    resumed.resume(checkpoint)
    remaining = resumed.train(max_steps=1)
    assert remaining.steps[0].sample_id.endswith("000002")
    assert (tmp_path / "periodic/step-000000001.pt").is_file()


def test_a04_amp_overflow_is_not_counted_as_optimizer_update(tmp_path, monkeypatch) -> None:
    class OverflowScaler:
        value = 16.0
        calls = 0

        def is_enabled(self):
            return True

        def get_scale(self):
            return self.value

        def scale(self, loss):
            return loss

        def step(self, optimizer):
            if self.calls:
                optimizer.step()

        def update(self):
            if self.calls == 0:
                self.value /= 2
            self.calls += 1

    trainer = _multi_trainer(tmp_path / "data")
    monkeypatch.setattr(trainer, "scaler", OverflowScaler())
    events = []
    result = trainer.train(progress=events.append)
    assert result.completed_steps == 2
    assert result.steps[0].skipped
    assert trainer.training_progress["optimizer_steps"] == 2
    assert trainer.training_progress["samples_seen"] == 3
    assert events[1]["skip_reason"] == "amp_overflow"


def test_training_jsonl_flushes_and_refuses_overwrite(tmp_path, capsys) -> None:
    path = tmp_path / "session.jsonl"
    with TrainingLog(path) as log:
        log.emit({"event": "sample_progress", "loss": 0.123, "optimizer_steps": 4})
        assert json.loads(path.read_text())["optimizer_steps"] == 4
    assert json.loads(capsys.readouterr().out)["loss"] == 0.123
    with pytest.raises(FileExistsError), TrainingLog(path):
        pass


def test_a04_train_cli_harness_options() -> None:
    from reliable_endo_gs.cli.main import build_parser

    args = build_parser().parse_args(
        [
            "a04",
            "train",
            "--data-root",
            "external",
            "--manifest",
            "index.json",
            "--epochs",
            "3",
            "--log-every",
            "7",
            "--checkpoint-every",
            "19",
            "--checkpoint-dir",
            "outside/checkpoints",
            "--log-file",
            "outside/log.jsonl",
        ]
    )
    assert args.epochs == 3 and args.log_every == 7 and args.checkpoint_every == 19
    assert args.checkpoint_dir == Path("outside/checkpoints")


def test_a04_cli_trains_checkpoints_and_resumes_next_sample(tmp_path, monkeypatch, capsys) -> None:
    import yaml

    from reliable_endo_gs.cli import main as cli

    fixture = _multi_trainer(tmp_path / "data")
    config_path = tmp_path / "a04.yaml"
    config_path.write_text(yaml.safe_dump(fixture.config.to_dict()))
    monkeypatch.setattr(cli, "load_teacher_factory", lambda factory: _teacher())
    common = [
        "a04",
        "train",
        "--data-root",
        str(tmp_path / "data"),
        "--manifest",
        str(tmp_path / "data.json"),
        "--config",
        str(config_path),
        "--teacher-factory",
        "fake:teacher",
        "--max-steps",
        "1",
        "--log-every",
        "1",
        "--checkpoint-every",
        "1",
        "--checkpoint-dir",
        str(tmp_path / "periodic"),
        "--cache-root",
        str(tmp_path / "no-cache"),
    ]
    first = cli._run_a04(
        cli.build_parser().parse_args(
            common
            + [
                "--output-checkpoint",
                str(tmp_path / "first.pt"),
                "--log-file",
                str(tmp_path / "first.jsonl"),
            ]
        )
    )
    assert first["optimizer_steps"] == 1
    resumed = cli._run_a04(
        cli.build_parser().parse_args(
            common
            + [
                "--checkpoint",
                str(tmp_path / "periodic/step-000000001.pt"),
                "--output-checkpoint",
                str(tmp_path / "second.pt"),
                "--log-file",
                str(tmp_path / "second.jsonl"),
            ]
        )
    )
    assert resumed["completed_steps"] == 1 and resumed["optimizer_steps"] == 2
    logs = [json.loads(line) for line in (tmp_path / "second.jsonl").read_text().splitlines()]
    sample = next(event for event in logs if event["event"] == "sample_progress")
    assert sample["sample_id"].endswith("000001")
    assert logs[0]["metadata"]["git_commit"]
    assert len(logs[0]["metadata"]["source_sha256"]) == 64
    assert len(capsys.readouterr().out.splitlines()) == 10


def test_a04_fallback_save_preserves_rng_mode_and_zero_cursor(tmp_path) -> None:
    trainer = _multi_trainer(tmp_path / "data")
    rng = capture_rng(trainer.device)
    trainer.save(tmp_path / "fresh.pt")
    _assert_tree_equal(capture_rng(trainer.device), rng)
    assert trainer.model.training
    assert trainer.training_progress["samples_seen"] == 0


def test_a04_resume_old_step_rejects_later_checkpoint_directory(tmp_path) -> None:
    trainer = _multi_trainer(tmp_path / "data")
    trainer.train(max_steps=2, checkpoint_every=1, checkpoint_dir=tmp_path / "periodic")
    resumed = _multi_trainer(tmp_path / "data")
    resumed.resume(tmp_path / "periodic/step-000000001.pt")
    with pytest.raises(A04TrainingError, match="contains later steps"):
        resumed.train(checkpoint_dir=tmp_path / "periodic")
    assert resumed.training_progress["optimizer_steps"] == 1
