"""Regression coverage for clipped samples, safe recovery and deployment preflight."""

import json
import signal
from dataclasses import replace

import pytest
import torch

from reliable_endo_gs.contracts.rendering import RenderOutput
from reliable_endo_gs.rendering.cross_view import ReferenceRenderer
from reliable_endo_gs.rendering.field import ExistingRendererBridge
from reliable_endo_gs.teachers import TeacherPrediction
from reliable_endo_gs.training.a04_reference import A04TrainingError
from reliable_endo_gs.training.harness import TrainingLog, TrainingSignals, capture_rng
from reliable_endo_gs.training.preflight import preflight_a04_data
from tests.integration.test_a01_a04_server_smoke import (
    _assert_tree_equal,
    _multi_trainer,
    _teacher,
)


class NearCullingRenderer:
    identity = "test-metric-near-culling"
    metric_near_plane_m = 0.01

    def __init__(self):
        self.renderer = ExistingRendererBridge(ReferenceRenderer())

    def render(self, request):
        field = request.field
        visible = field.valid_mask & (field.means3d[..., 2] > self.metric_near_plane_m)
        if not bool(visible.any()):
            h, w = request.image_hw
            return RenderOutput(torch.zeros(1, 3, h, w), visibility=visible)
        return replace(self.renderer.render(request), visibility=visible)


def clipping_teacher():
    def infer(model, left, right):
        disparity = 2500.0 if abs(float(left.mean()) - 129 / 255) < 0.0001 else 8.0
        return TeacherPrediction(torch.full((1, 1, 4, 5), disparity))

    return _teacher(infer_fn=infer)


def test_clipped_sample_is_logged_skipped_and_cursor_resumes(tmp_path, monkeypatch):
    trainer = _multi_trainer(
        tmp_path / "data", teacher=clipping_teacher(), renderer=NearCullingRenderer()
    )
    events = []
    result = trainer.train(progress=events.append, log_every=1)
    assert result.completed_steps == 2
    assert result.steps[1].skip_reason == "all_centres_before_near"
    assert result.steps[1].valid_pixels == 0 and result.steps[1].loss is None
    assert trainer.training_progress["samples_seen"] == 3
    assert trainer.training_progress["skip_counts"] == {"all_centres_before_near": 1}
    clipped_event = next(e for e in events if e.get("skip_reason") == "all_centres_before_near")
    assert clipped_event["diagnostics"]["visible_gaussians"] == 0
    assert clipped_event["diagnostics"]["depth_max_m"] < 0.01
    checkpoint = trainer.save(tmp_path / "clipped.pt")
    resumed = _multi_trainer(
        tmp_path / "data", teacher=clipping_teacher(), renderer=NearCullingRenderer()
    )
    resumed.resume(checkpoint)
    assert resumed.training_progress["skip_counts"] == {"all_centres_before_near": 1}
    assert resumed.train(epochs=2, max_steps=1).steps[0].sample_id.endswith("000000")


def test_clipping_smoke_is_strict(tmp_path):
    def near_teacher(model, left, right):
        return TeacherPrediction(torch.full((1, 1, 4, 5), 2500.0))

    trainer = _multi_trainer(
        tmp_path / "data", teacher=_teacher(infer_fn=near_teacher), renderer=NearCullingRenderer()
    )
    with pytest.raises(A04TrainingError, match="no visible primitives"):
        trainer.smoke()


def test_empty_geometry_support_is_explicitly_skipped_without_parameter_change(tmp_path):
    def invalid_teacher(model, left, right):
        return TeacherPrediction(torch.zeros((1, 1, 4, 5)))

    trainer = _multi_trainer(
        tmp_path / "data",
        teacher=_teacher(infer_fn=invalid_teacher),
        renderer=NearCullingRenderer(),
    )
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    result = trainer.train()
    assert result.completed_steps == 0
    assert all(step.skip_reason == "empty_geometry_support" for step in result.steps)
    assert trainer.training_progress["skip_counts"] == {"empty_geometry_support": 3}
    assert trainer.training_progress["samples_seen"] == 3
    assert not trainer.optimizer.state_dict()["state"]
    _assert_tree_equal(trainer.model.state_dict(), before)


def test_consecutive_skips_preserve_counts_and_recovery_checkpoint(tmp_path, monkeypatch):
    trainer = _multi_trainer(
        tmp_path / "data", teacher=clipping_teacher(), renderer=NearCullingRenderer()
    )
    with pytest.raises(A04TrainingError, match="consecutive unsupported") as failure:
        trainer.train(max_consecutive_skips=1)
    assert trainer.training_progress["next_sample_index"] == 2
    checkpoint = trainer.save_recovery(tmp_path / "recover", failure.value)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["training_state"]["optimizer_steps"] == 1
    assert payload["training_state"]["skip_counts"] == {"all_centres_before_near": 1}


def test_recovery_retains_last_good_cursor_and_rng(tmp_path, monkeypatch):
    trainer = _multi_trainer(tmp_path / "data")
    trainer.train(max_steps=1)
    expected_rng = capture_rng(trainer.device)
    weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    forward = trainer._forward_step

    def fail(sample):
        torch.rand(10)
        raise A04TrainingError("injected forward failure")

    monkeypatch.setattr(trainer, "_forward_step", fail)
    with pytest.raises(A04TrainingError) as error:
        trainer.train(max_steps=1)
    checkpoint = trainer.save_recovery(tmp_path / "recover", error.value)
    restored = _multi_trainer(tmp_path / "data")
    restored.resume(checkpoint)
    assert restored.training_progress["next_sample_index"] == 1
    _assert_tree_equal(capture_rng(restored.device), expected_rng)
    _assert_tree_equal(restored.model.state_dict(), weights)
    assert restored.train(max_steps=1).steps[0].sample_id.endswith("000001")
    monkeypatch.setattr(trainer, "_forward_step", forward)


def test_unsafe_optimizer_or_cuda_state_cannot_make_recovery(tmp_path, monkeypatch):
    trainer = _multi_trainer(tmp_path / "data")
    trainer.train(max_steps=1)
    with pytest.raises(A04TrainingError, match="CUDA context is unsafe"):
        trainer.save_recovery(tmp_path, RuntimeError("CUDA error: illegal memory access"))

    def fail():
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(trainer.optimizer, "step", fail)
    with pytest.raises(RuntimeError, match="optimizer failure"):
        trainer.train(max_steps=1)
    with pytest.raises(A04TrainingError, match="no safe completed"):
        trainer.save_recovery(tmp_path, RuntimeError("optimizer failure"))


def test_sigterm_drains_update_then_checkpoint_and_restores_handler(tmp_path, monkeypatch):
    trainer = _multi_trainer(tmp_path / "data")
    forward = trainer._forward_step
    original = signal.getsignal(signal.SIGTERM)

    def term_during_forward(sample):
        signal.raise_signal(signal.SIGTERM)
        return forward(sample)

    monkeypatch.setattr(trainer, "_forward_step", term_during_forward)
    with TrainingSignals() as signals:
        result = trainer.train(stop_requested=signals.requested)
        checkpoint = trainer.save(tmp_path / "stopped.pt")
    assert result.stop_reason == "SIGTERM" and result.completed_steps == 1
    assert signal.getsignal(signal.SIGTERM) == original
    state = torch.load(checkpoint, weights_only=True)["training_state"]
    assert state["optimizer_steps"] == 1 and state["next_sample_index"] == 1


def test_stdout_disconnect_keeps_disk_log_and_training_alive(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise BrokenPipeError("closed terminal")

    monkeypatch.setattr("builtins.print", broken)
    with TrainingLog(tmp_path / "log.jsonl") as log:
        log.emit({"event": "first"})
        log.emit({"event": "second"})
    events = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["first", "console_disconnected", "second"]


def test_preflight_validates_full_role_without_supervision_and_reports_corruption(tmp_path):
    trainer = _multi_trainer(tmp_path / "data")
    for path in (tmp_path / "data").rglob("*.tiff"):
        path.unlink()
    report = preflight_a04_data(trainer.adapter, workers=2)
    assert report["status"] == "PASS" and report["sample_count"] == 3
    assert report["ground_truth_loaded"] is False
    data = tmp_path / "data/dataset_1/keyframe_1/data"
    (data / "left_finalpass/frame_data000001.png").write_bytes(b"not an image")
    calibration = data / "newpram_data/frame_data000002.json"
    value = json.loads(calibration.read_text())
    value["intr0"][0][0] = -100
    calibration.write_text(json.dumps(value))
    failed = preflight_a04_data(trainer.adapter, workers=2)
    assert failed["status"] == "FAIL" and len(failed["errors"]) == 2
    assert failed["errors"][0]["sample_id"].endswith("000001")


def test_cli_preflight_cannot_construct_teacher_or_model(tmp_path, monkeypatch):
    from reliable_endo_gs.cli import main as cli

    _multi_trainer(tmp_path / "data")

    def forbidden(*args, **kwargs):
        raise AssertionError("no teacher or model is allowed in preflight")

    monkeypatch.setattr(cli, "load_teacher_factory", forbidden)
    monkeypatch.setattr(cli, "A04ReferenceTrainer", forbidden)
    args = cli.build_parser().parse_args(
        [
            "a04",
            "preflight",
            "--data-root",
            str(tmp_path / "data"),
            "--manifest",
            str(tmp_path / "data.json"),
            "--teacher-factory",
            "forbidden:teacher",
            "--report",
            str(tmp_path / "report.json"),
            "--workers",
            "1",
        ]
    )
    assert cli._run_a04(args)["status"] == "PASS"
    assert (tmp_path / "report.json").is_file()


def test_cli_python_failure_logs_original_error_and_saves_safe_recovery(tmp_path, monkeypatch):
    from reliable_endo_gs.cli import main as cli

    trainer = _multi_trainer(tmp_path / "data")
    original = trainer._forward_step

    def fail_second(sample):
        if sample.sample_id.endswith("000001"):
            raise A04TrainingError("injected CLI forward failure")
        return original(sample)

    monkeypatch.setattr(trainer, "_forward_step", fail_second)
    monkeypatch.setattr(cli, "A04ReferenceTrainer", lambda **kwargs: trainer)
    args = cli.build_parser().parse_args(
        [
            "a04",
            "train",
            "--data-root",
            str(tmp_path / "data"),
            "--manifest",
            str(tmp_path / "data.json"),
            "--renderer",
            "reference",
            "--output-checkpoint",
            str(tmp_path / "final.pt"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
        ]
    )
    with pytest.raises(A04TrainingError, match="injected CLI forward failure"):
        cli._run_a04(args)
    events = [json.loads(line) for line in (tmp_path / "final.jsonl").read_text().splitlines()]
    error = next(e for e in events if e["event"] == "train_error")
    assert error["next_sample_index"] == 1 and error["error"] == "injected CLI forward failure"
    saved = next(e for e in events if e["event"] == "recovery_checkpoint_saved")
    state = torch.load(saved["checkpoint"], weights_only=True)["training_state"]
    assert state["next_sample_index"] == 1 and state["optimizer_steps"] == 1
    assert not (tmp_path / "final.pt").exists()
