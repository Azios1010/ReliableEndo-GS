"""CLI orchestration with real configs/updates and explicit CPU native substitutes.

These tests do not certify the compiled CUDA renderer or server assets. They
exercise the startup/config/cache paths missed by learner-only unit tests.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from reliable_endo_gs.cli.student import build_parser
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.reconstruction import ReferenceConfig, load_reference_config
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.teachers.base import LazyTeacherAdapter
from reliable_endo_gs.training import student_runtime
from reliable_endo_gs.training.a04_reference import A04ReferenceTrainer
from reliable_endo_gs.training.student_checkpoint import read_student_checkpoint
from tests.architecture.test_a04_reference_reconstruction import _batch, _FakeTeacher, _target
from tests.routing.test_frozen_a04 import AnalyticRenderer


def test_fp32_runtime_roundtrip_and_frozen_default():
    original = load_reference_config(Path("configs/reconstruction/a04_reference.yaml"))
    assert original.runtime.precision == "fp16"
    assert original.runtime.geometry_precision == "fp32"
    student = replace(original, runtime=replace(original.runtime, precision="fp32"))
    assert ReferenceConfig.from_dict(student.to_dict()) == student
    # FP32 must not silently select the old trainer's BF16 autocast fallback.
    trainer = object.__new__(A04ReferenceTrainer)
    trainer.config, trainer.device = student, torch.device("cuda:0")
    with trainer._autocast_context():
        assert not torch.is_autocast_enabled("cuda")


@pytest.mark.parametrize("command", ["train", "smoke"])
def test_cli_fp32_startup_reaches_data_boundary(command, tmp_path, monkeypatch):
    argv = [
        command,
        "--config",
        "configs/training/student_joint.yaml",
        "--data-root",
        str(tmp_path / "data"),
        "--manifest",
        str(tmp_path / "manifest"),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    if command == "smoke":
        argv += ["--sample-id", "dataset_1/keyframe_1/frame_data000000"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def data_boundary(_):
        raise LookupError("data boundary reached after FP32 config validation")

    monkeypatch.setattr(student_runtime, "load_a01_roles", data_boundary)
    with pytest.raises(LookupError, match="data boundary reached"):
        student_runtime.run_student_command(build_parser().parse_args(argv))


@pytest.mark.parametrize("command", ["train", "smoke"])
@pytest.mark.parametrize("cache_hit", [True, False])
def test_cached_cli_updates_without_loading_teacher(tmp_path, monkeypatch, command, cache_hit):
    import reliable_endo_gs.teachers.workflow as workflow

    batch = _batch(dtype=torch.float32)
    sample = ScaredA00Sample("dataset_1/keyframe_1/frame_data000000", "final_train", batch, {})
    target = _target(batch)
    target = replace(target, disparity=target.disparity.float())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "CPU native substitute")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="test"))
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda _: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda _: 0)
    monkeypatch.setattr(
        student_runtime, "set_seed", lambda seed: torch.random.default_generator.manual_seed(seed)
    )
    monkeypatch.setattr(
        student_runtime, "load_a01_roles", lambda _: SimpleNamespace(role_hash="test")
    )
    monkeypatch.setattr(
        student_runtime, "load_a01_manifest", lambda *a, **k: SimpleNamespace(manifest_hash="test")
    )
    monkeypatch.setattr(
        student_runtime,
        "ScaredA00Adapter",
        lambda *a, **k: SimpleNamespace(load=lambda *a, **k: sample),
    )
    monkeypatch.setattr(student_runtime, "ScaredA00Dataset", lambda *a, **k: [sample])

    def construct(**kwargs):
        assert kwargs["config"].runtime.precision == "fp32"
        kwargs["renderer"] = AnalyticRenderer()
        model = StudentReconstructionModel(**kwargs)
        # Only the runtime's native device transfer is substituted. Learner,
        # losses, actions, parameter updates, checkpoints and configs are real.
        monkeypatch.setattr(model, "to", lambda _: model)
        return model

    monkeypatch.setattr(student_runtime, "StudentReconstructionModel", construct)

    def forbidden_load():
        raise AssertionError("cached CLI must never load teacher weights")

    teacher = LazyTeacherAdapter(
        _FakeTeacher().identity,
        forbidden_load,
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected teacher inference")),
    )
    monkeypatch.setattr(workflow, "load_teacher_factory", lambda _: teacher)
    monkeypatch.setattr(workflow, "target_cache_key", lambda *a, **k: "pinned-key")
    monkeypatch.setattr(
        student_runtime,
        "TargetCache",
        lambda *a, **k: SimpleNamespace(get=lambda _: target if cache_hit else None),
    )
    output = tmp_path / "output"
    argv = [
        command,
        "--data-root",
        str(tmp_path / "data"),
        "--manifest",
        str(tmp_path / "manifest"),
        "--output-dir",
        str(output),
        "--cache-root",
        str(tmp_path / "cache"),
    ]
    if command == "train":
        argv += ["--max-steps", "2"]
    else:
        argv += ["--sample-id", sample.sample_id]
    arguments = build_parser().parse_args(argv)
    if not cache_hit:
        with pytest.raises(ValueError, match="missing pinned KD cache item"):
            student_runtime.run_student_command(arguments)
        assert not (output / "final.pt").exists()
        assert read_student_checkpoint(output / "initial.pt")["cursor"]["optimizer_steps"] == 0
    else:
        report = student_runtime.run_student_command(arguments)
        assert report["schedule_complete"]
        assert report["cursor"]["optimizer_steps"] == 2
        raw = read_student_checkpoint(output / "final.pt")
        assert raw["construction_config"]["runtime"]["precision"] == "fp32"
        assert ReferenceConfig.from_dict(raw["construction_config"]).runtime.precision == "fp32"
        assert raw["cost_observations"] == 1
        assert (output / "result.json").exists()
        if command == "smoke":
            assert all(report["parameters_changed"].values())
            assert report["elapsed_training_seconds"] > 0
            assert report["peak_allocated_bytes"] == report["peak_reserved_bytes"] == 0
        else:
            # Exercise FP32 checkpoint reconstruction through the inference CLI;
            # teacher-free core inference alone would miss parser/schema errors.
            monkeypatch.setattr(workflow, "load_teacher_factory", lambda _: forbidden_load())
            infer = build_parser().parse_args(
                [
                    "infer",
                    "--data-root",
                    str(tmp_path / "data"),
                    "--manifest",
                    str(tmp_path / "manifest"),
                    "--output-dir",
                    str(tmp_path / "inference"),
                    "--checkpoint",
                    str(output / "final.pt"),
                    "--sample-id",
                    sample.sample_id,
                ]
            )
            deployment = student_runtime.run_student_command(infer)
            assert deployment["teacher_loaded"] is False
            assert (tmp_path / "inference" / "prediction.pt").exists()
    assert not teacher.model_loaded
