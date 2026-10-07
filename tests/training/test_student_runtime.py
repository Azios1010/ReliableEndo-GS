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


@pytest.fixture
def cpu_native_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake the CLI GPU gate while keeping real CPU optimizers off accelerators."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    # Recent PyTorch optimizers also query the generic accelerator API. The
    # model and Adam tensors in these tests stay on CPU, despite the fake gate.
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        monkeypatch.setattr(accelerator, "is_available", lambda: False)

    def forbidden_cuda_init() -> None:
        raise AssertionError("CPU runtime substitutes must never initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden_cuda_init)


def test_cpu_native_runtime_keeps_accelerator_probe_on_cpu(cpu_native_runtime):
    # CUDA availability is faked only to reach CLI orchestration. A generic
    # availability/synchronize probe must not reach the real driver.
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        available = accelerator.is_available()
        if available:
            accelerator.synchronize()
        assert not available
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.square().backward()
    optimizer.step()
    assert parameter.device.type == "cpu"
    assert parameter.item() < 1.0


def test_native_zero_repair_error_bounds():
    original = torch.zeros(1, 3, 100, 100)
    repaired = original.clone()
    repaired[0, 0, 0, 0] = 1.2e-4
    errors = student_runtime._check_zero_repair_render(repaired, original)
    assert errors["max_abs"] > 1e-4 and errors["mean_abs"] < 1e-7
    with pytest.raises(AssertionError):
        student_runtime._check_zero_repair_render(original + 3e-4, original)
    with pytest.raises(ValueError, match="mean RGB"):
        student_runtime._check_zero_repair_render(original + 1e-5, original)


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
def test_cli_fp32_startup_reaches_data_boundary(command, tmp_path, monkeypatch, cpu_native_runtime):
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

    def data_boundary(_):
        raise LookupError("data boundary reached after FP32 config validation")

    monkeypatch.setattr(student_runtime, "load_a01_roles", data_boundary)
    with pytest.raises(LookupError, match="data boundary reached"):
        student_runtime.run_student_command(build_parser().parse_args(argv))


@pytest.mark.parametrize(
    "command,cache_hit,populate,balanced",
    [
        ("train", True, False, False),
        ("train", False, False, False),
        ("smoke", True, False, False),
        ("smoke", False, False, False),
        ("train", True, True, False),
        ("train", False, True, False),
        ("smoke", True, False, True),
        ("smoke", True, False, "priority"),
    ],
)
def test_cached_cli_updates_without_loading_teacher(
    tmp_path, monkeypatch, command, cache_hit, populate, balanced, cpu_native_runtime
):
    import reliable_endo_gs.teachers.workflow as workflow
    from reliable_endo_gs.training.harness import capture_rng, restore_rng

    batch = _batch(dtype=torch.float32)
    sample = ScaredA00Sample("dataset_1/keyframe_1/frame_data000000", "final_train", batch, {})
    target = _target(batch)
    target = replace(target, disparity=target.disparity.float())
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
    population_calls = []
    before_teacher_rng = []

    def freeze(_):
        before_teacher_rng.append(torch.get_rng_state().clone())
        torch.rand(3)  # Simulate random constructor initialization.

    def populate_target(*args, **kwargs):
        population_calls.append(args[0].sample_id)
        torch.rand(5)
        assert kwargs["split_identity"] == "a01-scared-final-train-1-7-test-8-9-v1"
        return SimpleNamespace(target=target)

    monkeypatch.setattr(student_runtime, "_freeze_injected_teacher", freeze)
    monkeypatch.setattr(workflow, "collect_one_target", populate_target)
    monkeypatch.setattr(student_runtime, "capture_rng", lambda _: capture_rng(torch.device("cpu")))
    monkeypatch.setattr(
        student_runtime, "restore_rng", lambda state, _: restore_rng(state, torch.device("cpu"))
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
    if populate:
        argv += ["--populate-cache"]
    if balanced:
        argv += [
            "--config",
            "configs/training/intervention_balanced_smoke.yaml",
            "--intervention-config",
            "configs/routing/intervention_stereo_priority_smoke.yaml"
            if balanced == "priority"
            else "configs/routing/intervention_balanced_smoke.yaml",
        ]
    arguments = build_parser().parse_args(argv)
    if not cache_hit and not populate:
        with pytest.raises(ValueError, match="missing pinned KD cache item"):
            student_runtime.run_student_command(arguments)
        assert not (output / "final.pt").exists()
        assert read_student_checkpoint(output / "initial.pt")["cursor"]["optimizer_steps"] == 0
    else:
        report = student_runtime.run_student_command(arguments)
        assert report["schedule_complete"]
        assert report["cursor"]["optimizer_steps"] == (3 if balanced else 2)
        raw = read_student_checkpoint(output / "final.pt")
        assert raw["identity"]["renderer_identity"] == AnalyticRenderer().identity
        assert raw["identity"]["target_mode"] == (
            "cache_populate" if populate else "cache_readonly"
        )
        if not cache_hit:
            assert len(population_calls) == 2
            assert torch.equal(raw["rng"]["torch_cpu"], before_teacher_rng[-1])
        assert raw["construction_config"]["runtime"]["precision"] == "fp32"
        assert ReferenceConfig.from_dict(raw["construction_config"]).runtime.precision == "fp32"
        assert raw["cost_observations"] == (2 if balanced else 1)
        if balanced:
            assert raw["schema"] == "student_joint_intervention_checkpoint.v2"
            assert bool(raw["selector"]["normalization.fitted"])
            if balanced == "priority":
                assert raw["intervention_config"]["stereo_training_priority"] == 3.0
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
    if cache_hit or not populate:
        assert not population_calls and not before_teacher_rng


@pytest.mark.parametrize("command", ["smoke", "infer"])
def test_populate_cache_rejects_nontraining(command, tmp_path):
    argv = [
        command,
        "--data-root",
        str(tmp_path),
        "--manifest",
        str(tmp_path / "manifest"),
        "--output-dir",
        str(tmp_path / "output"),
        "--cache-root",
        str(tmp_path / "cache"),
        "--populate-cache",
        "--sample-id",
        "dataset_1/keyframe_1/frame_data000000",
    ]
    if command == "infer":
        argv += ["--checkpoint", str(tmp_path / "checkpoint.pt")]
    with pytest.raises(ValueError):
        student_runtime.run_student_command(build_parser().parse_args(argv))


def test_populate_cache_requires_explicit_cache_root(tmp_path):
    args = build_parser().parse_args(
        [
            "train",
            "--data-root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "manifest"),
            "--output-dir",
            str(tmp_path / "output"),
            "--populate-cache",
        ]
    )
    with pytest.raises(ValueError, match="requires train and --cache-root"):
        student_runtime.run_student_command(args)


def test_populate_cache_rejects_repository_output(tmp_path):
    repo = Path(student_runtime.__file__).resolve().parents[3]
    args = build_parser().parse_args(
        [
            "train",
            "--data-root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "manifest"),
            "--output-dir",
            str(tmp_path / "output"),
            "--populate-cache",
            "--cache-root",
            str(repo / "artifacts" / "forbidden_cache"),
        ]
    )
    with pytest.raises(ValueError, match="cache must be outside"):
        student_runtime.run_student_command(args)
