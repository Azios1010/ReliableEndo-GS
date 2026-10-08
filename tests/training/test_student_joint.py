"""CPU synthetic student/joint gradient, deployment and recovery contracts."""

from __future__ import annotations

import copy
import importlib.abc
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.actions.repair import Action, primitive_core
from reliable_endo_gs.cli.student import build_parser
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.gaussians import GeometryPriorPolicy
from reliable_endo_gs.reconstruction import ReferenceConfig, ReferenceReconstructionModel
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.regions.grid import grid_regions
from reliable_endo_gs.routing.config import RoutingConfig
from reliable_endo_gs.routing.selector import region_features
from reliable_endo_gs.training.action_learning import ActionLearner
from reliable_endo_gs.training.interventions import reconstruction_loss
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_config import StudentLearningConfig, load_student_config
from reliable_endo_gs.training.student_learning import StudentLearner
from reliable_endo_gs.training.student_run import run_student_training
from reliable_endo_gs.training.student_runtime import run_student_command
from tests.architecture.test_a04_reference_reconstruction import _batch, _FakeTeacher, _target
from tests.routing.test_frozen_a04 import AnalyticRenderer


@pytest.fixture
def setup():
    torch.manual_seed(0)
    config = replace(
        ReferenceConfig(),
        head_hidden_dims=(8, 8),
        prior_policy=GeometryPriorPolicy(alpha_valid=0.5),
    )
    model = StudentReconstructionModel(renderer=AnalyticRenderer(), config=config, width=8, bins=4)
    learner = StudentLearner(
        model, StudentLearningConfig(width=8, bins=4), RoutingConfig(width=8, tile_hw=(3, 3))
    )
    sample = ScaredA00Sample(
        "dataset_1/keyframe_1/frame_data000000", "final_train", _batch(dtype=torch.float32), {}
    )
    target = _target(sample.batch)
    target = replace(target, disparity=target.disparity.float().requires_grad_(True))
    return learner, sample, target


def snapshot(module):
    return {k: v.clone() for k, v in module.state_dict().items()}


def changed(module, before):
    return any(not torch.equal(v, before[k]) for k, v in module.state_dict().items())


def test_student_contract_and_reconstruction_gradient(setup):
    learner, sample, _ = setup
    state = learner.state(sample)
    assert state.base.stereo.disparity.shape == (1, 1, 5, 5)
    assert state.base.target.disparity.requires_grad
    assert state.base.render_right is not None
    assert state.base.provenance_record.target_source == "trainable_student_prediction"
    assert learner.model.teacher is None
    assert state.base.provenance_record.supervision_views == ("left", "right")
    assert state.base.provenance["held_out_supervision_view"] is False
    assert all("teacher" not in name for name in learner.model.state_dict())
    full = grid_regions((5, 5), (5, 5), 4)[0]
    loss = reconstruction_loss(
        state, learner.actions.execute(state, full, Action.STOP, learner.model)
    )
    loss.backward()
    grads = [p.grad for p in learner.model.stereo_student.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


@pytest.mark.parametrize("joint", [False, True])
def test_schedule_updates_and_optimizer_ownership(setup, joint):
    learner, sample, target = setup
    before = [
        snapshot(m)
        for m in (
            learner.model.stereo_student,
            learner.model.head,
            learner.actions,
            learner.selector,
        )
    ]
    groups = [
        {id(p) for g in o.param_groups for p in g["params"]}
        for o in (learner.model_optimizer, learner.action_optimizer, learner.selector_optimizer)
    ]
    assert groups[0] == {id(p) for p in learner.model.parameters()}
    assert groups[1] == {id(p) for p in learner.actions.parameters()}
    assert groups[2] == {id(p) for p in learner.selector.parameters()}
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    result = learner.step(sample, joint=joint, teacher_target=target)
    assert result["loss"] > 0
    assert target.disparity.grad is None
    for i, module in enumerate(
        (learner.model.stereo_student, learner.model.head, learner.actions, learner.selector)
    ):
        assert changed(module, before[i]) == (i < 2 or joint)
    if joint:
        assert learner.cost_observations == 1
        assert any(
            p.grad is not None and bool((p.grad != 0).any())
            for p in learner.actions.stereo.parameters()
        )
        assert any(
            p.grad is not None and bool((p.grad != 0).any())
            for p in learner.actions.gaussian.parameters()
        )


def test_reference_teacher_unchanged(setup):
    learner, sample, _ = setup
    teacher = _FakeTeacher()
    reference = (
        ReferenceReconstructionModel(
            teacher=teacher, renderer=AnalyticRenderer(), config=learner.model.config
        )
        .eval()
        .requires_grad_(False)
    )
    before = snapshot(reference), snapshot(teacher)
    with torch.no_grad():
        target = reference(sample.batch).target
    learner.step(sample, joint=True, teacher_target=target)
    assert not changed(reference, before[0]) and not changed(teacher, before[1])
    assert all(p.grad is None for p in reference.parameters())
    assert teacher.weight.grad is None


def test_same_state_detached_labels_zero_identity_and_correspondence(setup, monkeypatch):
    learner, sample, target = setup
    state = learner.state(sample)
    region = grid_regions((5, 5), (5, 5), 4)[0]
    stop = learner.actions.execute(state, region, Action.STOP, learner.model)
    assert stop.field is state.base.gaussians and stop.left is state.base.render_left
    for action in (Action.STEREO, Action.GAUSSIAN):
        outcome = learner.actions.execute(state, region, action, learner.model)
        for name in ("means3d", "scales", "rotations", "opacities"):
            torch.testing.assert_close(getattr(outcome.field, name), getattr(stop.field, name))
    assert torch.equal(primitive_core(state, region), state.base.gaussians.valid_mask)
    seen = []
    execute = learner.actions.execute

    def record(state, *args):
        seen.append(state)
        return execute(state, *args)

    monkeypatch.setattr(learner.actions, "execute", record)
    learner.step(sample, joint=True, teacher_target=target)
    assert len(seen) == 6 and all(s is seen[0] for s in seen)
    learner.model.zero_grad(set_to_none=True)
    features = region_features(state, (region,))
    assert not features.requires_grad
    learner.selector(features).square().sum().backward()
    assert all(p.grad is None for p in learner.model.parameters())


def test_student_inference_and_evaluation_preserve_state(setup, monkeypatch):
    learner, sample, target = setup
    learner.step(sample, joint=True, teacher_target=target)
    modules = (learner.model, learner.actions, learner.selector)
    before = [snapshot(m) for m in modules]
    modes = [m.training for module in modules for m in module.modules()]
    flags = [p.requires_grad for module in modules for p in module.parameters()]
    optimizer = copy.deepcopy(learner.model_optimizer.state_dict())
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("inference loaded weights"))
    learner.infer(sample)
    learner.infer(sample, selected=True)
    assert all(not changed(m, b) for m, b in zip(modules, before, strict=True))
    assert modes == [m.training for module in modules for m in module.modules()]
    assert flags == [p.requires_grad for module in modules for p in module.parameters()]
    assert optimizer["param_groups"] == learner.model_optimizer.state_dict()["param_groups"]


def test_gt_and_final_test_exclusion(setup):
    learner, sample, target = setup
    with pytest.raises(ValueError, match="GT"):
        learner.infer(replace(sample, supervision=object()))
    with pytest.raises(ValueError, match="final_test"):
        learner.step(replace(sample, data_role="final_test"), joint=False, teacher_target=target)
    with pytest.raises(ValueError, match="teacher targets"):
        learner.model(sample.batch, target=target)


def test_config_validation(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("joint_epochs: 0\n", encoding="utf-8")
    assert load_student_config(path).joint_epochs == 0
    path.write_text("unknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_student_config(path)
    for kwargs in (
        {"width": True},
        {"learning_rate": float("nan")},
        {"joint_epochs": -1},
        {"student_epochs": -1},
        {"student_epochs": 0, "joint_epochs": 0},
        {"schema": "frozen_a04_action_selector.v1"},
    ):
        with pytest.raises(ValueError):
            StudentLearningConfig(**kwargs)


def test_joint_from_first_sample_and_resume(setup, tmp_path):
    learner, sample, _ = setup
    learner.config = replace(
        learner.config, student_epochs=0, joint_epochs=2, kd_weight=0, checkpoint_every=1
    )
    first = run_student_training([sample], learner, {}, tmp_path / "first", max_steps=1)
    assert learner.cost_observations == 1
    for optimizer in (
        learner.model_optimizer,
        learner.action_optimizer,
        learner.selector_optimizer,
    ):
        assert all(float(state["step"]) == 1 for state in optimizer.state.values())
    load_student_checkpoint(Path(first["checkpoint"]), learner, {}, 1)
    result = run_student_training(
        [sample], learner, {}, tmp_path / "resumed", resume=Path(first["checkpoint"])
    )
    assert result["schedule_complete"]
    assert learner.cost_observations == 2
    import json

    event = json.loads((tmp_path / "resumed" / "train.jsonl").read_text())
    assert event["stage"] == "joint"
    assert event["elapsed_seconds"] > 0
    for component in ("student", "head", "stereo_repair", "gaussian_repair", "selector"):
        assert event[f"grad_norm_{component}"] > 0
    for action in ("stop", "stereo", "gaussian"):
        assert f"utility_{action}" in event


def test_save_load_resume_exact_student_schedule(setup, tmp_path):
    learner, sample, _ = setup
    learner.config = replace(
        learner.config, kd_weight=0, student_epochs=2, joint_epochs=0, checkpoint_every=1
    )
    samples = [sample, replace(sample, sample_id="dataset_1/keyframe_1/frame_data000001")]
    continuous = copy.deepcopy(learner)
    full = run_student_training(samples, continuous, {}, tmp_path / "continuous")
    first = run_student_training(samples, learner, {}, tmp_path / "bounded", max_steps=1)
    assert first["cursor"]["next_sample_index"] == 1
    restored = copy.deepcopy(learner)
    with torch.no_grad():
        for p in restored.model.parameters():
            p.add_(1)
    final = run_student_training(
        samples, restored, {}, tmp_path / "resume", resume=Path(first["checkpoint"])
    )
    assert final["cursor"] == full["cursor"]
    assert final["cursor"]["optimizer_steps"] == 4
    for name, value in continuous.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])
    assert len(restored.model_optimizer.state) > 0


@pytest.mark.parametrize(
    "corruption",
    ["schema", "identity", "hashes", "cursor", "weights", "optimizer", "costs", "teacher", "rng"],
)
def test_checkpoint_compatibility_failure_is_nonmutating(setup, tmp_path, corruption):
    learner, sample, target = setup
    learner.step(sample, joint=True, teacher_target=target)
    path = tmp_path / "valid.pt"
    cursor = {"epoch": 1, "next_sample_index": 0, "samples_seen": 1, "optimizer_steps": 1}
    save_student_checkpoint(path, learner, {"source": "x"}, cursor)
    raw = torch.load(path, weights_only=True)
    if corruption == "schema":
        raw["schema"] = "action_learning_checkpoint.v1"
    elif corruption == "identity":
        raw["identity"] = {"source": "wrong"}
    elif corruption == "hashes":
        raw["hashes"][0] = "wrong"
    elif corruption == "cursor":
        raw["cursor"]["next_sample_index"] = 1
    elif corruption == "weights":
        raw["model"][next(iter(raw["model"]))].fill_(float("nan"))
    elif corruption == "optimizer":
        raw["selector_optimizer"]["param_groups"][0]["params"] = []
    elif corruption == "costs":
        raw["cost_max_ms"][0] = 1
    elif corruption == "rng":
        raw["rng"] = {}
    else:
        raw["teacher_in_checkpoint"] = True
    broken = tmp_path / "broken.pt"
    torch.save(raw, broken)
    before = [snapshot(m) for m in (learner.model, learner.actions, learner.selector)]
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        load_student_checkpoint(broken, learner, {"source": "x"}, 1)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        not changed(m, b)
        for m, b in zip((learner.model, learner.actions, learner.selector), before, strict=True)
    )


def test_checkpoint_ownership_and_teacher_free_deployment(setup, tmp_path, monkeypatch):
    learner, sample, target = setup
    learner.step(sample, joint=True, teacher_target=target)
    path = tmp_path / "joint.pt"
    identity = {
        k: "test-v1" for k in ("source_sha256", "manifest_hash", "role_hash", "renderer_identity")
    }
    cursor = {"epoch": 1, "next_sample_index": 0, "samples_seen": 1, "optimizer_steps": 1}
    save_student_checkpoint(path, learner, identity, cursor)
    raw = torch.load(path, weights_only=True)
    assert raw["teacher_in_checkpoint"] is False and raw["reference_in_checkpoint"] is False
    assert set(raw["model"]) == set(learner.model.state_dict())
    assert raw["action_optimizer"]["state"] and raw["selector_optimizer"]["state"]

    class BlockTeacher(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith(
                ("reliable_endo_gs.teachers.external", "FoundationStereo", "foundation_stereo")
            ):
                pytest.fail("student deployment imported teacher: " + fullname)
            return None

    monkeypatch.setattr(sys, "meta_path", [BlockTeacher(), *sys.meta_path])
    monkeypatch.setattr(
        "reliable_endo_gs.teachers.workflow.load_teacher_factory",
        lambda *a, **k: pytest.fail("teacher resolved"),
    )
    restored = copy.deepcopy(learner)
    with torch.no_grad():
        for p in restored.model.parameters():
            p.add_(1)
    before_optimizer = copy.deepcopy(restored.model_optimizer.state_dict())
    before_rng = torch.get_rng_state().clone()
    load_student_checkpoint(path, restored, {**identity, "hardware": "other"}, 1, resume=False)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert before_optimizer["param_groups"] == restored.model_optimizer.state_dict()["param_groups"]
    result = restored.infer(sample, selected=True)
    assert result.render_right is not None
    with pytest.raises(FileExistsError):
        save_student_checkpoint(path, learner, identity, cursor)


def test_failed_run_keeps_safe_checkpoint_and_rejects_gt_before_teacher(setup, tmp_path):
    learner, sample, _ = setup

    def provider(_):
        pytest.fail("teacher must not be called for GT source")

    with pytest.raises(ValueError, match="GT/role"):
        run_student_training(
            [replace(sample, supervision=object())],
            learner,
            {},
            tmp_path / "failed",
            target_provider=provider,
        )
    initial = torch.load(tmp_path / "failed/initial.pt", weights_only=True)
    assert initial["cursor"]["samples_seen"] == 0
    assert not (tmp_path / "failed/final.pt").exists()


def test_cli_and_checked_in_configs():
    args = build_parser().parse_args(
        [
            "train",
            "--data-root",
            "data",
            "--manifest",
            "manifest",
            "--output-dir",
            "out",
            "--max-steps",
            "2",
        ]
    )
    assert args.max_steps == 2 and args.teacher_factory is None
    assert load_student_config(Path("configs/training/student.yaml")).joint_epochs == 0
    assert load_student_config(Path("configs/training/student_joint.yaml")).joint_epochs == 1


def test_frozen_mode_rejects_student_instead_of_freezing_it(setup):
    learner, _, _ = setup
    with pytest.raises(ValueError, match="StudentLearner"):
        ActionLearner(learner.model, learner.routing)
    assert all(p.requires_grad for p in learner.model.parameters())


@pytest.mark.parametrize(
    "extra", [["--teacher-factory", "forbidden:factory"], ["--cache-root", "cache"]]
)
def test_cli_inference_rejects_teacher_inputs_before_gpu_loading(extra):
    args = build_parser().parse_args(
        [
            "infer",
            "--data-root",
            "data",
            "--manifest",
            "manifest",
            "--output-dir",
            "out",
            "--checkpoint",
            "own.pt",
            "--sample-id",
            "dataset_1/a",
            *extra,
        ]
    )
    with pytest.raises(ValueError, match="student-only inference"):
        run_student_command(args)


def test_cli_metric_deployment_inputs_are_complete_and_infer_only():
    base = [
        "infer",
        "--data-root",
        "data",
        "--manifest",
        "manifest",
        "--output-dir",
        "out",
        "--checkpoint",
        "own.pt",
        "--sample-id",
        "dataset_1/a",
    ]
    parser = build_parser()
    args = parser.parse_args([*base, "--metric-selector", "predictor.pt"])
    with pytest.raises(ValueError, match="metric-selector requires"):
        run_student_command(args)

    args = parser.parse_args(
        [
            "train",
            "--data-root",
            "data",
            "--manifest",
            "manifest",
            "--output-dir",
            "out",
            "--cost-calibration",
            "timing.pt",
        ]
    )
    with pytest.raises(ValueError, match="inference-only"):
        run_student_command(args)


def test_eval_error_restores_modes_without_mutation(setup):
    learner, sample, _ = setup
    modules = learner.model, learner.actions, learner.selector
    before = [snapshot(m) for m in modules]
    modes = [m.training for module in modules for m in module.modules()]
    with pytest.raises(ValueError, match="measured cost"):
        learner.infer(sample, selected=True)
    assert modes == [m.training for module in modules for m in module.modules()]
    assert all(not changed(m, b) for m, b in zip(modules, before, strict=True))


def test_student_support_and_repair_core_are_predicted_only(setup):
    learner, sample, target = setup
    support = sample.batch.support.clone()
    support[..., 0, :] = False
    sample = replace(sample, batch=replace(sample.batch, support=support))
    learner.step(sample, joint=True, teacher_target=target)
    state = learner.state(sample)
    region = grid_regions((5, 5), (3, 3), 4)[0]
    slots = primitive_core(state, region)
    for action in (Action.STEREO, Action.GAUSSIAN):
        result = learner.actions.execute(state, region, action, learner.model)
        for name in ("means3d", "scales", "rotations", "opacities"):
            assert torch.equal(
                getattr(result.field, name)[~slots], getattr(state.base.gaussians, name)[~slots]
            )
    assert not bool(state.support[..., 0, :].any())


def test_joint_resume_restores_all_optimizers_then_advances_schedule(setup, tmp_path):
    learner, sample, target = setup
    learner.config = replace(learner.config, joint_epochs=2)
    first = run_student_training(
        [sample],
        learner,
        {},
        tmp_path / "joint-bounded",
        max_steps=2,
        target_provider=lambda _: target,
    )
    restored = StudentLearner(copy.deepcopy(learner.model), learner.config, learner.routing)
    path = Path(first["checkpoint"])
    cursor = load_student_checkpoint(path, restored, {}, 1)
    assert cursor["epoch"] == 2 and restored.cost_observations == 1
    torch.testing.assert_close(learner.cost_max_ms, restored.cost_max_ms)
    for source, loaded in zip(
        (learner.model, learner.actions, learner.selector),
        (restored.model, restored.actions, restored.selector),
        strict=True,
    ):
        for name, value in source.state_dict().items():
            assert torch.equal(value, loaded.state_dict()[name])
    for original, loaded in zip(
        (learner.model_optimizer, learner.action_optimizer, learner.selector_optimizer),
        (restored.model_optimizer, restored.action_optimizer, restored.selector_optimizer),
        strict=True,
    ):
        original_state, loaded_state = original.state_dict(), loaded.state_dict()
        assert original_state["param_groups"] == loaded_state["param_groups"]
        for index, state in original_state["state"].items():
            for name, value in state.items():
                assert torch.equal(value, loaded_state["state"][index][name])
    final = run_student_training(
        [sample],
        restored,
        {},
        tmp_path / "joint-resumed",
        resume=path,
        target_provider=lambda _: target,
    )
    assert final["schedule_complete"] and final["cursor"]["optimizer_steps"] == 3
    assert restored.cost_observations == 2
    assert all(float(state["step"]) == 3 for state in restored.model_optimizer.state.values())
    assert all(float(state["step"]) == 2 for state in restored.action_optimizer.state.values())
    assert all(float(state["step"]) == 2 for state in restored.selector_optimizer.state.values())


def test_smoke_checkpoint_cannot_be_deployed(setup, tmp_path):
    learner, _, _ = setup
    identity = {
        k: "test-v1" for k in ("source_sha256", "manifest_hash", "role_hash", "renderer_identity")
    }
    path = tmp_path / "smoke.pt"
    cursor = {"epoch": 0, "next_sample_index": 0, "samples_seen": 0, "optimizer_steps": 0}
    save_student_checkpoint(path, learner, {**identity, "run_kind": "bounded_smoke"}, cursor)
    with pytest.raises(ValueError, match="smoke checkpoints"):
        load_student_checkpoint(path, learner, identity, 1, resume=False)
