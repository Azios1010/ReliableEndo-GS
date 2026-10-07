"""Shared A03/head/render integration and pinned frozen-import contracts."""

import copy
import hashlib
import json
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

from reliable_endo_gs.actions.geometry_repair import (
    GeometryRepair,
    GeometryRepairConfig,
    GeometryRepairResult,
)
from reliable_endo_gs.actions.repair import geometry_repair_outcome, primitive_core
from reliable_endo_gs.evaluation.geometry_native import evaluate_native_geometry_candidates
from reliable_endo_gs.regions.grid import grid_regions
from reliable_endo_gs.training.student_checkpoint import (
    import_frozen_student_actions,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_learning import StudentLearner
from tests.routing.test_balanced import setup as balanced_fixture

native_setup = balanced_fixture


def test_geometry_candidate_executes_shared_a03_head_and_two_renders(native_setup, monkeypatch):
    learner, sample, _ = native_setup
    learner.model.eval().requires_grad_(False)
    state = learner.state(sample)
    region = grid_regions((5, 5), (3, 3), learner.routing.halo)[0]
    action = GeometryRepair(GeometryRepairConfig())

    def propose(batch, disparity, core):
        return GeometryRepairResult(torch.where(core, disparity + 0.1, disparity), core)

    monkeypatch.setattr(action, "apply", propose)
    before = copy.deepcopy(learner.model.state_dict())
    outcome = action.execute(state, region, learner.model)
    assert outcome.left.image.shape == outcome.right.image.shape == sample.batch.left.shape
    assert not torch.equal(outcome.field.means3d, state.base.gaussians.means3d)
    outside = ~primitive_core(state, region)
    for item in fields(outcome.field):
        old, new = getattr(state.base.gaussians, item.name), getattr(outcome.field, item.name)
        if old is not None:
            assert torch.equal(old[outside], new[outside])
    assert all(torch.equal(v, before[k]) for k, v in learner.model.state_dict().items())
    invalid = state.base.target.disparity + 0.1
    with pytest.raises(ValueError, match="outside core"):
        geometry_repair_outcome(state, region, invalid, learner.model)


def test_geometry_abstention_preserves_actual_base_without_roundtrip(native_setup, monkeypatch):
    learner, sample, _ = native_setup
    learner.model.eval().requires_grad_(False)
    state = learner.state(sample)
    region = grid_regions((5, 5), (3, 3), learner.routing.halo)[0]
    action = GeometryRepair(GeometryRepairConfig())
    monkeypatch.setattr(
        action, "apply", lambda b, d, c: GeometryRepairResult(d, torch.zeros_like(c))
    )
    outcome = action.execute(state, region, learner.model)
    assert outcome.field is state.base.gaussians
    assert outcome.left is state.base.render_left and outcome.right is state.base.render_right
    assert not outcome.changed_fields


def test_frozen_import_is_pinned_and_does_not_import_optimizers_or_selector(native_setup, tmp_path):
    learner, sample, target = native_setup
    learner.step(sample, joint=True, teacher_target=target, position=0)
    path = tmp_path / "weights.pt"
    identity = dict(source_sha256="historical", run_kind="full_manifest")
    save_student_checkpoint(
        path,
        learner,
        identity,
        dict(epoch=0, next_sample_index=1, samples_seen=1, optimizer_steps=1),
    )
    clone = StudentLearner(
        copy.deepcopy(learner.model),
        learner.config,
        learner.routing,
        intervention_config=learner.intervention_config,
    )
    selector_before = copy.deepcopy(clone.selector.state_dict())
    with pytest.raises(ValueError, match="SHA-256"):
        import_frozen_student_actions(path, clone, expected_sha256="0" * 64)
    imported = import_frozen_student_actions(
        path, clone, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert imported["source_sha256"] == "historical"
    assert imported["selector_imported"] is imported["optimizer_imported"] is False
    assert not clone.model_optimizer.state and not clone.action_optimizer.state
    assert all(torch.equal(v, selector_before[k]) for k, v in clone.selector.state_dict().items())
    assert all(not p.requires_grad for p in clone.model.parameters())
    assert all(not p.requires_grad for p in clone.actions.parameters())
    assert all(
        torch.equal(v, learner.actions.state_dict()[k])
        for k, v in clone.actions.state_dict().items()
    )


def test_native_gpu_gate_does_not_fall_back_to_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="visible GPU"):
        evaluate_native_geometry_candidates(
            checkpoint=tmp_path / "missing",
            probe_report=tmp_path / "missing",
            data_root=tmp_path,
            manifest_path=tmp_path / "missing",
            training_manifest=tmp_path / "missing",
            roles_path=tmp_path / "missing",
            output_dir=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("collection_role", [None, "train", "development"])
def test_native_orchestration_with_explicit_cpu_renderer_substitution(
    native_setup, tmp_path, monkeypatch, collection_role
):
    """Exercise native startup/order/outputs; does NOT certify CUDA kernels."""
    import reliable_endo_gs.evaluation.geometry_native as runtime
    from reliable_endo_gs.actions.repair import RepairActions
    from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
    from tests.routing.test_frozen_a04 import AnalyticRenderer

    class BoundedRenderer(AnalyticRenderer):
        def render(self, request):
            result = super().render(request)
            return replace(result, image=result.image.sigmoid())

    learner, sample, _ = native_setup
    learner.model.config = replace(
        learner.model.config,
        renderer_identity=AnalyticRenderer.identity,
        runtime=replace(learner.model.config.runtime, precision="fp32", resolution_hw=(5, 5)),
    )
    path = tmp_path / "snapshot.pt"
    save_student_checkpoint(
        path,
        learner,
        dict(
            source_sha256="historical",
            run_kind="full_manifest",
            manifest_hash="train",
            role_hash="roles",
            renderer_identity=AnalyticRenderer.identity,
        ),
        dict(epoch=0, next_sample_index=0, samples_seen=0, optimizer_steps=0),
    )
    cfg = GeometryRepairConfig()
    from dataclasses import asdict

    ids = [f"dataset_{i}/keyframe_1/a" for i in range(1, 8)]
    probe = tmp_path / "probe.json"
    probe.write_text(
        json.dumps(
            dict(
                status="cpu_geometry_candidate_screen_complete",
                sample_ids=ids,
                development_manifest_hash="dev",
                training_manifest_hash="train",
                role_hash="roles",
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                minimum_local_gain_mm=0.1,
                configs=[{"resolved": asdict(cfg), "hash": cfg.config_hash}],
            )
        )
    )
    entries = [SimpleNamespace(sample_id=sid, data_role="final_train") for sid in ids]
    training_entries = [
        SimpleNamespace(sample_id=sid.replace("/a", "/train"), data_role="final_train")
        for sid in ids
    ]
    monkeypatch.setattr(runtime, "load_a01_roles", lambda _: SimpleNamespace(role_hash="roles"))
    monkeypatch.setattr(
        runtime,
        "load_a01_manifest",
        lambda p, **_: SimpleNamespace(
            manifest_hash="train" if p.name == "train" else "dev",
            entries=training_entries if p.name == "train" else entries,
        ),
    )
    monkeypatch.setattr(runtime, "ProductionFieldRenderer", BoundedRenderer)
    original_to = torch.Tensor.to

    def cpu_transfer(tensor, *args, **kwargs):
        if (
            args
            and isinstance(args[0], (torch.device, str))
            and torch.device(args[0]).type == "cuda"
        ):
            args = (torch.device("cpu"), *args[1:])
        if kwargs.get("device") is not None and torch.device(kwargs["device"]).type == "cuda":
            kwargs["device"] = torch.device("cpu")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_transfer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "CPU native substitute")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="synthetic")
    )
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 0)
    executions = [0]
    old_execute = RepairActions.execute
    old_candidate = GeometryRepair.execute

    def registered(self, *args, **kwargs):
        executions[0] += 1
        return old_execute(self, *args, **kwargs)

    def candidate(self, *args, **kwargs):
        executions[0] += 1
        return old_candidate(self, *args, **kwargs)

    monkeypatch.setattr(RepairActions, "execute", registered)
    monkeypatch.setattr(GeometryRepair, "execute", candidate)

    class Adapter:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, sid, *, include_supervision, **kwargs):
            if include_supervision:
                assert executions[0] == 13  # STOP + 4 cores x 3 actual action implementations
                return SimpleNamespace(
                    supervision=SimpleNamespace(
                        disparity=torch.full_like(sample.batch.support.float(), 2.0),
                        valid_mask=sample.batch.support,
                    )
                )
            executions[0] = 0
            return ScaredA00Sample(sid, "final_train", sample.batch, sample.calibration_metadata)

    monkeypatch.setattr(runtime, "ScaredA00Adapter", Adapter)
    extra = {}
    if collection_role is not None:
        from dataclasses import asdict

        from reliable_endo_gs.routing.tradeoff import TradeoffConfig

        config = tmp_path / "tradeoff.yaml"
        config.write_text(json.dumps(asdict(TradeoffConfig())))
        extra = dict(
            tradeoff_config_path=config,
            collection_role=collection_role,
            per_dataset=1,
            timing_repeats=1,
        )
    report = evaluate_native_geometry_candidates(
        checkpoint=path,
        probe_report=probe,
        data_root=tmp_path,
        manifest_path=tmp_path / "dev",
        training_manifest=tmp_path / "train",
        roles_path=tmp_path / "roles",
        output_dir=tmp_path / "output",
        **extra,
    )
    assert report["status"] == (
        "native_geometry_comparison_complete"
        if collection_role is None
        else "metric_outcome_table_complete"
    )
    assert report["samples"] == 7 and report["summary"]["correspondence"]["regions"] == 28
    assert report["weights_unchanged"] and not report["training_executed"]
    assert not report["selector_executed"] and not report["teacher_inference_executed"]
    if collection_role is not None:
        from reliable_endo_gs.training.outcome_learning import load_outcome_table

        table = load_outcome_table(tmp_path / "output", collection_role)
        assert table.features.shape == (28, 61)
        assert table.changes.shape == (28, 3, 12)
        assert all("left_ssim" in record["base_metrics"] for record in table.records)


def test_fixed_metric_support_cannot_hide_invalid_depth(native_setup):
    from reliable_endo_gs.evaluation.tradeoff_outcomes import fixed_depth_metrics
    from reliable_endo_gs.geometry.rectified import reconstruct_rectified

    learner, sample, _ = native_setup
    learner.model.eval().requires_grad_(False)
    state = learner.state(sample)
    disparity = state.base.target.disparity
    truth = reconstruct_rectified(
        disparity, state.batch.camera, state.support, min_effective_disparity=1e-6
    )
    good = fixed_depth_metrics(disparity, truth.depth.depth, truth.depth.valid_mask, state)
    assert good == {"mae_mm": 0.0, "rmse_mm": 0.0, "p95_mm": 0.0}
    invalid = disparity.clone()
    invalid[truth.depth.valid_mask] = float("nan")
    lost = fixed_depth_metrics(invalid, truth.depth.depth, truth.depth.valid_mask, state)
    assert all(value is None for value in lost.values())
    empty = fixed_depth_metrics(
        disparity, truth.depth.depth, torch.zeros_like(state.support), state
    )
    assert all(value is None for value in empty.values())


def test_metric_vector_signs_and_missing_ssim_are_not_fabricated():
    from reliable_endo_gs.evaluation.tradeoff_outcomes import metric_changes
    from reliable_endo_gs.routing.tradeoff import COMPONENT_NAMES

    before = {
        key: {"value": 1.0}
        for key in ("left_psnr", "right_psnr", "left_ssim", "right_ssim", "geometry_coverage")
    }
    after = copy.deepcopy(before)
    after["left_psnr"]["value"] = 1.5
    depth_before = {
        key: {"mae_mm": 2.0, "rmse_mm": 3.0, "p95_mm": 4.0} for key in ("global", "local")
    }
    depth_after = {
        key: {"mae_mm": 1.0, "rmse_mm": 2.0, "p95_mm": 3.0} for key in ("global", "local")
    }
    values = metric_changes(before, after, depth_before, depth_after, 0.2, 0.1, 0.4, 0.3)
    assert values is not None
    result = dict(zip(COMPONENT_NAMES, values, strict=True))
    assert result["global_depth_mae_gain_mm"] == 1.0
    assert result["global_depth_p95_gain_mm"] == 1.0
    assert result["left_psnr_gain_db"] == 0.5
    assert result["outside_rgb_gain"] == pytest.approx(0.1)
    assert result["local_rgb_gain"] == pytest.approx(0.1)
    after["right_ssim"]["value"] = None
    assert metric_changes(before, after, depth_before, depth_after, 0.2, 0.1, 0.4, 0.3) is None
