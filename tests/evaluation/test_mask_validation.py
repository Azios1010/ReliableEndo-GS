"""Bounded mask pilot contracts; CPU substitution does not certify CUDA."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from reliable_endo_gs.actions.repair import Action, RepairActions
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.data.target_masks import TargetMaskIndex
from reliable_endo_gs.evaluation.mask_metrics import (
    intervention_metrics,
    observation_metrics,
    summarize_metric_diagnostics,
)
from reliable_endo_gs.regions.masks import MaskRegion
from reliable_endo_gs.routing.tradeoff import TradeoffConfig
from reliable_endo_gs.training.mask_validation import MaskValidationSamples, validate_worklist
from reliable_endo_gs.training.student_checkpoint import save_student_checkpoint
from tests.routing.test_balanced import setup as balanced_fixture
from tests.routing.test_frozen_a04 import AnalyticRenderer
from tests.routing.test_mask_routing import index_file

setup = balanced_fixture


class BoundedRenderer(AnalyticRenderer):
    def render(self, request):
        result = super().render(request)
        return replace(result, image=result.image.sigmoid())


def test_worklist_is_bounded_disjoint_and_preserves_original_manifest_load():
    tr = {f"dataset_{i}/keyframe_1/train" for i in range(1, 8)}
    dv = {s.replace("train", "dev") for s in tr}
    raw = {
        "schema": "mask_validation_worklist.v1",
        "training": sorted(tr),
        "development": sorted(dv),
    }
    a, b = validate_worklist(raw, tr, dv)
    assert set(a) == tr and set(b) == dv
    for changed in (a + a, (*a[:-1], a[0])):
        with pytest.raises(ValueError):
            validate_worklist({**raw, "training": list(changed)}, tr, dv)
    with pytest.raises(ValueError, match="overlaps"):
        validate_worklist({**raw, "development": sorted(tr)}, tr, tr)
    calls = []
    adapter = SimpleNamespace(load=lambda sid, **kw: calls.append((sid, kw)))
    MaskValidationSamples(adapter, a, (512, 640))[0]
    assert calls == [
        (a[0], dict(data_role="final_train", include_supervision=False, output_hw=(512, 640)))
    ]


def test_fixed_metric_zero_action_and_missing_support(setup):
    learner, sample, _ = setup
    object.__setattr__(learner.model, "_renderer", BoundedRenderer())
    with torch.no_grad():
        state = learner.state(sample)
        region = MaskRegion.from_indices(0, (5, 5), (6, 7, 11, 12), 1)
        stop = learner.actions.execute(state, region, Action.STOP, learner.model)
        truth = stop.disparity.clone()
        metrics = observation_metrics(stop, state, truth, state.support)
        result = intervention_metrics(
            stop, stop, state, region, truth, state.support, TradeoffConfig(), metrics, metrics
        )
        assert result["metric_changes"] == [0.0] * 12
        assert result["metric_feasible"] is True
        empty = intervention_metrics(
            stop,
            stop,
            state,
            region,
            truth,
            torch.zeros_like(state.support),
            TradeoffConfig(),
            metrics,
            metrics,
        )
        assert empty["metric_changes"] is None and empty["metric_feasible"] is None


def test_diagnostic_does_not_certify_unknown_or_mix_units():
    config = TradeoffConfig()
    safe = [0.0] * 12
    safe[0] = 0.2
    row = dict(
        region_id=0,
        action="STEREO",
        metric_changes=safe,
        metric_feasible=True,
        component_feasible=[True] * 12,
        within_estimated_budget=True,
    )
    record = {
        "counterfactual_metric_outcomes": [row, dict(row, action="GAUSSIAN", metric_changes=None)],
        "policies": {"selector": {"region": None, "action": "STOP"}},
    }
    out = summarize_metric_diagnostics([record], config)
    assert (
        out["samples_with_budgeted_opportunity"] == out["selector_missed_opportunity_samples"] == 1
    )
    assert out["actions"]["GAUSSIAN"]["unknown"] == 1
    assert out["actions"]["STEREO"]["mean_changes_raw_units"]["global_depth_mae_gain_mm"] == 0.2
    assert "normalized utility" in out["semantics"]
    record["counterfactual_metric_outcomes"][0]["within_estimated_budget"] = False
    assert summarize_metric_diagnostics([record], config)["samples_with_budgeted_opportunity"] == 0


@pytest.mark.parametrize("quiet", [False, True])
@pytest.mark.parametrize("base_only", [False, True])
def test_native_evaluator_mask_checkpoint_cursor_gt_order_and_immutability(
    setup, tmp_path, monkeypatch, quiet, base_only
):
    import reliable_endo_gs.evaluation.intervention_selection as runtime
    from reliable_endo_gs.routing.masks import MaskProposalConfig
    from reliable_endo_gs.training.student_learning import StudentLearner

    learner, sample, target = setup
    object.__setattr__(learner.model, "_renderer", BoundedRenderer())
    learner.model.config = replace(
        learner.model.config,
        renderer_identity=AnalyticRenderer.identity,
        runtime=replace(learner.model.config.runtime, precision="fp32", resolution_hw=(5, 5)),
    )
    # Fit the cold normalizer and record real shared-action costs synthetically.
    for i in range(0 if base_only else 2):
        learner.step(sample, joint=True, teacher_target=target, position=i)
    index = index_file(
        tmp_path / "roi",
        torch.full((5, 5), not quiet, dtype=torch.bool),
        sample_id="dataset_1/keyframe_1/dev",
    )
    masks = TargetMaskIndex(index)
    masked = StudentLearner(
        copy.deepcopy(learner.model),
        learner.config,
        learner.routing,
        intervention_config=learner.intervention_config,
        mask_config=MaskProposalConfig(
            threshold=0.0, cell_size=2, max_pixels=12, max_primitives=12
        ),
        target_masks=masks,
    )
    masked.actions.load_state_dict(learner.actions.state_dict())
    masked.selector.load_state_dict(learner.selector.state_dict())
    masked.cost_max_ms.copy_(learner.cost_max_ms)
    masked.cost_observations = learner.cost_observations
    for name in ("model_optimizer", "action_optimizer", "selector_optimizer"):
        getattr(masked, name).load_state_dict(getattr(learner, name).state_dict())
    tr = [f"dataset_{i}/keyframe_1/train" for i in range(1, 8)]
    identity = dict(
        run_kind="bounded_mask_validation",
        source_sha256="source",
        manifest_hash="train",
        role_hash="roles",
        renderer_identity=AnalyticRenderer.identity,
        training_worklist=tr,
        samples_per_epoch=7,
    )
    checkpoint = tmp_path / "mask.pt"
    save_student_checkpoint(
        checkpoint,
        masked,
        identity,
        dict(
            epoch=0,
            next_sample_index=0 if base_only else 2,
            samples_seen=0 if base_only else 2,
            optimizer_steps=0 if base_only else 2,
        ),
    )
    monkeypatch.setattr(runtime, "source_sha256", lambda _: "source")
    monkeypatch.setattr(runtime, "load_a01_roles", lambda _: SimpleNamespace(role_hash="roles"))
    monkeypatch.setattr(
        runtime,
        "load_a01_manifest",
        lambda p, **_: SimpleNamespace(
            manifest_hash=p.name,
            entries=[
                SimpleNamespace(sample_id=s, data_role="final_train")
                for s in (tr if p.name == "train" else ["dataset_1/keyframe_1/dev"])
            ],
        ),
    )
    monkeypatch.setattr(runtime, "ProductionFieldRenderer", BoundedRenderer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "synthetic CPU renderer")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="synthetic")
    )
    calls = []
    old_execute = RepairActions.execute

    def execute(self, *args, **kwargs):
        calls.append("action")
        return old_execute(self, *args, **kwargs)

    monkeypatch.setattr(RepairActions, "execute", execute)

    class Adapter:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, sid, *, include_supervision, **kwargs):
            if include_supervision:
                assert calls and calls[-1] == "action"
                calls.append("GT")
                return SimpleNamespace(
                    supervision=SimpleNamespace(
                        disparity=torch.full_like(sample.batch.support.float(), 2.0),
                        valid_mask=sample.batch.support,
                    )
                )
            return ScaredA00Sample(sid, "final_train", sample.batch, sample.calibration_metadata)

    monkeypatch.setattr(runtime, "ScaredA00Adapter", Adapter)
    metric = tmp_path / "metric.yaml"
    metric.write_text(json.dumps({"schema": "metric_action_tradeoff.v1"}))
    rng_before = torch.get_rng_state().clone()
    report = runtime.evaluate_intervention_selection(
        checkpoint=checkpoint,
        data_root=tmp_path,
        manifest_path=tmp_path / "dev",
        training_manifest=tmp_path / "train",
        roles_path=tmp_path / "roles",
        output_dir=tmp_path / "result",
        device_name="cpu",
        max_samples=1,
        target_mask_index=index,
        sample_ids=("dataset_1/keyframe_1/dev",),
        metric_config_path=metric,
        base_only=base_only,
    )
    assert report["weights_unchanged"] and not report["teacher_inference_executed"]
    assert report["samples"] == 1 and report["region_protocol"] == "target_mask_proposals.v1"
    record = json.loads((tmp_path / "result/samples.jsonl").read_text())
    assert torch.equal(torch.get_rng_state(), rng_before)
    if quiet or base_only:
        assert not record["mask_candidates"] and not record["counterfactual_metric_outcomes"]
        assert all(r["action"] == "STOP" for r in record["policies"].values())
    else:
        assert record["mask_candidates"] and record["counterfactual_metric_outcomes"]
    assert calls[-1] == "GT"


@pytest.mark.parametrize("preflight_only", [True, False])
def test_orchestrator_cpu_preflight_has_no_gt_or_teacher_and_stops_without_cuda(
    setup, tmp_path, monkeypatch, preflight_only
):
    import reliable_endo_gs.training.mask_validation as runtime

    learner, sample, target = setup
    tr = [f"dataset_{i}/keyframe_1/train" for i in range(1, 8)]
    dev = [sid.replace("train", "dev") for sid in tr]
    work = tmp_path / "work.json"
    work.write_text(
        json.dumps(
            dict(
                schema="mask_validation_worklist.v1",
                training=tr,
                development=dev,
                training_manifest_hash="train",
                development_manifest_hash="dev",
                role_hash="roles",
                target_mask_index_sha256="ROI",
            )
        )
    )
    monkeypatch.setattr(runtime, "load_a01_roles", lambda _: SimpleNamespace(role_hash="roles"))
    monkeypatch.setattr(
        runtime,
        "load_a01_manifest",
        lambda p, **_: SimpleNamespace(
            manifest_hash=p.name,
            entries=[
                SimpleNamespace(sample_id=s, data_role="final_train")
                for s in (tr if p.name == "train" else dev)
            ],
        ),
    )
    construction = replace(
        learner.model.config,
        runtime=replace(
            learner.model.config.runtime, device="cuda:0", precision="fp32", resolution_hw=(5, 5)
        ),
    )
    monkeypatch.setattr(
        runtime,
        "read_student_checkpoint",
        lambda _: dict(
            construction_config=construction.to_dict(),
            identity=dict(manifest_hash="train", role_hash="roles"),
        ),
    )
    reads = []

    class Adapter:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, sid, *, include_supervision, **kwargs):
            assert not include_supervision
            reads.append(sid)
            return ScaredA00Sample(sid, "final_train", sample.batch, {})

        def validate_deployment_files(self, sid, **kwargs):
            reads.append(sid)

    class Masks:
        identity = "ROI"

        def __init__(self, *args):
            pass

        def require_samples(self, *args):
            pass

        def load(self, *args):
            assert args[-1].type == "cpu"

    class Cache:
        def __init__(self, *args, read_only, **kwargs):
            assert read_only

        def get(self, key):
            return target

    monkeypatch.setattr(runtime, "ScaredA00Adapter", Adapter)
    monkeypatch.setattr(runtime, "TargetMaskIndex", Masks)
    monkeypatch.setattr(runtime, "TargetCache", Cache)
    monkeypatch.setattr(runtime, "build_foundation_stereo", lambda: object())
    monkeypatch.setattr(
        runtime,
        "target_cache_key",
        lambda sample, *a, **kw: SimpleNamespace(digest=sample.sample_id),
    )
    monkeypatch.setattr(runtime, "source_sha256", lambda _: "source")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    kwargs = dict(
        data_root=tmp_path,
        training_manifest=tmp_path / "train",
        development_manifest=tmp_path / "dev",
        roles_path=tmp_path / "roles",
        worklist_path=work,
        target_mask_index=tmp_path / "roi",
        cache_root=tmp_path / "cache",
        initial_checkpoint=tmp_path / "initial.pt",
        output_dir=tmp_path / "out",
        mask_config_path=Path("configs/routing/target_masks.yaml"),
        intervention_config_path=Path("configs/routing/intervention_balanced.yaml"),
        routing_config_path=Path("configs/routing/frozen_a04.yaml"),
        metric_config_path=Path("configs/routing/metric_tradeoff_balanced.yaml"),
        preflight_only=preflight_only,
    )
    if preflight_only:
        result = runtime.run_mask_validation(**kwargs)
        assert result["cache_hits"] == 7 and not result["ground_truth_loaded"]
    else:
        with pytest.raises(ValueError, match="CUDA is unavailable"):
            runtime.run_mask_validation(**kwargs)
    assert reads == tr + dev
    report = json.loads((tmp_path / "out/preflight.json").read_text())
    assert not report["teacher_inference_executed"] and not report["training_executed"]
    assert not (tmp_path / "out/initial.pt").exists()
