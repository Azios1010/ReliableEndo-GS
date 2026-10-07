"""CPU screen ordering: real action implementation precedes offline GT access."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from reliable_endo_gs.actions.geometry_repair import GeometryRepair
from reliable_endo_gs.evaluation import geometry_candidates as screen
from reliable_endo_gs.reconstruction.student import StereoStudent
from tests.architecture.test_a04_reference_reconstruction import _batch


def test_cpu_screen_executes_shared_actions_before_gt(tmp_path, monkeypatch):
    torch.manual_seed(4)
    batch = _batch(dtype=torch.float32)
    model = StereoStudent(4, 4, 0.25)
    raw = {
        "identity": {"manifest_hash": "train", "role_hash": "roles", "source_sha256": "old"},
        "student_config": {"width": 4, "bins": 4, "minimum_effective_px": 0.25},
        "model": {"stereo_student." + k: v for k, v in model.state_dict().items()},
        "construction_config": {"runtime": {"resolution_hw": [5, 5]}},
    }
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"synthetic checkpoint identity")
    development = SimpleNamespace(
        manifest_hash="dev",
        entries=[
            SimpleNamespace(sample_id=f"dataset_{i}/keyframe_1/a", data_role="final_train")
            for i in range(1, 8)
        ],
    )
    training = SimpleNamespace(manifest_hash="train", entries=[])
    monkeypatch.setattr(screen, "read_student_checkpoint", lambda _: raw)
    monkeypatch.setattr(screen, "load_a01_roles", lambda _: SimpleNamespace(role_hash="roles"))
    monkeypatch.setattr(
        screen,
        "load_a01_manifest",
        lambda path, **_: training if path.name == "train" else development,
    )
    events = []
    apply = GeometryRepair.apply

    def recorded_apply(self, batch, disparity, core):
        events.append("action")
        assert not disparity.requires_grad
        return apply(self, batch, disparity, core)

    monkeypatch.setattr(GeometryRepair, "apply", recorded_apply)

    class Adapter:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, sid, *, include_supervision, **kwargs):
            events.append("GT" if include_supervision else "deployment")
            return SimpleNamespace(
                batch=batch,
                supervision=SimpleNamespace(
                    disparity=torch.full_like(batch.support.float(), 2.0),
                    valid_mask=batch.support,
                )
                if include_supervision
                else None,
            )

    monkeypatch.setattr(screen, "ScaredA00Adapter", Adapter)
    report = screen.screen_geometry_candidates(
        checkpoint=checkpoint,
        data_root=tmp_path,
        manifest_path=Path("dev"),
        training_manifest=Path("train"),
        roles_path=Path("roles"),
        candidate_configs=(
            Path("configs/routing/geometry_correspondence_candidate.yaml"),
            Path("configs/routing/geometry_support_candidate.yaml"),
        ),
        output_dir=tmp_path / "screen",
        per_dataset=1,
    )
    assert events == (["deployment"] + ["action"] * 8 + ["GT"]) * 7
    assert report["samples"] == 7 and report["device"] == "cpu"
    assert not report["training_executed"] and not report["native_renderer_executed"]
    assert report["model_weights_unchanged"]
    assert report["summary"]["correspondence"]["regions"] == 28
    with pytest.raises(FileExistsError):
        screen.screen_geometry_candidates(
            checkpoint=checkpoint,
            data_root=tmp_path,
            manifest_path=Path("dev"),
            training_manifest=Path("train"),
            roles_path=Path("roles"),
            candidate_configs=(Path("configs/routing/geometry_correspondence_candidate.yaml"),),
            output_dir=tmp_path / "screen",
            per_dataset=1,
        )
