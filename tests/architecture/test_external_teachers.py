"""CPU contract tests of first-party bindings with small fake upstream APIs."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from reliable_endo_gs.teachers.base import TeacherContractError, TeacherUnavailableError
from reliable_endo_gs.teachers.candidates import sha256_file
from reliable_endo_gs.teachers.external import (
    build_defom_stereo,
    build_foundation_stereo,
    load_external_teacher_config,
    source_tree_sha256,
    teacher_preflight,
)
from reliable_endo_gs.teachers.workflow import load_teacher_factory

_MODEL = """import torch
class FakeModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.args = args
        self.calls = []
    def forward(self, left, right, **kwargs):
        self.calls.append((left.clone(), right.clone(), kwargs,
                           torch.is_grad_enabled(), self.training))
        result = self.weight.expand(left.shape[0], 1, *left.shape[-2:]).clone()
        return result
    def run_hierachical(self, left, right, **kwargs):
        return self.forward(left, right, **kwargs)
FoundationStereo = FakeModel
DEFOMStereo = FakeModel
"""

_PADDER = """import torch.nn.functional as F
class InputPadder:
    def __init__(self, shape, divis_by):
        h, w = shape[-2:]
        ph, pw = (-h) % divis_by, (-w) % divis_by
        self.padding = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        self.h, self.w = h, w
    def pad(self, *images):
        return tuple(F.pad(image, self.padding, mode="replicate") for image in images)
    def unpad(self, image):
        l, r, t, b = self.padding
        return image[..., t:t+self.h, l:l+self.w]
"""


class _UnsupportedMetadata:
    """An arbitrary checkpoint class must never be accepted as metadata."""


@pytest.fixture
def external(tmp_path, monkeypatch):
    source = tmp_path / "upstream"
    (source / "core" / "utils").mkdir(parents=True)
    (source / "core" / "__init__.py").write_text("")
    (source / "core" / "utils" / "__init__.py").write_text("")
    (source / "core" / "utils" / "utils.py").write_text(_PADDER)
    for name in ("foundation_stereo", "defom_stereo"):
        (source / "core" / f"{name}.py").write_text(_MODEL)
    (source / "LICENSE").write_text("synthetic test-only license\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Synthetic",
            "-c",
            "user.email=synthetic@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "synthetic fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    checkpoint = tmp_path / "weights.pth"
    torch.save({"model": {"weight": torch.tensor(8.0)}}, checkpoint)
    architecture = tmp_path / "architecture.yaml"
    architecture.write_text("mixed_precision: false\nvit_size: vitl\n")
    raw = {
        "schema_version": 1,
        "model": "foundation_stereo",
        "source_root": str(source),
        "source_revision": revision,
        "source_sha256": source_tree_sha256(source),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "license_file": str(source / "LICENSE"),
        "license_sha256": sha256_file(source / "LICENSE"),
        "architecture": str(architecture),
        "architecture_sha256": sha256_file(architecture),
        "precision": "float32",
        "valid_iters": 7,
        "mode": "regular",
        "small_ratio": 0.5,
        "scale_iters": 3,
        "checkpoint_format": "model",
    }
    path = tmp_path / "teacher.yaml"
    path.write_text(yaml.safe_dump(raw))
    existing = {
        name: module
        for name, module in sys.modules.items()
        if name == "core" or name.startswith("core.")
    }
    for name in existing:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.setitem(
        sys.modules,
        "omegaconf",
        SimpleNamespace(OmegaConf=SimpleNamespace(create=lambda value: dict(value))),
    )
    yield path, raw, source
    for name in tuple(sys.modules):
        if name == "core" or name.startswith("core."):
            sys.modules.pop(name)
    sys.modules.update(existing)


def _rewrite(path, raw, **changes):
    raw.update(changes)
    path.write_text(yaml.safe_dump(raw))


@pytest.mark.parametrize(
    "family,mode",
    [
        ("foundation_stereo", "regular"),
        ("foundation_stereo", "hierarchical"),
        ("defom_stereo", "regular"),
    ],
)
def test_lazy_frozen_rgb_padding_and_native_disparity(external, family, mode):
    path, raw, _ = external
    _rewrite(path, raw, model=family, mode=mode)
    factory = build_foundation_stereo if family == "foundation_stereo" else build_defom_stereo
    teacher = factory(path)
    assert not teacher.model_loaded
    assert f"core.{family}" not in sys.modules
    assert teacher.availability().status == "configured"
    image = torch.linspace(0, 1, 3 * 9 * 13).reshape(1, 3, 9, 13).requires_grad_()
    prediction = teacher.infer(image, image * 0.5)
    assert prediction.disparity.shape == (1, 1, 9, 13)
    assert prediction.disparity.dtype == torch.float32
    assert torch.equal(prediction.disparity, torch.full((1, 1, 9, 13), 8.0))
    assert prediction.validity.all() and prediction.support.all()
    assert prediction.predicted_confidence is None
    assert not prediction.disparity.requires_grad
    model = teacher.load()
    assert not model.training and not model.weight.requires_grad
    left, right, kwargs, grad, training = model.calls[0]
    assert left.shape[-2:] == (32, 32)
    assert left.max() == 255 and left.min() == 0
    assert torch.equal(right, left * 0.5)
    assert not grad and not training
    assert kwargs["iters"] == 7 and kwargs["test_mode"] is True
    if family == "defom_stereo":
        assert kwargs["scale_iters"] == 3
    elif mode == "hierarchical":
        assert kwargs["small_ratio"] == 0.5


def test_preflight_never_deserializes_or_imports_model(external, monkeypatch):
    path, _, _ = external
    # Dependency discovery uses module specs; fake OmegaConf has no spec.
    monkeypatch.delitem(sys.modules, "omegaconf")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("checkpoint deserialized"))
    report = teacher_preflight(path)
    assert report["artifacts_verified"] and report["model_loaded"] is False
    assert report["runtime_verified"] is False
    assert "core.foundation_stereo" not in sys.modules


def test_existing_cli_factory_uses_environment_config(external, monkeypatch):
    path, _, _ = external
    monkeypatch.setenv("RELIABLE_ENDO_TEACHER_CONFIG", str(path))
    teacher = load_teacher_factory("reliable_endo_gs.teachers.external:build_foundation_stereo")
    assert not teacher.model_loaded


@pytest.mark.parametrize("artifact", ["source", "checkpoint", "architecture", "license"])
def test_changed_artifacts_rejected_before_loading(external, artifact):
    path, raw, source = external
    teacher = build_foundation_stereo(path)
    targets = {
        "source": source / "core" / "foundation_stereo.py",
        "checkpoint": Path(raw["checkpoint"]),
        "architecture": Path(raw["architecture"]),
        "license": Path(raw["license_file"]),
    }
    with targets[artifact].open("ab") as handle:
        handle.write(b"\n# changed\n")
    with pytest.raises(TeacherUnavailableError, match="mismatch"):
        teacher.load()


def test_source_namespace_collision_fails_closed(external, monkeypatch, tmp_path):
    path, _, _ = external
    monkeypatch.setitem(
        sys.modules, "core", SimpleNamespace(__file__=str(tmp_path / "other" / "core.py"))
    )
    teacher = build_foundation_stereo(path)
    with pytest.raises(TeacherUnavailableError, match="another project"):
        teacher.load()


def test_disparity_invalidity_is_not_confidence(external):
    path, _, _ = external
    teacher = build_foundation_stereo(path)
    model = teacher.load()
    with torch.no_grad():
        model.weight.fill_(-1)
    prediction = teacher.infer(torch.zeros(1, 3, 5, 7), torch.zeros(1, 3, 5, 7))
    assert not prediction.validity.any()
    assert prediction.support.all() and prediction.predicted_confidence is None


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_input_range_rejected(external, value):
    path, _, _ = external
    teacher = build_foundation_stereo(path)
    image = torch.full((1, 3, 4, 5), value)
    with pytest.raises(TeacherContractError, match="finite A00 RGB"):
        teacher.infer(image, image)


def test_checkpoint_format_is_explicit_and_strict(external):
    path, raw, _ = external
    torch.save({"weight": torch.tensor(4.0)}, raw["checkpoint"])
    _rewrite(path, raw, checkpoint_sha256=sha256_file(Path(raw["checkpoint"])))
    with pytest.raises(TeacherUnavailableError, match="declared state-dict format"):
        build_foundation_stereo(path).load()
    _rewrite(path, raw, checkpoint_format="state_dict")
    assert build_foundation_stereo(path).load().weight == 4
    torch.save({"module.weight": torch.tensor(4.0)}, raw["checkpoint"])
    _rewrite(path, raw, checkpoint_sha256=sha256_file(Path(raw["checkpoint"])))
    with pytest.raises(TeacherUnavailableError, match="state_dict"):
        build_foundation_stereo(path).load()


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
def test_foundation_numpy_metadata_is_scoped_and_weights_only(external, monkeypatch, scalar_type):
    path, raw, _ = external
    torch.save({"model": {"weight": torch.tensor(8.0)}, "metric": scalar_type(0.1)}, raw["checkpoint"])
    _rewrite(path, raw, checkpoint_sha256=sha256_file(Path(raw["checkpoint"])))
    original_load = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["weights_only"] is True and kwargs["map_location"] == "cpu"
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    before = set(torch.serialization.get_safe_globals())
    assert build_foundation_stereo(path).load().weight == 8
    assert calls and calls[0]["mmap"] is True
    assert set(torch.serialization.get_safe_globals()) == before


def test_foundation_omegaconf_training_metadata(external, monkeypatch):
    path, raw, _ = external
    monkeypatch.delitem(sys.modules, "omegaconf")
    from omegaconf import OmegaConf

    torch.save(
        {"model": {"weight": torch.tensor(8.0)}, "args": OmegaConf.create({"dims": [128, 128]})},
        raw["checkpoint"],
    )
    _rewrite(path, raw, checkpoint_sha256=sha256_file(Path(raw["checkpoint"])))
    before = set(torch.serialization.get_safe_globals())
    assert build_foundation_stereo(path).load().weight == 8
    assert set(torch.serialization.get_safe_globals()) == before


def test_foundation_read_preserves_outer_safe_globals(external):
    path, raw, _ = external
    torch.save({"model": {"weight": torch.tensor(8.0)}, "metric": np.float64(0.1)}, raw["checkpoint"])
    _rewrite(path, raw, checkpoint_sha256=sha256_file(Path(raw["checkpoint"])))
    with torch.serialization.safe_globals([type(np.dtype(np.float64))]):
        before = set(torch.serialization.get_safe_globals())
        assert build_foundation_stereo(path).load().weight == 8
        assert set(torch.serialization.get_safe_globals()) == before


def test_unknown_checkpoint_metadata_rejected_before_constructor(external):
    path, raw, _ = external
    torch.save(
        {"model": {"weight": torch.tensor(8.0)}, "unknown": _UnsupportedMetadata()},
        raw["checkpoint"],
    )
    _rewrite(path, raw, checkpoint_sha256=sha256_file(Path(raw["checkpoint"])))
    before = set(torch.serialization.get_safe_globals())
    with pytest.raises(TeacherUnavailableError, match="unsupported FoundationStereo checkpoint globals"):
        build_foundation_stereo(path).load()
    assert "core.foundation_stereo" not in sys.modules
    assert set(torch.serialization.get_safe_globals()) == before


def test_legacy_tensor_checkpoint_keeps_weights_only_behavior(external):
    path, raw, _ = external
    torch.save(
        {"model": {"weight": torch.tensor(8.0)}}, raw["checkpoint"],
        _use_new_zipfile_serialization=False,
    )
    _rewrite(path, raw, checkpoint_sha256=sha256_file(Path(raw["checkpoint"])))
    assert build_foundation_stereo(path).load().weight == 8


def test_defom_does_not_accept_foundation_metadata(external):
    path, raw, _ = external
    torch.save({"model": {"weight": torch.tensor(8.0)}, "metric": np.float64(0.1)}, raw["checkpoint"])
    _rewrite(path, raw, model="defom_stereo", checkpoint_sha256=sha256_file(Path(raw["checkpoint"])))
    before = set(torch.serialization.get_safe_globals())
    with pytest.raises(TeacherUnavailableError, match="Weights only load failed"):
        build_defom_stereo(path).load()
    assert set(torch.serialization.get_safe_globals()) == before


def test_config_pins_and_iteration_changes_invalidate_identity(external):
    path, raw, _ = external
    first = build_foundation_stereo(path).identity.identity_hash
    _rewrite(path, raw, valid_iters=9)
    assert build_foundation_stereo(path).identity.identity_hash != first
    _rewrite(path, raw, checkpoint_sha256=None)
    with pytest.raises(TeacherContractError, match="pinned"):
        load_external_teacher_config(path)


def test_cli_preflight_parser():
    from reliable_endo_gs.cli.main import build_parser

    parsed = build_parser().parse_args(
        ["a01", "teacher-preflight", "--teacher-config", "teacher.yaml"]
    )
    assert parsed.teacher_config == Path("teacher.yaml")


def test_config_portable_paths_and_env(external, monkeypatch):
    path, raw, source = external
    monkeypatch.setenv("TEST_TEACHER_ROOT", str(source))
    _rewrite(path, raw, source_root="${TEST_TEACHER_ROOT}", checkpoint="weights.pth")
    config = load_external_teacher_config(path)
    assert config.source_root == source and config.checkpoint == path.parent / "weights.pth"
    monkeypatch.delenv("TEST_TEACHER_ROOT")
    with pytest.raises(TeacherContractError, match="unresolved environment"):
        load_external_teacher_config(path)


@pytest.mark.parametrize("output", ["tuple", "wrong_grid", "integer"])
def test_wrong_upstream_output_rejected(external, output):
    path, _, _ = external
    teacher = build_foundation_stereo(path)
    model = teacher.load()

    def forward(left, right, **kwargs):
        disparity = torch.zeros((1, 1, *left.shape[-2:]))
        if output == "tuple":
            return (disparity, disparity)
        if output == "wrong_grid":
            return disparity[..., :-1, :]
        return disparity.long()

    model.forward = forward
    image = torch.zeros(1, 3, 5, 7)
    with pytest.raises(TeacherContractError, match="padded shape"):
        teacher.infer(image, image)


def test_unpadding_preserves_pixel_correspondence(external):
    path, _, _ = external
    teacher = build_foundation_stereo(path)
    model = teacher.load()
    model.forward = lambda left, right, **kwargs: left[:, :1] / 255 + 8
    image = torch.linspace(0, 1, 3 * 9 * 13).reshape(1, 3, 9, 13)
    output = teacher.infer(image, image)
    torch.testing.assert_close(output.disparity, image[:, :1] + 8, rtol=0, atol=1e-6)


def test_workflow_uses_sensor_support_without_reading_gt(external):
    from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera
    from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
    from reliable_endo_gs.teachers.workflow import infer_and_align

    class ForbiddenGT:
        def __getattribute__(self, name):
            raise AssertionError(f"teacher read GT: {name}")

    path, _, _ = external
    image = torch.full((1, 3, 5, 7), 0.5)
    intrinsics = torch.tensor([[[100.0, 0, 3], [0, 100.0, 2], [0, 0, 1]]])
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 0, 3] = -0.04
    camera = RectifiedStereoCamera(intrinsics, intrinsics.clone(), transform, True)
    support = torch.ones(1, 1, 5, 7, dtype=torch.bool)
    support[..., 0, 0] = False
    batch = DeploymentStereoBatch(image, image, camera, support)
    sample = ScaredA00Sample(
        "dataset_1/keyframe_1/synthetic",
        "final_train",
        batch,
        {"protocol_identity": "synthetic"},
        supervision=ForbiddenGT(),
    )
    aligned = infer_and_align(sample, build_foundation_stereo(path), device="cpu")
    assert aligned.disparity.shape == (1, 1, 5, 7)
    assert torch.equal(aligned.support, support)
    assert aligned.calibrated_risk is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("valid_iters", True),
        ("valid_iters", 0),
        ("small_ratio", 1.0),
        ("mode", "guess"),
        ("precision", "bfloat16"),
        ("schema_version", True),
    ],
)
def test_bad_runtime_config_rejected(external, field, value):
    path, raw, _ = external
    _rewrite(path, raw, **{field: value})
    with pytest.raises(TeacherContractError):
        load_external_teacher_config(path)
