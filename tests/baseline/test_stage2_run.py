"""Synthetic-only identity/launch guards; no datasets, checkpoints or CUDA loaded."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliable_endo_gs.baseline.stage2_bundle import build_stage2_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = REPO_ROOT / "third_party" / "endo_e2e_gs"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(scope="module")
def original_upstream(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("original_upstream")
    archive = subprocess.run(
        ["git", "-C", str(UPSTREAM), "archive", "--format=zip", "HEAD"],
        capture_output=True,
        check=True,
    ).stdout
    with zipfile.ZipFile(io.BytesIO(archive)) as original:
        original.extractall(root)
    return root


@pytest.fixture
def control_fixture(
    tmp_path: Path,
    original_upstream: Path,
    stage2_modules: SimpleNamespace,
) -> SimpleNamespace:
    runtime = stage2_modules.stage2_run
    spec = importlib.util.spec_from_file_location(
        "stage2_config_test", UPSTREAM / "config/stereo_config.py"
    )
    assert spec is not None and spec.loader is not None
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.ConfigStereo()
    config.load(str(UPSTREAM / "config/stage2.yaml"))
    cfg = json.loads(json.dumps(config.get_cfg()))
    data_root = tmp_path / "dataset_3" / "keyframe_1"
    data_root.mkdir(parents=True)
    checkpoint = tmp_path / "synthetic-stage1.bin"
    checkpoint.write_bytes(b"synthetic identity only; not a PyTorch checkpoint")
    cfg["dataset"]["data_root"] = str(data_root)
    cfg["stage1_ckpt"] = str(checkpoint)
    old_cfg = copy.deepcopy(cfg)
    old_cfg["name"] = "synthetic-old-10k-control"
    old_cfg["stage1_ckpt"] = "relocated-original-input.pth"
    for key in (
        "diagnostics_freq",
        "diagnostics_quantiles",
        "scale_boundary_margin",
        "coverage_pixel_threshold",
    ):
        old_cfg["record"].pop(key)
    config_path = tmp_path / "old-cfg.json"
    _write(config_path, old_cfg)
    split = {
        "dataset_id": "dataset_3/keyframe_1",
        "train_frame_ids": [f"t{index:04d}" for index in range(8)],
        "validation_frame_ids": [f"v{index:04d}" for index in range(422)],
    }
    split_path = tmp_path / "control-split.json"
    _write(split_path, split)
    stage1_path = tmp_path / "stage1-record.json"
    _write(
        stage1_path, {"training_steps": 60000, "checkpoint_sha256": runtime.file_sha256(checkpoint)}
    )
    manifest_path = tmp_path / "control.json"
    runtime.make_control_manifest(
        original_upstream,
        config_path,
        split_path,
        checkpoint,
        stage1_path,
        manifest_path,
        "synthetic-only-control",
    )
    return SimpleNamespace(
        runtime=runtime,
        cfg=cfg,
        checkpoint=checkpoint,
        manifest_path=manifest_path,
        config_path=config_path,
        split_path=split_path,
        split=split,
        tmp_path=tmp_path,
    )


def _verify(fixture: SimpleNamespace) -> dict:
    return fixture.runtime.verify_control(fixture.cfg, fixture.manifest_path, UPSTREAM, seed=1314)


def test_control_config_and_actual_split_verified(control_fixture: SimpleNamespace) -> None:
    f = control_fixture
    identity = _verify(f)
    train = SimpleNamespace(frame_ids=f.split["train_frame_ids"], train_idxs=list(range(8)))
    val = SimpleNamespace(frame_ids=f.split["validation_frame_ids"], test_idxs=list(range(422)))
    f.runtime.verify_loaded_split(train, val, identity)
    val.test_idxs.reverse()
    with pytest.raises(ValueError, match="membership/order"):
        f.runtime.verify_loaded_split(train, val, identity)


@pytest.mark.parametrize(
    "field,value", [("num_steps", 3000), ("restore_ckpt", "completed-3k.pth"), ("lr", 0.01)]
)
def test_reject_changed_training_protocol(
    control_fixture: SimpleNamespace, field: str, value: object
) -> None:
    control_fixture.cfg[field] = value
    with pytest.raises(ValueError):
        _verify(control_fixture)


def test_checkpoint_hash_is_checked(control_fixture: SimpleNamespace) -> None:
    control_fixture.checkpoint.write_bytes(b"different synthetic checkpoint")
    with pytest.raises(ValueError, match="Stage1 checkpoint differs"):
        _verify(control_fixture)


def test_split_hash_is_checked(control_fixture: SimpleNamespace) -> None:
    control_fixture.split_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _verify(control_fixture)


def test_dataset6_is_rejected_before_loading(control_fixture: SimpleNamespace) -> None:
    forbidden = control_fixture.tmp_path / "dataset_6" / "keyframe_1"
    forbidden.mkdir(parents=True)
    control_fixture.cfg["dataset"]["data_root"] = str(forbidden)
    with pytest.raises(ValueError, match="no dataset_6"):
        _verify(control_fixture)


def test_source_semantics_must_match_control(control_fixture: SimpleNamespace) -> None:
    f = control_fixture
    manifest = json.loads(f.manifest_path.read_text(encoding="utf-8"))
    manifest["training_signature_sha256"] = "0" * 64
    _write(f.manifest_path, manifest)
    with pytest.raises(ValueError, match="semantics differ"):
        _verify(f)


def test_fresh_directory_and_provenance(control_fixture: SimpleNamespace) -> None:
    f = control_fixture
    identity = _verify(f)
    run_dir = f.tmp_path / "fresh"
    paths = f.runtime.reserve_run_directory(run_dir)
    assert all(Path(value).is_dir() for value in paths.values())
    with pytest.raises(FileExistsError):
        f.runtime.reserve_run_directory(run_dir)
    # Explicit fixture metadata, never presented as actual server hardware.
    f.runtime.write_run_provenance(
        run_dir, f.cfg, identity, UPSTREAM, {"fixture": "synthetic-only"}
    )
    record = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert record["resolved_config_sha256"] == f.runtime.json_sha256(f.cfg)
    assert record["identity"]["split_manifest_sha256"] == f.runtime.file_sha256(f.split_path)
    assert record["hardware"] == {"fixture": "synthetic-only"}
    assert "lib/stage2_run.py" in record["source_sha256"]
    with pytest.raises(FileExistsError):
        f.runtime.write_run_provenance(run_dir, f.cfg, identity, UPSTREAM, {})


def test_transfer_bundle_applies_and_includes_untracked_modules(
    tmp_path: Path,
    original_upstream: Path,
    stage2_modules: SimpleNamespace,
) -> None:
    import shutil

    output = tmp_path / "transfer.zip"
    manifest = build_stage2_bundle(REPO_ROOT, output)
    with pytest.raises(FileExistsError):
        build_stage2_bundle(REPO_ROOT, output)
    with zipfile.ZipFile(output) as bundle:
        assert set(bundle.namelist()) == {"stage2.patch", "manifest.json", "README.md"}
        patch = bundle.read("stage2.patch")
    target = tmp_path / "apply-target"
    shutil.copytree(original_upstream, target)
    # Isolate Git from any repository above the OS temp directory.
    subprocess.run(["git", "init", "--quiet", str(target)], capture_output=True, check=True)
    for args in (["--check", "-"], ["-"]):
        subprocess.run(
            ["git", "apply", *args], cwd=target, input=patch, capture_output=True, check=True
        )
    hashes = stage2_modules.stage2_run.source_hashes(target)
    for relative, digest in manifest["target_source_sha256"].items():
        assert hashes[relative] == digest, relative
    for name in ("scale_parameterization.py", "stage2_diagnostics.py", "stage2_run.py"):
        assert (target / "lib" / name).is_file()
