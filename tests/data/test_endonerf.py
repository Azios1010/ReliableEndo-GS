"""EndoNeRF adapter on a tiny synthetic copy of the dataset layout (the real data are never read by tests)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from reliable_endo_gs.data.endonerf import (
    EXTERNAL_TEST_ROLE,
    EndoNerfAdapter,
    EndoNerfConfig,
    EndoNerfError,
)

H, W = 8, 12


def _png(path: Path, array: np.ndarray, mode: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode=mode).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(tmp_path: Path, *, role: str = "test_only (unit test)") -> tuple[Path, Path]:
    root, sidecar = tmp_path / "data", tmp_path / "sidecar"
    sidecar.mkdir()
    rng = np.random.default_rng(0)
    items = []
    for index in range(2):
        item: dict[str, object] = {"index": index}
        for key, folder, mode, shape in (
            ("left", "images", "RGB", (H, W, 3)),
            ("right", "images_right", "RGB", (H, W, 3)),
            ("depth", "depth", "L", (H, W)),
            ("mask", "masks", "L", (H, W)),
            ("gt_mask", "gt_masks", "L", (H, W)),
        ):
            array = rng.integers(1, 255, size=shape, dtype=np.uint8)
            if key == "mask":
                array[:] = 0
                array[0, :3] = 255
            if key == "gt_mask":
                array[:] = 0
                array[1, :2] = 255
            relative = f"scene_a/{folder}/{index:06d}.png"
            item[key] = {"path": relative, "sha256": _png(root / relative, array, mode)}
        items.append(item)
    (sidecar / "frames_manifest.json").write_text(
        json.dumps(
            {
                "schema": "endonerf_sidecar_manifest.v1",
                "role": role,
                "scenes": {"scene_a": {"items": items}},
            }
        )
    )
    (sidecar / "calibration_assumed.json").write_text(
        json.dumps(
            {
                "status": "assumed",
                "image_hw": [H, W],
                "scenes": {
                    "scene_a": {
                        "focal_px": {"value": 10.0},
                        "principal_point_px": {"value": [5.5, 3.5]},
                        "baseline_m": {"value": 0.004, "source": "estimated"},
                    }
                },
            }
        )
    )
    (sidecar / "calibration_estimate_raw.json").write_text(
        json.dumps({"scenes": {"scene_a": {"b_median_px": 0.5}}})
    )
    return root, sidecar


def test_sample_follows_the_deployment_contract(tmp_path: Path) -> None:
    root, sidecar = _build(tmp_path)
    adapter = EndoNerfAdapter(EndoNerfConfig(root, sidecar, baseline_scale=1.2))
    assert adapter.sample_ids == ["scene_a/000000", "scene_a/000001"]
    sample = adapter.load("scene_a/000001")
    assert sample.data_role == EXTERNAL_TEST_ROLE and sample.supervision is None
    assert tuple(sample.batch.left.shape) == (1, 3, H, W) and sample.batch.support.all()
    assert float(sample.batch.camera.baseline) == pytest.approx(0.0048)
    assert float(sample.batch.camera.K_L[0, 0, 2]) == 5.5
    assert float(sample.batch.camera.K_R[0, 0, 2]) == 5.5  # offset_mode zero
    assert sample.calibration_metadata["scene"] == "scene_a"
    json.dumps(sample.calibration_metadata)  # provenance must be JSON-serialisable


def test_fitted_offset_moves_only_the_right_principal_point(tmp_path: Path) -> None:
    root, sidecar = _build(tmp_path)
    camera = (
        EndoNerfAdapter(EndoNerfConfig(root, sidecar, offset_mode="fitted"))
        .load("scene_a/000000")
        .batch.camera
    )
    assert float(camera.K_L[0, 0, 2] - camera.K_R[0, 0, 2]) == pytest.approx(0.5)
    assert float(camera.K_L[0, 1, 2]) == float(camera.K_R[0, 1, 2])


def test_a_changed_dataset_file_is_refused(tmp_path: Path) -> None:
    root, sidecar = _build(tmp_path)
    path = root / "scene_a/images/000000.png"
    array = np.asarray(Image.open(path)).copy()
    array[0, 0, 0] ^= 1
    Image.fromarray(array).save(path)
    with pytest.raises(EndoNerfError, match="SHA-256"):
        EndoNerfAdapter(EndoNerfConfig(root, sidecar)).load("scene_a/000000")


def test_sidecar_must_declare_test_only(tmp_path: Path) -> None:
    root, sidecar = _build(tmp_path, role="train")
    with pytest.raises(EndoNerfError, match="test_only"):
        EndoNerfAdapter(EndoNerfConfig(root, sidecar))


def test_tool_mask_is_the_union_and_depth_is_read_unscaled(tmp_path: Path) -> None:
    root, sidecar = _build(tmp_path)
    adapter = EndoNerfAdapter(EndoNerfConfig(root, sidecar))
    mask = adapter.tool_mask("scene_a/000000")
    assert mask.dtype == torch.bool and int(mask.sum()) == 5
    depth = adapter.reference_depth("scene_a/000000")
    assert depth.shape == (1, 1, H, W) and float(depth.min()) >= 1.0


def test_invalid_assumptions_are_rejected(tmp_path: Path) -> None:
    root, sidecar = _build(tmp_path)
    with pytest.raises(EndoNerfError):
        EndoNerfConfig(root, sidecar, baseline_scale=0.0)
    with pytest.raises(EndoNerfError):
        EndoNerfConfig(root, sidecar, offset_mode="other")  # type: ignore[arg-type]
