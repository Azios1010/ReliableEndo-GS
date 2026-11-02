"""Read-only adapter for the EndoNeRF sample data (external, test-only).

The dataset terms forbid editing, remixing or redistributing the material, so nothing is copied or rewritten: the adapter reads
the dataset files where they are and takes everything the dataset does not provide (frame pairing, file hashes, an assumed
calibration) from a *sidecar* folder. The sidecar holds ``frames_manifest.json`` (paired files and SHA-256 of every dataset
file used) and ``calibration_assumed.json`` / ``calibration_estimate_raw.json`` (focal length from the dataset; principal point and
baseline are assumptions, see the sidecar README). Every file is verified against its manifest hash when it is read.

Tensor conventions follow the A00 deployment contract: RGB ``[1, 3, H, W]`` in ``[0, 1]``, integer pixel centres, rectified left/right
cameras with ``T_R_from_L`` = identity and an x translation of ``-B`` metres. Images are used at their native 512 x 640; no resampling.
The calibration is not a measurement. ``baseline_scale`` and ``offset_mode`` expose the two assumptions for sensitivity runs.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample

EXTERNAL_TEST_ROLE = "external_test"
OffsetMode = Literal["zero", "fitted"]


class EndoNerfError(ValueError):
    """Raised when the dataset, its sidecar or a requested frame is missing or inconsistent."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class EndoNerfConfig:
    """Locations and the two calibration assumptions. Paths come from the caller (spec or environment), never from the code."""

    dataset_root: Path
    sidecar_dir: Path
    baseline_scale: float = 1.0
    offset_mode: OffsetMode = "zero"

    def __post_init__(self) -> None:
        if not self.baseline_scale > 0:
            raise EndoNerfError("baseline_scale must be positive")
        if self.offset_mode not in ("zero", "fitted"):
            raise EndoNerfError("offset_mode must be 'zero' or 'fitted'")


class EndoNerfAdapter:
    """Frames of the sample scenes as deployment samples (no ground truth is ever attached)."""

    def __init__(self, config: EndoNerfConfig) -> None:
        self.config = config
        manifest_path = config.sidecar_dir / "frames_manifest.json"
        calibration_path = config.sidecar_dir / "calibration_assumed.json"
        estimate_path = config.sidecar_dir / "calibration_estimate_raw.json"
        for path in (manifest_path, calibration_path, estimate_path):
            if not path.is_file():
                raise EndoNerfError(f"sidecar file missing: {path.name}")
        self.manifest_sha256 = sha256_bytes(manifest_path.read_bytes())
        self._manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        self._calibration: dict[str, Any] = json.loads(calibration_path.read_text())
        self._estimate: dict[str, Any] = json.loads(estimate_path.read_text())
        if self._manifest.get("schema") != "endonerf_sidecar_manifest.v1":
            raise EndoNerfError("unsupported sidecar manifest schema")
        if not str(self._manifest.get("role", "")).startswith("test_only"):
            raise EndoNerfError("the sidecar must declare role test_only")
        self._frames: dict[str, dict[str, Any]] = {}
        for scene, entry in self._manifest["scenes"].items():
            for item in entry["items"]:
                self._frames[f"{scene}/{item['index']:06d}"] = {"scene": scene, **item}

    @property
    def sample_ids(self) -> list[str]:
        """Frame ids in manifest order: ``<scene>/<six-digit index>``."""
        return list(self._frames)

    def _read_verified(self, sample_id: str, key: str) -> bytes:
        frame = self._frames.get(sample_id)
        if frame is None:
            raise EndoNerfError(f"unknown frame id {sample_id!r}")
        record = frame[key]
        path = self.config.dataset_root / record["path"]
        data: bytes = path.read_bytes()
        if sha256_bytes(data) != record["sha256"]:
            raise EndoNerfError(f"{path.name}: SHA-256 differs from the sidecar manifest")
        return data

    @staticmethod
    def _decode(data: bytes, mode: str) -> Any:
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert(mode), dtype=np.uint8).copy()

    def _camera(self, scene: str, height: int, width: int) -> RectifiedStereoCamera:
        entry = self._calibration["scenes"][scene]
        focal = float(entry["focal_px"]["value"])
        cx, cy = (float(v) for v in entry["principal_point_px"]["value"])
        if (height, width) != tuple(self._calibration["image_hw"]):
            raise EndoNerfError("frame size differs from the sidecar calibration")
        offset = 0.0
        if self.config.offset_mode == "fitted":
            offset = float(self._estimate["scenes"][scene]["b_median_px"])
        baseline_m = float(entry["baseline_m"]["value"]) * self.config.baseline_scale
        k_left = torch.tensor([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]]).unsqueeze(0)
        k_right = k_left.clone()
        k_right[:, 0, 2] = (
            cx - offset
        )  # offset = cx_left - cx_right (geometry.rectified: d_eff = d - offset)
        transform = torch.eye(4).unsqueeze(0)
        transform[:, 0, 3] = -baseline_m
        return RectifiedStereoCamera(k_left, k_right, transform, True)

    def load(self, sample_id: str, *, device: torch.device | str | None = None) -> ScaredA00Sample:
        """One deployment sample at the native 512 x 640 grid. ``supervision`` is always ``None``."""
        frame = self._frames.get(sample_id)
        if frame is None:
            raise EndoNerfError(f"unknown frame id {sample_id!r}")
        destination = torch.device("cpu" if device is None else device)
        images = []
        for key in ("left", "right"):
            array = self._decode(self._read_verified(sample_id, key), "RGB")
            images.append(torch.from_numpy(array).permute(2, 0, 1).float().div(255.0))
        height, width = int(images[0].shape[1]), int(images[0].shape[2])
        if images[1].shape != images[0].shape:
            raise EndoNerfError("left and right frames differ in size")
        camera = self._camera(frame["scene"], height, width)
        camera = RectifiedStereoCamera(
            camera.K_L.to(destination),
            camera.K_R.to(destination),
            camera.T_R_from_L.to(destination),
            True,
        )
        left, right = (image.unsqueeze(0).to(destination) for image in images)
        support = torch.ones((1, 1, height, width), dtype=torch.bool, device=destination)
        batch = DeploymentStereoBatch(left, right, camera, support)
        calibration = self._calibration["scenes"][frame["scene"]]
        metadata: dict[str, object] = {
            "protocol_schema": "endonerf_sidecar.v1",
            "image_range": "uint8_0_255",
            "rectified": "assumed",
            "pixel_center": "integer",
            "disparity_units": "input_grid_pixels",
            "disparity_sign": "u_left_minus_right",
            "extrinsic_units": "m",
            "extrinsic_direction": "right_from_left",
            "baseline_m": float(-camera.T_R_from_L[0, 0, 3]),
            "baseline_scale": self.config.baseline_scale,
            "principal_point_offset_px": float(camera.K_L[0, 0, 2] - camera.K_R[0, 0, 2]),
            "offset_mode": self.config.offset_mode,
            "calibration_status": str(self._calibration["status"]),
            "baseline_source": str(calibration["baseline_m"]["source"]),
            "input_hw": [height, width],
            "output_hw": [height, width],
            "scene": frame["scene"],
            "frame_index": int(frame["index"]),
            "sidecar_manifest_sha256": self.manifest_sha256,
            "source_file_hashes": {k: frame[k]["sha256"] for k in ("left", "right")},
        }
        return ScaredA00Sample(sample_id, EXTERNAL_TEST_ROLE, batch, metadata, None)

    def tool_mask(
        self, sample_id: str, *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """Pixels of either dataset mask image (bool ``[1, 1, H, W]``): the union that evaluations exclude as 'tool'."""
        masks = [
            self._decode(self._read_verified(sample_id, key), "L") > 0
            for key in ("mask", "gt_mask")
        ]
        union = masks[0] | masks[1]
        return torch.from_numpy(union).reshape(1, 1, *union.shape).to(device or "cpu")

    def reference_depth(
        self, sample_id: str, *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """The dataset's own stereo-derived depth PNG as float ``[1, 1, H, W]`` (unit unknown, 0 = invalid). Not ground truth."""
        array = self._decode(self._read_verified(sample_id, "depth"), "L")
        return torch.from_numpy(array).float().reshape(1, 1, *array.shape).to(device or "cpu")
