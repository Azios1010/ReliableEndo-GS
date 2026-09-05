"""Manifest-backed multi-sequence SCARED dataset and Stage-1 tensor contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from reliable_endo_gs.data.scared_manifest import (
    SplitManifest,
    load_scared_split_manifest,
    make_scared_sample_id,
    parse_scared_keyframe_id,
    validate_scared_split_manifest,
)

# Canonical per-dataset skip rates from upstream SCARED_Dataset
UPSTREAM_SKIP_EVERY: dict[str, int] = {
    "dataset_1": 2,
    "dataset_2": 1,
    "dataset_3": 4,
    "dataset_6": 8,
    "dataset_7": 8,
}
DEFAULT_SKIP_EVERY: int = 2


class ScaredStage1Sample(dict):
    """Stage-1 sample contract container with tensor and provenance properties.

    Inherits from dict so that PyTorch ``DataLoader(batch_size=1, num_workers=0)``
    and batch collation work transparently.
    """

    def __init__(
        self,
        *,
        dataset_id: str,
        keyframe_id: str,
        frame_id: str,
        left: torch.Tensor,
        right: torch.Tensor,
        disparity: torch.Tensor,
        intr: torch.Tensor,
        extr: torch.Tensor,
        Q: torch.Tensor,
        right_intr: torch.Tensor | None = None,
        sample_id: str | None = None,
        **extra: Any,
    ) -> None:
        sid = sample_id or make_scared_sample_id(dataset_id, keyframe_id, frame_id)
        r_intr = right_intr if right_intr is not None else intr.clone()
        payload: dict[str, Any] = {
            "dataset_id": dataset_id,
            "keyframe_id": keyframe_id,
            "frame_id": frame_id,
            "sample_id": sid,
            "left": left,
            "right": right,
            "disparity": disparity,
            "disp": disparity,
            "intr": intr,
            "right_intr": r_intr,
            "extr": extr,
            "Q": Q,
        }
        payload.update(extra)
        super().__init__(payload)

    @property
    def dataset_id(self) -> str:
        return self["dataset_id"]

    @property
    def keyframe_id(self) -> str:
        return self["keyframe_id"]

    @property
    def frame_id(self) -> str:
        return self["frame_id"]

    @property
    def sample_id(self) -> str:
        return self["sample_id"]

    @property
    def left(self) -> torch.Tensor:
        return self["left"]

    @property
    def right(self) -> torch.Tensor:
        return self["right"]

    @property
    def disparity(self) -> torch.Tensor:
        return self["disparity"]

    @property
    def intr(self) -> torch.Tensor:
        return self["intr"]

    @property
    def right_intr(self) -> torch.Tensor:
        return self["right_intr"]

    @property
    def extr(self) -> torch.Tensor:
        return self["extr"]

    @property
    def Q(self) -> torch.Tensor:
        return self["Q"]

    def to_native_dict(self) -> dict[str, Any]:
        """Convert into the dictionary representation expected by upstream models."""
        return {
            "name": self.sample_id,
            "lmain": {
                "img": self.left,
                "disp": self.disparity.unsqueeze(0),
                "intr": self.intr,
                "extr": self.extr,
                "Q": self.Q,
            },
            "rmain": {
                "img": self.right,
                "intr": self.right_intr,
            },
        }


def validate_stage1_sample(sample: Mapping[str, Any]) -> None:
    """Strict shape, type, range, and finite checks for Stage-1 tensor contracts.

    Enforces:
    - Provenance: dataset_id, keyframe_id, frame_id non-empty strings.
    - Images: [3, H, W] float32 in [-1.0, 1.0], all finite.
    - Disparity: [H, W] float32 matching image spatial dims, all finite.
    - Intrinsics: [3, 3] float32, all finite.
    - Extrinsics: [3, 4] float32, all finite.
    - Q matrix: [4, 4] float32, all finite.
    """
    for key in ("dataset_id", "keyframe_id", "frame_id"):
        if key not in sample:
            raise ValueError(f"sample missing required provenance field {key!r}")
        val = sample[key]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"provenance field {key!r} must be a non-empty string; got {val!r}")

    # Check left image
    if "left" not in sample:
        raise ValueError("sample missing 'left' image tensor")
    left = sample["left"]
    if not isinstance(left, torch.Tensor):
        raise TypeError(f"'left' must be a torch.Tensor, got {type(left).__name__}")
    if left.ndim != 3 or left.shape[0] != 3:
        raise ValueError(f"'left' must have shape [3, H, W]; got {tuple(left.shape)}")
    if left.dtype != torch.float32:
        raise TypeError(f"'left' must have dtype float32; got {left.dtype}")
    if not torch.isfinite(left).all():
        raise ValueError("'left' image contains non-finite (NaN or Inf) values")
    if (left < -1.0001).any() or (left > 1.0001).any():
        raise ValueError(
            f"'left' values must fall within [-1.0, 1.0]; got range [{left.min().item():.3f}, {left.max().item():.3f}]"
        )

    _, h, w = left.shape

    # Check right image
    if "right" not in sample:
        raise ValueError("sample missing 'right' image tensor")
    right = sample["right"]
    if not isinstance(right, torch.Tensor):
        raise TypeError(f"'right' must be a torch.Tensor, got {type(right).__name__}")
    if right.ndim != 3 or right.shape != (3, h, w):
        raise ValueError(f"'right' shape {tuple(right.shape)} must match 'left' shape {(3, h, w)}")
    if right.dtype != torch.float32:
        raise TypeError(f"'right' must have dtype float32; got {right.dtype}")
    if not torch.isfinite(right).all():
        raise ValueError("'right' image contains non-finite (NaN or Inf) values")
    if (right < -1.0001).any() or (right > 1.0001).any():
        raise ValueError(
            f"'right' values must fall within [-1.0, 1.0]; got range [{right.min().item():.3f}, {right.max().item():.3f}]"
        )

    # Check disparity
    if "disparity" not in sample:
        raise ValueError("sample missing 'disparity' tensor")
    disp = sample["disparity"]
    if not isinstance(disp, torch.Tensor):
        raise TypeError(f"'disparity' must be a torch.Tensor, got {type(disp).__name__}")
    if disp.ndim != 2 or disp.shape != (h, w):
        raise ValueError(f"'disparity' must have shape [H, W] = {(h, w)}; got {tuple(disp.shape)}")
    if disp.dtype != torch.float32:
        raise TypeError(f"'disparity' must have dtype float32; got {disp.dtype}")
    if not torch.isfinite(disp).all():
        raise ValueError("'disparity' map contains non-finite (NaN or Inf) values")

    # Check intr
    if "intr" not in sample:
        raise ValueError("sample missing 'intr' tensor")
    intr = sample["intr"]
    if not isinstance(intr, torch.Tensor):
        raise TypeError(f"'intr' must be a torch.Tensor, got {type(intr).__name__}")
    if intr.shape != (3, 3):
        raise ValueError(f"'intr' must have shape [3, 3]; got {tuple(intr.shape)}")
    if intr.dtype != torch.float32:
        raise TypeError(f"'intr' must have dtype float32; got {intr.dtype}")
    if not torch.isfinite(intr).all():
        raise ValueError("'intr' contains non-finite (NaN or Inf) values")

    # Check extr
    if "extr" not in sample:
        raise ValueError("sample missing 'extr' tensor")
    extr = sample["extr"]
    if not isinstance(extr, torch.Tensor):
        raise TypeError(f"'extr' must be a torch.Tensor, got {type(extr).__name__}")
    if extr.shape != (3, 4):
        raise ValueError(f"'extr' must have shape [3, 4]; got {tuple(extr.shape)}")
    if extr.dtype != torch.float32:
        raise TypeError(f"'extr' must have dtype float32; got {extr.dtype}")
    if not torch.isfinite(extr).all():
        raise ValueError("'extr' contains non-finite (NaN or Inf) values")

    # Check Q
    if "Q" not in sample:
        raise ValueError("sample missing 'Q' tensor")
    Q = sample["Q"]
    if not isinstance(Q, torch.Tensor):
        raise TypeError(f"'Q' must be a torch.Tensor, got {type(Q).__name__}")
    if Q.shape != (4, 4):
        raise ValueError(f"'Q' must have shape [4, 4]; got {tuple(Q.shape)}")
    if Q.dtype != torch.float32:
        raise TypeError(f"'Q' must have dtype float32; got {Q.dtype}")
    if not torch.isfinite(Q).all():
        raise ValueError("'Q' contains non-finite (NaN or Inf) values")


def _read_image(path: Path) -> np.ndarray:
    """Read an 8-bit RGB image as an [H, W, 3] uint8 numpy array."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            rgb = img.convert("RGB")
            return np.array(rgb, dtype=np.uint8)
    except Exception as error:
        raise OSError(f"unable to read image {path}: {error}") from error


def _read_tiff(path: Path) -> np.ndarray:
    """Read a single-channel 2D float32 disparity TIFF."""
    try:
        import tifffile

        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        return np.ascontiguousarray(arr, dtype=np.float32)
    except Exception:
        try:
            import imageio.v2 as iio

            arr = iio.imread(path)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr.squeeze(-1)
            return np.ascontiguousarray(arr, dtype=np.float32)
        except Exception as error:
            raise OSError(f"unable to read disparity TIFF {path}: {error}") from error


class ScaredKeyframeDataset(Dataset):
    """Dataset for one validated, processed SCARED keyframe directory.

    Reads from:
    - data/frame_data/{frame_id}.json
    - data/left_finalpass/{frame_id}.png
    - data/right_finalpass/{frame_id}.png
    - data/disparity/{frame_id}.tiff
    - data/newpram_data/{frame_id}.json
    - data/reprojection_data/{frame_id}.json
    """

    def __init__(
        self,
        keyframe_dir: Path,
        dataset_id: str,
        keyframe_id: str,
        *,
        skip_every: int | None = None,
        phase: str = "all",
        test_every: int = 8,
        downsample: float = 1.0,
    ) -> None:
        self.keyframe_dir = Path(keyframe_dir)
        self.dataset_id = dataset_id
        self.keyframe_id = keyframe_id
        self.downsample = float(downsample)
        self.phase = phase

        if not self.keyframe_dir.is_dir():
            raise FileNotFoundError(f"keyframe directory does not exist: {self.keyframe_dir}")

        # Resolve subdirectories with standard and fallback paths
        data_parent = self.keyframe_dir / "data" if (self.keyframe_dir / "data").is_dir() else self.keyframe_dir
        self.calibs_dir = data_parent / "frame_data"
        self.rgbl_dir = data_parent / "left_finalpass"
        self.rgbr_dir = data_parent / "right_finalpass"
        self.disps_dir = data_parent / "disparity"
        self.newpram_dir = data_parent / "newpram_data"
        self.reproj_dir = data_parent / "reprojection_data"

        for name, dirpath in (
            ("frame_data", self.calibs_dir),
            ("left_finalpass", self.rgbl_dir),
            ("right_finalpass", self.rgbr_dir),
            ("disparity", self.disps_dir),
            ("newpram_data", self.newpram_dir),
            ("reprojection_data", self.reproj_dir),
        ):
            if not dirpath.is_dir():
                raise FileNotFoundError(
                    f"keyframe {keyframe_dir} is missing required subdirectory {name!r} ({dirpath})"
                )

        all_calib_files = sorted(self.calibs_dir.glob("*.json"))
        if not all_calib_files:
            raise ValueError(f"no frame JSON files found in {self.calibs_dir}")

        raw_frame_ids = [p.stem for p in all_calib_files]

        if skip_every is None:
            skip_every = UPSTREAM_SKIP_EVERY.get(dataset_id, DEFAULT_SKIP_EVERY)
        self.skip_every = max(1, skip_every)

        skipped_frame_ids = raw_frame_ids[::self.skip_every]
        n_frames = len(skipped_frame_ids)

        if phase == "all":
            selected_indices = list(range(n_frames))
        elif phase == "train":
            selected_indices = [i for i in range(n_frames) if (i - 1) % test_every != 0]
        elif phase in ("val", "test"):
            selected_indices = [i for i in range(n_frames) if (i - 1) % test_every == 0]
        else:
            raise ValueError(f"unrecognized phase {phase!r}; expected 'all', 'train', or 'val'")

        self.frame_ids = [skipped_frame_ids[i] for i in selected_indices]
        if not self.frame_ids:
            raise ValueError(f"no frames selected for phase {phase!r} in {keyframe_dir}")

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, index: int) -> ScaredStage1Sample:
        if index < 0 or index >= len(self.frame_ids):
            raise IndexError(f"index {index} out of range for dataset of size {len(self.frame_ids)}")

        frame_id = self.frame_ids[index]

        # Read RGB images
        left_arr = _read_image(self.rgbl_dir / f"{frame_id}.png")
        right_arr = _read_image(self.rgbr_dir / f"{frame_id}.png")

        # Convert to float32 tensors in [-1.0, 1.0] with shape [3, H, W]
        left_t = torch.from_numpy(left_arr).permute(2, 0, 1).to(dtype=torch.float32)
        left_t = (left_t / 127.5) - 1.0
        left_t = left_t.clamp(-1.0, 1.0)

        right_t = torch.from_numpy(right_arr).permute(2, 0, 1).to(dtype=torch.float32)
        right_t = (right_t / 127.5) - 1.0
        right_t = right_t.clamp(-1.0, 1.0)

        # Read disparity map
        disp_arr = _read_tiff(self.disps_dir / f"{frame_id}.tiff")
        disp_t = torch.from_numpy(disp_arr).to(dtype=torch.float32)

        # Read camera intrinsics and extrinsics
        with open(self.newpram_dir / f"{frame_id}.json", "r", encoding="utf-8") as f:
            newpram = json.load(f)
        intr_t = torch.tensor(newpram["intr0"], dtype=torch.float32)
        right_intr_t = torch.tensor(newpram["intr1"], dtype=torch.float32)
        extr_t = torch.tensor(newpram["extr0"], dtype=torch.float32)

        # Read reprojection matrix Q
        with open(self.reproj_dir / f"{frame_id}.json", "r", encoding="utf-8") as f:
            reproj = json.load(f)
        Q_t = torch.tensor(reproj["reprojection-matrix"], dtype=torch.float32)

        # Apply downsampling if requested
        if self.downsample != 1.0:
            inv_scale = 1.0 / self.downsample
            h, w = int(left_t.shape[1] * inv_scale), int(left_t.shape[2] * inv_scale)
            left_t = functional.interpolate(
                left_t.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)
            right_t = functional.interpolate(
                right_t.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)
            disp_t = functional.interpolate(
                disp_t.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest"
            ).squeeze(0).squeeze(0) * inv_scale

            # Scale camera matrices
            intr_t[0, 0] *= inv_scale
            intr_t[0, 2] *= inv_scale
            intr_t[1, 1] *= inv_scale
            intr_t[1, 2] *= inv_scale

            right_intr_t[0, 0] *= inv_scale
            right_intr_t[0, 2] *= inv_scale
            right_intr_t[1, 1] *= inv_scale
            right_intr_t[1, 2] *= inv_scale

        sample = ScaredStage1Sample(
            dataset_id=self.dataset_id,
            keyframe_id=self.keyframe_id,
            frame_id=frame_id,
            left=left_t,
            right=right_t,
            disparity=disp_t,
            intr=intr_t,
            extr=extr_t,
            Q=Q_t,
            right_intr=right_intr_t,
        )
        validate_stage1_sample(sample)
        return sample


class ScaredMultiSequenceDataset(Dataset):
    """Manifest-backed multi-sequence SCARED dataset.

    Composes validated per-keyframe processed roots according to a portable
    SplitManifest, enforcing complete case-level and keyframe-level split boundaries.
    """

    def __init__(
        self,
        scared_root: Path,
        split_manifest: SplitManifest | Path,
        split: str = "train",
        *,
        skip_every_map: Mapping[str, int] | None = None,
        phase: str = "all",
        test_every: int = 8,
        downsample: float = 1.0,
    ) -> None:
        self.scared_root = Path(scared_root)
        if isinstance(split_manifest, (str, Path)):
            self.manifest = load_scared_split_manifest(Path(split_manifest))
        elif isinstance(split_manifest, SplitManifest):
            validate_scared_split_manifest(split_manifest)
            self.manifest = split_manifest
        else:
            raise TypeError(
                f"split_manifest must be a SplitManifest or Path; got {type(split_manifest).__name__}"
            )

        split_norm = split.lower().strip()
        if split_norm in ("train", "training"):
            keyframe_entries = self.manifest.train
        elif split_norm in ("val", "validation"):
            keyframe_entries = self.manifest.validation
        elif split_norm in ("test", "testing"):
            keyframe_entries = self.manifest.test
        else:
            raise ValueError(
                f"unrecognized split {split!r}; expected 'train', 'validation', or 'test'"
            )

        self.split = split_norm
        self.keyframe_entries = tuple(keyframe_entries)
        skip_map = dict(skip_every_map) if skip_every_map is not None else {}

        self.keyframe_datasets: list[ScaredKeyframeDataset] = []
        self._index_map: list[tuple[int, int]] = []
        self._sample_ids: list[str] = []

        for kf_idx, entry in enumerate(self.keyframe_entries):
            dataset_id, keyframe_id = parse_scared_keyframe_id(entry)
            kf_dir = self.scared_root / entry
            kf_skip = skip_map.get(entry, skip_map.get(dataset_id, None))

            kf_ds = ScaredKeyframeDataset(
                keyframe_dir=kf_dir,
                dataset_id=dataset_id,
                keyframe_id=keyframe_id,
                skip_every=kf_skip,
                phase=phase,
                test_every=test_every,
                downsample=downsample,
            )
            self.keyframe_datasets.append(kf_ds)

            for local_idx, fid in enumerate(kf_ds.frame_ids):
                self._index_map.append((kf_idx, local_idx))
                self._sample_ids.append(make_scared_sample_id(dataset_id, keyframe_id, fid))

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, index: int) -> ScaredStage1Sample:
        if index < 0 or index >= len(self._index_map):
            raise IndexError(f"index {index} out of range for dataset with {len(self._index_map)} items")
        kf_idx, local_idx = self._index_map[index]
        return self.keyframe_datasets[kf_idx][local_idx]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Return all sample provenance identifiers in index order."""
        return tuple(self._sample_ids)

    def get_sample_provenance(self, index: int) -> tuple[str, str, str]:
        """Return (dataset_id, keyframe_id, frame_id) without decoding full tensors."""
        kf_idx, local_idx = self._index_map[index]
        ds = self.keyframe_datasets[kf_idx]
        return ds.dataset_id, ds.keyframe_id, ds.frame_ids[local_idx]


class ScaredUpstreamWrapper(Dataset):
    """Wrap an upstream SCARED_Dataset to strictly return ScaredStage1Sample."""

    def __init__(
        self,
        upstream_dataset: Any,
        dataset_id: str,
        keyframe_id: str,
    ) -> None:
        self.upstream = upstream_dataset
        self.dataset_id = dataset_id
        self.keyframe_id = keyframe_id

    def __len__(self) -> int:
        return len(self.upstream)

    def __getitem__(self, index: int) -> ScaredStage1Sample:
        raw = self.upstream[index]
        lmain = raw["lmain"]
        rmain = raw["rmain"]

        left = lmain["img"]
        right = rmain["img"]
        disp = lmain["disp"]
        if disp.ndim == 3 and disp.shape[0] == 1:
            disp = disp.squeeze(0)

        intr = lmain["intr"]
        extr = lmain["extr"]
        right_intr = rmain.get("intr", intr)

        # Reprojection matrix Q may be stored in raw or computed
        if "Q" in lmain:
            Q = lmain["Q"]
        else:
            # Construct OpenCV standard Q from disp_const, intr, baseline if available
            fl = intr[0, 0].item()
            cx = intr[0, 2].item()
            cy = intr[1, 2].item()
            disp_const = float(lmain.get("disp_const", 1.0))
            baseline = disp_const / fl if fl != 0 else 1.0
            Q = torch.tensor(
                [
                    [1.0, 0.0, 0.0, -cx],
                    [0.0, 1.0, 0.0, -cy],
                    [0.0, 0.0, 0.0, fl],
                    [0.0, 0.0, 1.0 / baseline, 0.0],
                ],
                dtype=torch.float32,
            )

        frame_id = str(raw.get("name", f"frame_{index:06d}"))
        sample = ScaredStage1Sample(
            dataset_id=self.dataset_id,
            keyframe_id=self.keyframe_id,
            frame_id=frame_id,
            left=left.float(),
            right=right.float(),
            disparity=disp.float(),
            intr=intr.float(),
            extr=extr.float(),
            Q=Q.float(),
            right_intr=right_intr.float(),
        )
        validate_stage1_sample(sample)
        return sample