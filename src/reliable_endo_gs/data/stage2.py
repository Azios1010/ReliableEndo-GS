"""Upstream-compatible Stage-2 data adapter and dataset wrappers.

Converts :class:`~reliable_endo_gs.data.scared_multi.ScaredStage1Sample` instances
into the dictionary layout expected by upstream Endo-E2E-GS Stage-2 models
and :class:`~reliable_endo_gs.baseline.native.NativeInferenceInput`, implementing exact
pinned upstream semantics:
- ``disp_const = Q[2, 3] / Q[3, 2]``
- Valid disparity: strictly finite and positive (``disp > 0``)
- Safe depth computation with ``znear=0.03``, ``zfar=300.0``, and upstream min-max normalization
- Camera center: ``C = -R.T @ t``
- World-view transform: extrinsic ``[R|t]`` homogeneous transpose (world-to-camera)
- Projection matrix: equivalent to ``lib.graphics_utils.getProjectionMatrix(...).t()``
- Full provenance preservation across multi-sequence datasets without hardcoding dataset_3.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from reliable_endo_gs.data.scared_manifest import (
    SplitManifest,
    make_scared_sample_id,
)
from reliable_endo_gs.data.scared_multi import (
    ScaredMultiSequenceDataset,
    ScaredStage1Sample,
)

DEFAULT_ZNEAR: float = 0.03
DEFAULT_ZFAR: float = 300.0


def compute_disp_const(Q: torch.Tensor) -> float:
    """Compute disparity constant from reprojection matrix Q.

    Formula: disp_const = Q[2, 3] / Q[3, 2] = fl * baseline.
    """
    if not isinstance(Q, torch.Tensor):
        raise TypeError(f"Q must be a torch.Tensor, got {type(Q).__name__}")
    if Q.shape != (4, 4):
        raise ValueError(f"Q matrix must have shape [4, 4]; got {tuple(Q.shape)}")
    q32 = Q[3, 2]
    if abs(q32.item()) < 1e-8:
        raise ValueError(f"Q[3, 2] is zero or near-zero ({q32.item()}), cannot compute disp_const")
    return float((Q[2, 3] / q32).item())


def compute_fov(focal: float, pixels: int | float) -> float:
    """Compute field of view in radians from focal length and pixel dimension.

    Formula: 2 * arctan(pixels / (2 * focal)), identical to upstream focal2fov.
    """
    if focal <= 0:
        raise ValueError(f"focal length must be positive; got {focal}")
    if pixels <= 0:
        raise ValueError(f"pixels must be positive; got {pixels}")
    return float(2.0 * math.atan(float(pixels) / (2.0 * float(focal))))


def compute_world_view_transform(extr: torch.Tensor) -> torch.Tensor:
    """Compute world_view_transform as homogeneous transpose of world-to-camera extrinsic.

    Given extr [3, 4] = [R | t]:
    M = [[R, t], [0, 1]] (4x4)
    world_view_transform = M.T
    """
    if not isinstance(extr, torch.Tensor):
        raise TypeError(f"extr must be a torch.Tensor, got {type(extr).__name__}")
    if extr.shape != (3, 4):
        raise ValueError(f"extr must have shape [3, 4]; got {tuple(extr.shape)}")
    w2c = torch.eye(4, dtype=extr.dtype, device=extr.device)
    w2c[:3, :3] = extr[:3, :3]
    w2c[:3, 3] = extr[:3, 3]
    return w2c.t().contiguous()


def compute_camera_center(extr: torch.Tensor) -> torch.Tensor:
    """Compute camera center in world coordinates.

    Formula: C = -R.T @ t, where extr = [R | t].
    """
    if not isinstance(extr, torch.Tensor):
        raise TypeError(f"extr must be a torch.Tensor, got {type(extr).__name__}")
    if extr.shape != (3, 4):
        raise ValueError(f"extr must have shape [3, 4]; got {tuple(extr.shape)}")
    R = extr[:3, :3]
    t = extr[:3, 3]
    return (-R.t() @ t).contiguous()


def compute_projection_matrix(
    intr: torch.Tensor,
    height: int,
    width: int,
    *,
    znear: float = DEFAULT_ZNEAR,
    zfar: float = DEFAULT_ZFAR,
) -> torch.Tensor:
    """Compute projection matrix transposed, matching lib.graphics_utils.getProjectionMatrix.

    Returns:
        projection_matrix: [4, 4] float32 tensor (transposed).
    """
    if not isinstance(intr, torch.Tensor):
        raise TypeError(f"intr must be a torch.Tensor, got {type(intr).__name__}")
    if intr.shape != (3, 3):
        raise ValueError(f"intr must have shape [3, 3]; got {tuple(intr.shape)}")
    if height <= 0 or width <= 0:
        raise ValueError(f"spatial dimensions must be positive; got height={height}, width={width}")
    if znear <= 0 or zfar <= znear:
        raise ValueError(f"invalid clip planes: znear={znear}, zfar={zfar}")

    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]

    near_fx = znear / fx
    near_fy = znear / fy
    left = -(float(width) - cx) * near_fx
    right = cx * near_fx
    bottom = (cy - float(height)) * near_fy
    top = cy * near_fy

    P = torch.zeros((4, 4), dtype=intr.dtype, device=intr.device)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P.t().contiguous()


def compute_safe_depth(
    disparity: torch.Tensor,
    disp_const: float,
    *,
    znear: float = DEFAULT_ZNEAR,
    zfar: float = DEFAULT_ZFAR,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Safely compute disparity, mask, and upstream normalized GT-depth.

    Guarantees:
    - Valid disparity: strictly finite and positive (disp > 0).
    - Zero/invalid/non-finite disparity is masked out (mask=0.0, disp=0.0).
    - Depth outside [znear, zfar] is zeroed out.
    - Normalized GT-depth matches upstream: (depth - min) / (max - min),
      with safe handling of constant/empty depth to guarantee no NaN or Inf.

    Returns:
        disp: [1, H, W] float32 tensor
        mask: [1, H, W] float32 tensor
        depth: [1, H, W] float32 tensor
    """
    if not isinstance(disparity, torch.Tensor):
        raise TypeError(f"disparity must be a torch.Tensor, got {type(disparity).__name__}")
    disp_2d = disparity.squeeze(0) if disparity.ndim == 3 and disparity.shape[0] == 1 else disparity
    if disp_2d.ndim != 2:
        raise ValueError(f"disparity must have shape [H, W] or [1, H, W]; got {tuple(disparity.shape)}")

    valid_mask = torch.isfinite(disp_2d) & (disp_2d > 0.0)
    clean_disp = torch.where(valid_mask, disp_2d, torch.zeros_like(disp_2d))
    mask = valid_mask.to(dtype=disp_2d.dtype)

    depth = torch.zeros_like(disp_2d)
    if valid_mask.any():
        depth[valid_mask] = float(disp_const) / disp_2d[valid_mask]
        depth[depth > zfar] = 0.0
        depth[depth < znear] = 0.0

        d_min = depth.min()
        d_max = depth.max()
        d_range = d_max - d_min
        if d_range > 0.0:
            depth = (depth - d_min) / d_range
        else:
            depth = torch.zeros_like(depth)

    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    return clean_disp.unsqueeze(0), mask.unsqueeze(0), depth.unsqueeze(0)


class ScaredStage2Sample(dict):
    """Stage-2 sample container matching upstream Endo-E2E-GS dictionary schema.

    Inherits from dict so that PyTorch ``DataLoader`` batch collation works transparently.
    Exposes all required upstream keys ('name', 'lmain', 'rmain') while preserving
    sample provenance identifiers ('dataset_id', 'keyframe_id', 'frame_id', 'sample_id').
    """

    def __init__(
        self,
        *,
        name: str,
        lmain: dict[str, Any],
        rmain: dict[str, Any],
        dataset_id: str | None = None,
        keyframe_id: str | None = None,
        frame_id: str | None = None,
        sample_id: str | None = None,
        **extra: Any,
    ) -> None:
        sid = sample_id or name
        payload: dict[str, Any] = {
            "name": name,
            "lmain": lmain,
            "rmain": rmain,
            "sample_id": sid,
            "dataset_id": dataset_id or "",
            "keyframe_id": keyframe_id or "",
            "frame_id": frame_id or "",
        }
        payload.update(extra)
        super().__init__(payload)

    @property
    def name(self) -> str:
        return self["name"]

    @property
    def lmain(self) -> dict[str, Any]:
        return self["lmain"]

    @property
    def rmain(self) -> dict[str, Any]:
        return self["rmain"]

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
        return self["lmain"]["img"]

    @property
    def right(self) -> torch.Tensor:
        return self["rmain"]["img"]

    @property
    def mask(self) -> torch.Tensor:
        return self["lmain"]["mask"]

    @property
    def disparity(self) -> torch.Tensor:
        return self["lmain"]["disp"]

    @property
    def depth(self) -> torch.Tensor:
        return self["lmain"]["depth"]

    @property
    def world_view_transform(self) -> torch.Tensor:
        return self["lmain"]["world_view_transform"]

    @property
    def full_proj_transform(self) -> torch.Tensor:
        return self["lmain"]["full_proj_transform"]

    @property
    def camera_center(self) -> torch.Tensor:
        return self["lmain"]["camera_center"]

    def to_native_dict(self) -> dict[str, Any]:
        """Return the exact minimal dictionary expected by upstream models."""
        return {
            "name": self.name,
            "lmain": dict(self.lmain),
            "rmain": dict(self.rmain),
        }

    def to_native_inference_input(self) -> Any:
        """Convert single sample to NativeInferenceInput (batch_size=1)."""
        from reliable_endo_gs.baseline.native import NativeInferenceInput

        lmain = self.lmain
        rmain = self.rmain
        return NativeInferenceInput(
            sample_names=(self.name,),
            left_image=lmain["img"].unsqueeze(0),
            right_image=rmain["img"].unsqueeze(0),
            left_mask=lmain["mask"].unsqueeze(0),
            left_disparity_constant=torch.tensor([lmain["disp_const"]], dtype=torch.float32),
            left_intrinsics=lmain["intr"].unsqueeze(0),
            right_intrinsics=rmain["intr"].unsqueeze(0),
            left_extrinsics=lmain["extr"].unsqueeze(0),
            left_fov_x=torch.tensor([lmain["FovX"]], dtype=torch.float32),
            left_fov_y=torch.tensor([lmain["FovY"]], dtype=torch.float32),
            left_width=torch.tensor([lmain["width"]], dtype=torch.int64),
            left_height=torch.tensor([lmain["height"]], dtype=torch.int64),
            left_world_view_transform=lmain["world_view_transform"].unsqueeze(0),
            left_full_projection_transform=lmain["full_proj_transform"].unsqueeze(0),
            left_camera_center=lmain["camera_center"].unsqueeze(0),
        )


def stage1_to_stage2_sample(
    sample: Mapping[str, Any],
    *,
    znear: float = DEFAULT_ZNEAR,
    zfar: float = DEFAULT_ZFAR,
) -> ScaredStage2Sample:
    """Transform a ScaredStage1Sample into an upstream-compatible ScaredStage2Sample.

    Enforces exact pinned Endo-E2E-GS semantics:
    - disp_const = Q[2, 3] / Q[3, 2]
    - valid disparity: strictly finite and positive
    - safe depth in [znear, zfar], normalized to [0, 1]
    - world_view_transform = M.T where M = [R|t; 0 1]
    - camera_center = -R.T @ t
    - projection_matrix equivalent to getProjectionMatrix.t()
    - full_proj_transform = world_view_transform @ projection_matrix
    """
    left = sample["left"]
    right = sample["right"]
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise TypeError("left and right must be torch.Tensor instances")
    if left.ndim != 3 or left.shape[0] != 3:
        raise ValueError(f"left must have shape [3, H, W]; got {tuple(left.shape)}")
    if right.shape != left.shape:
        raise ValueError(f"right shape {tuple(right.shape)} must match left shape {tuple(left.shape)}")

    _, height, width = left.shape

    intr = sample["intr"]
    right_intr = sample.get("right_intr")
    if right_intr is None:
        right_intr = intr.clone()

    extr = sample["extr"]
    Q = sample["Q"]

    disp_key = "disparity" if "disparity" in sample else "disp"
    if disp_key not in sample:
        raise ValueError("sample missing 'disparity' or 'disp' tensor")
    raw_disp = sample[disp_key]

    # Compute disparity constant from reprojection matrix Q
    disp_const = compute_disp_const(Q)

    # Safe disparity, mask, and depth computation
    disp, mask, depth = compute_safe_depth(raw_disp, disp_const, znear=znear, zfar=zfar)

    # Camera transforms
    world_view_transform = compute_world_view_transform(extr)
    camera_center = compute_camera_center(extr)
    projection_matrix = compute_projection_matrix(intr, height, width, znear=znear, zfar=zfar)
    full_proj_transform = (world_view_transform @ projection_matrix).contiguous()

    # Field of view
    fx = float(intr[0, 0].item())
    fy = float(intr[1, 1].item())
    fov_x = compute_fov(fx, width)
    fov_y = compute_fov(fy, height)

    # Provenance
    dataset_id = sample.get("dataset_id", "")
    keyframe_id = sample.get("keyframe_id", "")
    frame_id = sample.get("frame_id", "")
    sample_id = sample.get("sample_id") or (
        make_scared_sample_id(dataset_id, keyframe_id, frame_id)
        if dataset_id and keyframe_id and frame_id
        else sample.get("name", "sample")
    )
    name = str(sample.get("name", sample_id))

    lmain: dict[str, Any] = {
        "img": left.float(),
        "mask": mask.float(),
        "disp": disp.float(),
        "depth": depth.float(),
        "disp_const": disp_const,
        "intr": intr.float(),
        "extr": extr.float(),
        "FovX": fov_x,
        "FovY": fov_y,
        "width": width,
        "height": height,
        "world_view_transform": world_view_transform.float(),
        "full_proj_transform": full_proj_transform.float(),
        "camera_center": camera_center.float(),
    }
    rmain: dict[str, Any] = {
        "img": right.float(),
        "intr": right_intr.float(),
    }

    out_sample = ScaredStage2Sample(
        name=name,
        lmain=lmain,
        rmain=rmain,
        dataset_id=dataset_id,
        keyframe_id=keyframe_id,
        frame_id=frame_id,
        sample_id=sample_id,
    )
    validate_stage2_sample(out_sample)
    return out_sample


def validate_stage2_sample(sample: Mapping[str, Any]) -> None:
    """Strict validation for Stage-2 upstream-compatible sample contracts."""
    if "name" not in sample:
        raise ValueError("sample missing 'name' field")
    if not isinstance(sample["name"], str) or not sample["name"].strip():
        raise ValueError(f"'name' must be a non-empty string; got {sample.get('name')!r}")

    if "lmain" not in sample or not isinstance(sample["lmain"], Mapping):
        raise ValueError("sample missing 'lmain' mapping")
    if "rmain" not in sample or not isinstance(sample["rmain"], Mapping):
        raise ValueError("sample missing 'rmain' mapping")

    lmain = sample["lmain"]
    rmain = sample["rmain"]

    required_lmain_keys = (
        "img",
        "mask",
        "disp",
        "depth",
        "disp_const",
        "intr",
        "extr",
        "FovX",
        "FovY",
        "width",
        "height",
        "world_view_transform",
        "full_proj_transform",
        "camera_center",
    )
    for key in required_lmain_keys:
        if key not in lmain:
            raise ValueError(f"lmain missing required field {key!r}")

    img = lmain["img"]
    if not isinstance(img, torch.Tensor) or img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"lmain['img'] must have shape [3, H, W]; got {getattr(img, 'shape', type(img))}")
    if img.dtype != torch.float32:
        raise TypeError(f"lmain['img'] must have dtype float32; got {img.dtype}")
    if not torch.isfinite(img).all():
        raise ValueError("lmain['img'] contains non-finite values")

    _, h, w = img.shape
    width = lmain["width"]
    height = lmain["height"]
    if width != w or height != h:
        raise ValueError(f"lmain dimensions (w={width}, h={height}) do not match img ({w}, {h})")

    mask = lmain["mask"]
    if not isinstance(mask, torch.Tensor) or mask.shape != (1, h, w):
        raise ValueError(f"lmain['mask'] must have shape [1, {h}, {w}]; got {getattr(mask, 'shape', type(mask))}")
    if not torch.isfinite(mask).all():
        raise ValueError("lmain['mask'] contains non-finite values")

    disp = lmain["disp"]
    if not isinstance(disp, torch.Tensor) or disp.shape != (1, h, w):
        raise ValueError(f"lmain['disp'] must have shape [1, {h}, {w}]; got {getattr(disp, 'shape', type(disp))}")
    if not torch.isfinite(disp).all():
        raise ValueError("lmain['disp'] contains non-finite values")
    if (disp < 0).any():
        raise ValueError("lmain['disp'] contains negative values")

    depth = lmain["depth"]
    if not isinstance(depth, torch.Tensor) or depth.shape != (1, h, w):
        raise ValueError(f"lmain['depth'] must have shape [1, {h}, {w}]; got {getattr(depth, 'shape', type(depth))}")
    if not torch.isfinite(depth).all():
        raise ValueError("lmain['depth'] contains non-finite values")

    disp_const = lmain["disp_const"]
    if not isinstance(disp_const, (float, int)) or not math.isfinite(disp_const) or disp_const <= 0:
        raise ValueError(f"lmain['disp_const'] must be a positive finite float; got {disp_const}")

    intr = lmain["intr"]
    if not isinstance(intr, torch.Tensor) or intr.shape != (3, 3):
        raise ValueError(f"lmain['intr'] must have shape [3, 3]; got {getattr(intr, 'shape', type(intr))}")
    if not torch.isfinite(intr).all():
        raise ValueError("lmain['intr'] contains non-finite values")

    extr = lmain["extr"]
    if not isinstance(extr, torch.Tensor) or extr.shape != (3, 4):
        raise ValueError(f"lmain['extr'] must have shape [3, 4]; got {getattr(extr, 'shape', type(extr))}")
    if not torch.isfinite(extr).all():
        raise ValueError("lmain['extr'] contains non-finite values")

    w2v = lmain["world_view_transform"]
    if not isinstance(w2v, torch.Tensor) or w2v.shape != (4, 4):
        raise ValueError(f"lmain['world_view_transform'] must have shape [4, 4]; got {getattr(w2v, 'shape', type(w2v))}")
    if not torch.isfinite(w2v).all():
        raise ValueError("lmain['world_view_transform'] contains non-finite values")

    fproj = lmain["full_proj_transform"]
    if not isinstance(fproj, torch.Tensor) or fproj.shape != (4, 4):
        raise ValueError(f"lmain['full_proj_transform'] must have shape [4, 4]; got {getattr(fproj, 'shape', type(fproj))}")
    if not torch.isfinite(fproj).all():
        raise ValueError("lmain['full_proj_transform'] contains non-finite values")

    center = lmain["camera_center"]
    if not isinstance(center, torch.Tensor) or center.shape != (3,):
        raise ValueError(f"lmain['camera_center'] must have shape [3]; got {getattr(center, 'shape', type(center))}")
    if not torch.isfinite(center).all():
        raise ValueError("lmain['camera_center'] contains non-finite values")

    r_img = rmain.get("img")
    if not isinstance(r_img, torch.Tensor) or r_img.shape != (3, h, w):
        raise ValueError(f"rmain['img'] must have shape [3, {h}, {w}]; got {getattr(r_img, 'shape', type(r_img))}")
    if not torch.isfinite(r_img).all():
        raise ValueError("rmain['img'] contains non-finite values")

    r_intr = rmain.get("intr")
    if not isinstance(r_intr, torch.Tensor) or r_intr.shape != (3, 3):
        raise ValueError(f"rmain['intr'] must have shape [3, 3]; got {getattr(r_intr, 'shape', type(r_intr))}")
    if not torch.isfinite(r_intr).all():
        raise ValueError("rmain['intr'] contains non-finite values")


class ScaredStage2Dataset(Dataset):
    """Dataset wrapper producing upstream-compatible Stage-2 samples.

    Wraps a Stage-1 dataset (such as :class:`ScaredMultiSequenceDataset`) or
    instantiates one from root and manifest paths, transforming each sample into
    an upstream-compatible Stage-2 dictionary with full provenance preservation and
    exact pinned Endo-E2E-GS semantics.
    """

    def __init__(
        self,
        stage1_dataset: Dataset | None = None,
        *,
        scared_root: Path | str | None = None,
        split_manifest: SplitManifest | Path | str | None = None,
        split: str = "train",
        skip_every_map: Mapping[str, int] | None = None,
        phase: str = "all",
        test_every: int = 8,
        downsample: float = 1.0,
        znear: float = DEFAULT_ZNEAR,
        zfar: float = DEFAULT_ZFAR,
    ) -> None:
        self.znear = float(znear)
        self.zfar = float(zfar)

        if stage1_dataset is not None:
            self.stage1_dataset = stage1_dataset
        elif scared_root is not None:
            if split_manifest is None:
                raise ValueError("split_manifest must be provided when constructing from scared_root")
            self.stage1_dataset = ScaredMultiSequenceDataset(
                scared_root=Path(scared_root),
                split_manifest=split_manifest,
                split=split,
                skip_every_map=skip_every_map,
                phase=phase,
                test_every=test_every,
                downsample=downsample,
            )
        else:
            raise ValueError("either stage1_dataset or scared_root must be provided")

    def __len__(self) -> int:
        return len(self.stage1_dataset)

    def __getitem__(self, index: int) -> ScaredStage2Sample:
        sample = self.stage1_dataset[index]
        return stage1_to_stage2_sample(sample, znear=self.znear, zfar=self.zfar)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Return all sample provenance identifiers in index order."""
        if hasattr(self.stage1_dataset, "sample_ids"):
            return tuple(self.stage1_dataset.sample_ids)
        return tuple(self[i].sample_id for i in range(len(self)))

    def get_sample_provenance(self, index: int) -> tuple[str, str, str]:
        """Return (dataset_id, keyframe_id, frame_id) without full decode if underlying supports it."""
        if hasattr(self.stage1_dataset, "get_sample_provenance"):
            return self.stage1_dataset.get_sample_provenance(index)
        sample = self[index]
        return sample.dataset_id, sample.keyframe_id, sample.frame_id

    @property
    def split(self) -> str | None:
        return getattr(self.stage1_dataset, "split", None)

    @property
    def keyframe_entries(self) -> tuple[str, ...]:
        return tuple(getattr(self.stage1_dataset, "keyframe_entries", ()))


def stage2_collate_fn(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate a sequence of Stage-2 sample dicts into a batched upstream dictionary."""
    if not batch:
        raise ValueError("cannot collate empty batch")

    names = [str(item["name"]) for item in batch]
    sample_ids = [str(item.get("sample_id", item["name"])) for item in batch]
    dataset_ids = [str(item.get("dataset_id", "")) for item in batch]
    keyframe_ids = [str(item.get("keyframe_id", "")) for item in batch]
    frame_ids = [str(item.get("frame_id", "")) for item in batch]

    left_imgs = torch.stack([item["lmain"]["img"] for item in batch], dim=0)
    right_imgs = torch.stack([item["rmain"]["img"] for item in batch], dim=0)
    masks = torch.stack([item["lmain"]["mask"] for item in batch], dim=0)
    disps = torch.stack([item["lmain"]["disp"] for item in batch], dim=0)
    depths = torch.stack([item["lmain"]["depth"] for item in batch], dim=0)
    intrs = torch.stack([item["lmain"]["intr"] for item in batch], dim=0)
    right_intrs = torch.stack([item["rmain"]["intr"] for item in batch], dim=0)
    extrs = torch.stack([item["lmain"]["extr"] for item in batch], dim=0)
    w2vs = torch.stack([item["lmain"]["world_view_transform"] for item in batch], dim=0)
    fprojs = torch.stack([item["lmain"]["full_proj_transform"] for item in batch], dim=0)
    centers = torch.stack([item["lmain"]["camera_center"] for item in batch], dim=0)

    disp_consts = torch.tensor([float(item["lmain"]["disp_const"]) for item in batch], dtype=torch.float32)
    fov_xs = torch.tensor([float(item["lmain"]["FovX"]) for item in batch], dtype=torch.float32)
    fov_ys = torch.tensor([float(item["lmain"]["FovY"]) for item in batch], dtype=torch.float32)
    widths = torch.tensor([int(item["lmain"]["width"]) for item in batch], dtype=torch.int64)
    heights = torch.tensor([int(item["lmain"]["height"]) for item in batch], dtype=torch.int64)

    return {
        "name": names,
        "sample_id": sample_ids,
        "dataset_id": dataset_ids,
        "keyframe_id": keyframe_ids,
        "frame_id": frame_ids,
        "lmain": {
            "img": left_imgs,
            "mask": masks,
            "disp": disps,
            "depth": depths,
            "disp_const": disp_consts,
            "intr": intrs,
            "extr": extrs,
            "FovX": fov_xs,
            "FovY": fov_ys,
            "width": widths,
            "height": heights,
            "world_view_transform": w2vs,
            "full_proj_transform": fprojs,
            "camera_center": centers,
        },
        "rmain": {
            "img": right_imgs,
            "intr": right_intrs,
        },
    }


def stage2_batch_to_native_input(batch: Mapping[str, Any]) -> Any:
    """Convert a collated Stage-2 batch dictionary into NativeInferenceInput."""
    from reliable_endo_gs.baseline.native import NativeInferenceInput

    names = batch["name"]
    if isinstance(names, str):
        names = [names]
    lmain = batch["lmain"]
    rmain = batch["rmain"]

    return NativeInferenceInput(
        sample_names=tuple(names),
        left_image=lmain["img"],
        right_image=rmain["img"],
        left_mask=lmain["mask"],
        left_disparity_constant=lmain["disp_const"],
        left_intrinsics=lmain["intr"],
        right_intrinsics=rmain["intr"],
        left_extrinsics=lmain["extr"],
        left_fov_x=lmain["FovX"],
        left_fov_y=lmain["FovY"],
        left_width=lmain["width"],
        left_height=lmain["height"],
        left_world_view_transform=lmain["world_view_transform"],
        left_full_projection_transform=lmain["full_proj_transform"],
        left_camera_center=lmain["camera_center"],
    )
