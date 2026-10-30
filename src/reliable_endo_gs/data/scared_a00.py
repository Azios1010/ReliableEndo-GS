"""Strict A00 deployment access to the processed SCARED layout.

The historical :mod:`scared_multi` loader intentionally exposes upstream-shaped
``[-1, 1]`` images and supervision fields.  This module is the independent
deployment boundary used by A00/A01/A04.  It reads the same processed folders,
but emits only ``DeploymentStereoBatch`` for inference and keeps disparity in a
separate supervision object.

The processed SCARED files do not, by themselves, prove the units or direction
of the stereo extrinsic.  A00 therefore requires ``a00_protocol.json`` next to
the keyframe ``data`` directory (or an equivalent ``a00_protocol`` object in
the per-frame JSON).  The required fields are documented in
``docs/a00_scared_protocol.md`` and are validated before any tensor is used.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera
from reliable_endo_gs.data.scared_manifest import make_scared_sample_id
from reliable_endo_gs.geometry.rectified import crop_resize_camera
from reliable_endo_gs.teachers.roles import (
    DEFAULT_A01_SCARED_ROLES,
    A01ScaredDataRoles,
)
from reliable_endo_gs.utils.io import write_json_atomic


class ScaredA00ProtocolError(ValueError):
    """Raised when SCARED metadata cannot prove the A00 camera contract."""


class ScaredA00SampleError(ValueError):
    """Raised when one active SCARED sample is missing or malformed."""


_PROTOCOL_FILENAME = "a00_protocol.json"
A00_SUPERVISION_RESAMPLING = "nearest_exact_half_pixel_centres"
_REQUIRED_PROTOCOL_FIELDS = (
    "schema_version",
    "image_range",
    "rectified",
    "distortion_removed",
    "pixel_center",
    "disparity",
    "extrinsics",
    "preprocessing",
)
_ACTIVE_DATASET_PREFIX = "dataset_"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ScaredA00ProtocolError(f"{name} must be a string-keyed mapping")
    return dict(value)


def _positive_pair(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ScaredA00ProtocolError(f"{name} must contain two non-negative integers")
    result = tuple(int(item) for item in value)
    if any(isinstance(item, bool) or item < 0 for item in value):
        raise ScaredA00ProtocolError(f"{name} must contain two non-negative integers")
    return result  # type: ignore[return-value]


def _matrix(value: object, name: str, shape: tuple[int, int]) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError) as error:
        raise ScaredA00ProtocolError(f"{name} must be a numeric matrix") from error
    if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
        raise ScaredA00ProtocolError(f"{name} must be finite with shape {shape}")
    return tensor


def _homogeneous(value: object, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tuple(tensor.shape) == (4, 4):
        result = tensor
    elif tuple(tensor.shape) == (3, 4):
        result = torch.eye(4, dtype=torch.float64)
        result[:3] = tensor
    else:
        raise ScaredA00ProtocolError(f"{name} must have shape [4,4] or [3,4]")
    if not bool(torch.isfinite(result).all()):
        raise ScaredA00ProtocolError(f"{name} must be finite")
    last = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    if not torch.allclose(result[3], last, atol=1e-8, rtol=1e-8):
        raise ScaredA00ProtocolError(f"{name} must be homogeneous with final row [0,0,0,1]")
    return result


def _translation_scale(unit: object) -> float:
    normalized = str(unit).strip().casefold()
    scales = {
        "m": 1.0,
        "meter": 1.0,
        "metre": 1.0,
        "mm": 1.0e-3,
        "millimeter": 1.0e-3,
        "millimetre": 1.0e-3,
        "cm": 1.0e-2,
    }
    if normalized not in scales:
        raise ScaredA00ProtocolError(
            "extrinsics.units must explicitly be one of m, cm, or mm; no unit inference is allowed"
        )
    return scales[normalized]


def _safe_relative_id(sample_id: str) -> tuple[str, str, str]:
    cleaned = sample_id.strip().replace("\\", "/")
    parts = cleaned.split("/")
    if len(parts) != 3 or any(not item or item in (".", "..") for item in parts):
        raise ScaredA00SampleError(
            "sample_id must be dataset_N/keyframe_N/frame_id without traversal segments"
        )
    dataset_id, keyframe_id, frame_id = parts
    if dataset_id not in DEFAULT_A01_SCARED_ROLES.dataset_ids:
        raise ScaredA00SampleError(
            "active A01 SCARED path accepts only original SCARED dataset_1 through dataset_9; "
            "SCARED-C and dataset_10+ are rejected"
        )
    if not keyframe_id.startswith("keyframe_") or not frame_id:
        raise ScaredA00SampleError("sample_id has an invalid keyframe or frame component")
    return dataset_id, keyframe_id, frame_id


@dataclass(frozen=True, slots=True)
class A00Protocol:
    """Validated camera/image semantics for one processed SCARED keyframe."""

    schema_version: int
    image_range: str
    rectified: bool
    distortion_removed: bool
    pixel_center: str
    disparity_units: str
    disparity_sign: str
    disparity_reference: str
    extrinsic_source: str
    extrinsic_direction: str
    extrinsic_units: str
    preprocessing: Mapping[str, object]
    raw: Mapping[str, object]

    @property
    def identity_hash(self) -> str:
        return _hash_json(self.raw)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], *, source: str) -> A00Protocol:
        raw = _mapping(value, f"A00 protocol from {source}")
        missing = [key for key in _REQUIRED_PROTOCOL_FIELDS if key not in raw]
        if missing:
            raise ScaredA00ProtocolError(
                f"{source} is missing required A00 protocol field(s): {', '.join(missing)}; "
                f"provide {_PROTOCOL_FILENAME} with image_range, rectification, pixel_center, "
                "disparity semantics, extrinsics source/direction/units and preprocessing metadata"
            )
        if raw["schema_version"] != 1:
            raise ScaredA00ProtocolError("A00 protocol schema_version must be 1")
        if raw["image_range"] not in ("uint8_0_255", "float01"):
            raise ScaredA00ProtocolError("A00 protocol image_range must be uint8_0_255 or float01")
        if raw["rectified"] is not True or raw["distortion_removed"] is not True:
            raise ScaredA00ProtocolError(
                "A00 requires rectified=true and distortion_removed=true; raw distorted images are unsupported"
            )
        if raw["pixel_center"] != "integer":
            raise ScaredA00ProtocolError("A00 requires pixel_center=integer")
        disparity = _mapping(raw["disparity"], "A00 protocol disparity")
        units = disparity.get("units")
        sign = disparity.get("sign")
        reference = disparity.get("reference")
        if units not in ("pixels", "pixel") or sign != "u_left_minus_right" or reference != "left":
            raise ScaredA00ProtocolError(
                "A00 disparity must declare units=pixels, sign=u_left_minus_right, reference=left"
            )
        extrinsics = _mapping(raw["extrinsics"], "A00 protocol extrinsics")
        source_name = extrinsics.get("source")
        direction = extrinsics.get("direction")
        units_name = extrinsics.get("units")
        if not isinstance(source_name, str) or not source_name.strip():
            raise ScaredA00ProtocolError(
                "A00 extrinsics.source must identify a per-frame field (for example newpram.T_R_from_L)"
            )
        if direction not in ("right_from_left", "left_from_right"):
            raise ScaredA00ProtocolError(
                "A00 extrinsics.direction must be right_from_left or left_from_right"
            )
        _translation_scale(units_name)
        preprocessing = _mapping(raw["preprocessing"], "A00 protocol preprocessing")
        if preprocessing.get("images_grid", "processed") != "processed":
            raise ScaredA00ProtocolError(
                "A00 preprocessing.images_grid must be processed; transformations must be recorded explicitly"
            )
        return cls(
            schema_version=1,
            image_range=str(raw["image_range"]),
            rectified=True,
            distortion_removed=True,
            pixel_center="integer",
            disparity_units="pixels",
            disparity_sign="u_left_minus_right",
            disparity_reference="left",
            extrinsic_source=str(source_name),
            extrinsic_direction=str(direction),
            extrinsic_units=str(units_name),
            preprocessing=preprocessing,
            raw=raw,
        )


@dataclass(frozen=True, slots=True)
class ScaredSupervision:
    """Training-only GT; never placed inside :class:`DeploymentStereoBatch`."""

    disparity: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.disparity.ndim != 4 or self.disparity.shape[1] != 1:
            raise ValueError("supervision disparity must have shape [B,1,H,W]")
        if self.valid_mask.shape != self.disparity.shape or self.valid_mask.dtype != torch.bool:
            raise ValueError("supervision valid_mask must be bool with disparity shape")


@dataclass(frozen=True, slots=True)
class ScaredA00Sample:
    """One A00 deployment batch plus optional training-only supervision."""

    sample_id: str
    data_role: str
    batch: DeploymentStereoBatch
    calibration_metadata: Mapping[str, object]
    supervision: ScaredSupervision | None = None


@dataclass(frozen=True, slots=True)
class A01ManifestEntry:
    sample_id: str
    data_role: str

    def __post_init__(self) -> None:
        _safe_relative_id(self.sample_id)
        if self.data_role not in ("final_train", "final_test"):
            raise ScaredA00SampleError(f"unsupported active A01 role {self.data_role!r}")

    def to_dict(self) -> dict[str, str]:
        return {"sample_id": self.sample_id, "data_role": self.data_role}


@dataclass(frozen=True, slots=True)
class A01Manifest:
    """Explicit sample index for the active nine-fold A01 role policy."""

    role_contract_hash: str
    entries: tuple[A01ManifestEntry, ...]
    schema_version: int = 1
    dataset_name: str = "scared"

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.dataset_name != "scared":
            raise ScaredA00SampleError("active A01 manifest must be schema 1 for dataset scared")
        if not self.entries:
            raise ScaredA00SampleError("active A01 manifest must contain at least one sample")
        if len({entry.sample_id for entry in self.entries}) != len(self.entries):
            raise ScaredA00SampleError("active A01 manifest contains duplicate sample IDs")
        if len(self.role_contract_hash) != 64:
            raise ScaredA00SampleError("active A01 manifest requires a role-contract SHA-256")

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(entry.sample_id for entry in self.entries)

    @property
    def manifest_hash(self) -> str:
        """Return the immutable sample/role index identity."""

        return _hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "role_contract_hash": self.role_contract_hash,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _protocol_path(keyframe_dir: Path) -> Path:
    data_dir = keyframe_dir / "data" if (keyframe_dir / "data").is_dir() else keyframe_dir
    return data_dir / _PROTOCOL_FILENAME


def _load_json(path: Path, name: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScaredA00SampleError(f"unable to read {name} at {path}: {error}") from error
    return _mapping(value, name)


def _load_protocol(
    keyframe_dir: Path, frame_data: Mapping[str, object], newpram: Mapping[str, object]
) -> A00Protocol:
    path = _protocol_path(keyframe_dir)
    if path.is_file():
        return A00Protocol.from_mapping(_load_json(path, "A00 protocol"), source=str(path))
    for parent_name, parent in (("frame_data", frame_data), ("newpram_data", newpram)):
        embedded = parent.get("a00_protocol")
        if isinstance(embedded, Mapping):
            return A00Protocol.from_mapping(embedded, source=f"{parent_name}.a00_protocol")
    raise ScaredA00ProtocolError(
        f"missing {path.name} for {keyframe_dir}; exact required metadata is schema_version=1, "
        "image_range, rectified=true, distortion_removed=true, pixel_center=integer, "
        "disparity units/sign/reference, extrinsics source/direction/units, and "
        "preprocessing.images_grid=processed"
    )


def _lookup_nested(mapping: Mapping[str, object], dotted: str) -> object:
    current: object = mapping
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ScaredA00ProtocolError(f"declared extrinsics source {dotted!r} is absent")
        current = current[part]
    return current


def _extract_relative_transform(
    protocol: A00Protocol, frame_data: Mapping[str, object], newpram: Mapping[str, object]
) -> torch.Tensor:
    source = protocol.extrinsic_source
    if source.startswith("frame_data."):
        value = _lookup_nested(frame_data, source[len("frame_data.") :])
    elif source.startswith("newpram."):
        value = _lookup_nested(newpram, source[len("newpram.") :])
    elif source in frame_data:
        value = frame_data[source]
    elif source in newpram:
        value = newpram[source]
    else:
        raise ScaredA00ProtocolError(f"declared extrinsics source {source!r} is absent")
    try:
        transform = _homogeneous(value, f"extrinsics {source}")
    except ScaredA00ProtocolError as error:
        # Some processed SCARED metadata stores ``T`` as a 3x1 vector and
        # ``R`` beside it.  Accept that representation only when the protocol
        # names the companion rotation field explicitly.
        if torch.as_tensor(value).shape not in ((3,), (3, 1)):
            raise
        rotation_source = protocol.raw.get("extrinsics", {})
        if not isinstance(rotation_source, Mapping) or not isinstance(
            rotation_source.get("rotation_source"), str
        ):
            raise ScaredA00ProtocolError(
                f"extrinsics {source} is a translation vector; extrinsics.rotation_source must name its 3x3 rotation"
            ) from error
        rotation_name = str(rotation_source["rotation_source"])
        if rotation_name.startswith("frame_data."):
            rotation_value = _lookup_nested(frame_data, rotation_name[len("frame_data.") :])
        elif rotation_name.startswith("newpram."):
            rotation_value = _lookup_nested(newpram, rotation_name[len("newpram.") :])
        else:
            raise ScaredA00ProtocolError(
                "extrinsics.rotation_source must use frame_data. or newpram."
            ) from error
        rotation = _matrix(rotation_value, f"extrinsics {rotation_name}", (3, 3))
        translation = torch.as_tensor(value, dtype=torch.float64).reshape(3)
        transform = torch.eye(4, dtype=torch.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
    scale = _translation_scale(protocol.extrinsic_units)
    transform = transform.clone()
    transform[:3, 3] *= scale
    if protocol.extrinsic_direction == "left_from_right":
        transform = torch.linalg.inv(transform)
    return transform


def _read_image(
    path: Path, *, image_range: str, device: torch.device | str | None = None
) -> torch.Tensor:
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (ImportError, OSError, ValueError) as error:
        raise ScaredA00SampleError(f"unable to decode RGB image {path}: {error}") from error
    destination = torch.device("cpu" if device is None else device)
    if destination.type == "cuda":
        # Transfer decoded bytes before any full-image floating-point work.
        pixels = torch.from_numpy(array).to(device=destination)
        if image_range == "uint8_0_255":
            # NumPy's float32 division rounds n/255 once. CUDA's float32
            # division may use a reciprocal approximation; a 256-entry table
            # rounded from double preserves the canonical byte mapping exactly.
            lookup = (torch.arange(256, dtype=torch.float64, device=destination) / 255.0).float()
            return lookup[pixels.long()].permute(2, 0, 1).contiguous()
        if image_range != "float01":  # pragma: no cover - validated by A00Protocol
            raise ScaredA00ProtocolError(f"unsupported image range {image_range!r}")
        tensor = pixels.float()
    elif image_range == "uint8_0_255":
        tensor = torch.from_numpy(array.astype(np.float32) / 255.0).to(destination)
    elif image_range == "float01":
        tensor = torch.from_numpy(array.astype(np.float32)).to(destination)
    else:  # pragma: no cover - validated by A00Protocol
        raise ScaredA00ProtocolError(f"unsupported image range {image_range!r}")
    tensor = tensor.permute(2, 0, 1).contiguous()
    if not bool(torch.isfinite(tensor).all()) or not bool(((tensor >= 0) & (tensor <= 1)).all()):
        raise ScaredA00SampleError(f"image {path} is not finite in [0,1]")
    return tensor


def _resize_rgb(image: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
    """Resize FP32 RGB [3,H,W] with the canonical half-pixel convention.

    The CUDA path supports identity and exactly twofold reduction, as used by
    the current 1280x1024 -> 640x512 run. At twofold reduction all four weights
    are 1/2 per axis. The canonical single-thread RGB CPU kernel accumulates
    four quarter-weight products left to right; CUDA interpolate groups terms
    differently. Explicit eager operations preserve the reference rounding.
    Require a single-thread reference because CPU dispatch changes its
    expression with thread count. See PyTorch v2.7.1 UpSampleKernel.cpp,
    cpu_upsample_linear_channels_last and upsample_bilinear2d_kernel_impl_float.
    Other CUDA scales fail explicitly rather than changing input numerics or
    silently processing images on CPU. CPU resizing retains the general path.
    """
    if image.device.type != "cuda":
        resized: torch.Tensor = functional.interpolate(
            image.unsqueeze(0), size=output_hw, mode="bilinear", align_corners=False
        )
        return resized.squeeze(0)
    if tuple(image.shape[-2:]) == output_hw:
        return image
    if tuple(image.shape[-2:]) != (2 * output_hw[0], 2 * output_hw[1]):
        raise ScaredA00SampleError(
            "CUDA RGB preparation requires identity or exactly twofold resize; "
            "other scales have no verified canonical rounding implementation"
        )
    if torch.get_num_threads() != 1:
        raise ScaredA00SampleError(
            "CUDA twofold RGB preparation requires OMP_NUM_THREADS=1 to pin "
            "the canonical CPU reference rounding"
        )
    result = image[:, 0::2, 0::2] * 0.25 + image[:, 0::2, 1::2] * 0.25
    result = result + image[:, 1::2, 0::2] * 0.25
    return result + image[:, 1::2, 1::2] * 0.25


def _read_disparity(path: Path, *, device: torch.device | str | None = None) -> torch.Tensor:
    try:
        import tifffile

        array = tifffile.imread(path)
    except (ImportError, OSError, ValueError) as error:
        raise ScaredA00SampleError(f"unable to decode disparity {path}: {error}") from error
    tensor = torch.as_tensor(array)
    if device is not None:
        tensor = tensor.to(device=device)
    # Keep conversion after transfer: a combined CPU->CUDA/dtype copy may
    # stage its dtype conversion on the host for some source TIFF dtypes.
    tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    if tensor.ndim != 2:
        raise ScaredA00SampleError(f"disparity {path} must be a 2-D TIFF")
    return tensor.contiguous()


def _load_supervision(
    path: Path,
    *,
    input_hw: tuple[int, int],
    crop_xy_left: tuple[int, int],
    crop_xy_right: tuple[int, int],
    crop_hw: tuple[int, int],
    output_hw: tuple[int, int] | None,
    device: torch.device | str | None = None,
) -> ScaredSupervision:
    """Canonical TIFF-to-supervision transform; returns disparity/mask [1,1,H,W].

    CPU and CUDA use the same crop offset, nearest-exact indices and scaling.
    Disk decoding stays on CPU; optional device preparation never reads RGB.
    """
    disparity = _read_disparity(path, device=device)
    if tuple(disparity.shape) != input_hw:
        raise ScaredA00SampleError(
            f"disparity shape {tuple(disparity.shape)} does not match processed image {input_hw}"
        )
    # Preserve invalid source targets before an asymmetric crop changes coordinates.
    source_valid = torch.isfinite(disparity) & (disparity > 0)
    ys = slice(crop_xy_left[1], crop_xy_left[1] + crop_hw[0])
    xs = slice(crop_xy_left[0], crop_xy_left[0] + crop_hw[1])
    source_valid = source_valid[ys, xs]
    disparity = disparity[ys, xs] - float(crop_xy_left[0] - crop_xy_right[0])
    if output_hw is not None:
        # Half-pixel nearest-exact centres match RGB/camera resize without
        # blending sparse holes or different surfaces into synthetic truth.
        disparity = functional.interpolate(
            disparity[None, None], size=output_hw, mode="nearest-exact"
        )[0, 0]
        disparity = disparity * (output_hw[1] / crop_hw[1])
        source_valid = functional.interpolate(
            source_valid[None, None].float(), size=output_hw, mode="nearest-exact"
        )[0, 0].bool()
    valid = source_valid & torch.isfinite(disparity) & (disparity > 0)
    safe = torch.where(torch.isfinite(disparity), disparity, torch.zeros_like(disparity))
    return ScaredSupervision(safe[None, None], valid[None, None])


class ScaredA00Adapter:
    """Read active SCARED samples and construct strict A00 deployments."""

    def __init__(
        self,
        root: Path,
        manifest: A01Manifest | Path,
        *,
        roles: A01ScaredDataRoles = DEFAULT_A01_SCARED_ROLES,
    ) -> None:
        self.root = Path(root)
        self.roles = roles
        self._file_hash_cache: dict[tuple[Path, int, int], str] = {}
        self.manifest = (
            load_a01_manifest(manifest, roles=roles) if isinstance(manifest, Path) else manifest
        )
        if self.manifest.role_contract_hash != roles.role_hash:
            raise ScaredA00SampleError(
                "A01 manifest role contract does not match active role configuration"
            )

    def __len__(self) -> int:
        return len(self.manifest.entries)

    def _source_hash(self, path: Path) -> str:
        """Hash one source member so target caches cannot survive data edits."""

        stat = path.stat()
        cache_key = (path, int(stat.st_size), int(stat.st_mtime_ns))
        cached = self._file_hash_cache.get(cache_key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self._file_hash_cache[cache_key] = value
        return value

    def __iter__(self) -> Iterator[ScaredA00Sample]:
        yield from self.iter_samples(include_supervision=True)

    def iter_samples(
        self,
        *,
        include_supervision: bool = True,
        output_hw: tuple[int, int] | None = None,
    ) -> Iterator[ScaredA00Sample]:
        """Iterate samples with deployment and optional training-only GT separated."""

        for entry in self.manifest.entries:
            yield self.load(
                entry.sample_id,
                data_role=entry.data_role,
                include_supervision=include_supervision,
                output_hw=output_hw,
            )

    def _paths(
        self, sample_id: str, *, include_supervision: bool
    ) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
        dataset_id, keyframe_id, frame_id = _safe_relative_id(sample_id)
        keyframe_dir = self.root / dataset_id / keyframe_id
        data_dir = keyframe_dir / "data" if (keyframe_dir / "data").is_dir() else keyframe_dir
        paths = (
            data_dir / "left_finalpass" / f"{frame_id}.png",
            data_dir / "right_finalpass" / f"{frame_id}.png",
            data_dir / "frame_data" / f"{frame_id}.json",
            data_dir / "newpram_data" / f"{frame_id}.json",
            data_dir / "disparity" / f"{frame_id}.tiff",
            data_dir / _PROTOCOL_FILENAME,
            keyframe_dir,
        )
        # Protocol metadata may be embedded in frame_data/newpram_data, so it
        # is validated by _load_protocol rather than treated as a path-only
        # requirement here.
        required_paths: tuple[Path, ...] = paths[:4]
        if include_supervision:
            required_paths += (paths[4],)
        missing = [
            str(path.relative_to(self.root)) for path in required_paths if not path.is_file()
        ]
        if missing:
            raise ScaredA00SampleError(
                f"missing active SCARED sample files for {sample_id}: {', '.join(missing)}"
            )
        return paths

    def source_members(
        self, sample_id: str, *, include_supervision: bool = False
    ) -> tuple[Path, ...]:
        """Canonical raw-byte provenance members; no images or GT tensors are decoded."""
        normalized = self.roles.validate_sample_id(sample_id, role="final_train", cache=False)
        paths = self._paths(normalized, include_supervision=include_supervision)
        members = list(paths[:5] if include_supervision else paths[:4])
        protocol = _protocol_path(paths[-1])
        if protocol.is_file():
            members.append(protocol)
        return tuple(members)

    def validate_deployment_files(self, sample_id: str, *, data_role: str) -> dict[str, object]:
        """Decode deployment RGB and validate calibration/protocol, without GT or hashing images.

        This CPU preflight uses the loader's metadata readers, RGB codec
        conversion and camera contract. It retains only shapes/counts,
        never deployment tensors.
        """

        normalized = self.roles.validate_sample_id(sample_id, role=data_role, cache=False)
        left_path, right_path, frame_path, newpram_path, _, _, keyframe_dir = self._paths(
            normalized, include_supervision=False
        )
        frame_data = _load_json(frame_path, "frame_data")
        newpram = _load_json(newpram_path, "newpram_data")
        protocol = _load_protocol(keyframe_dir, frame_data, newpram)
        camera = RectifiedStereoCamera(
            _matrix(newpram.get("intr0"), "newpram.intr0", (3, 3)).float().unsqueeze(0),
            _matrix(newpram.get("intr1"), "newpram.intr1", (3, 3)).float().unsqueeze(0),
            _extract_relative_transform(protocol, frame_data, newpram).float().unsqueeze(0),
            True,
        )

        def image_shape(path: Path) -> tuple[int, int]:
            # RGB conversion is the same codec operation as _read_image.
            # Pillow RGB is uint8: finiteness and 0..255 are structural.
            # Avoid allocating/checking full float tensors during preflight.
            try:
                from PIL import Image

                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    rgb.load()  # Decode the complete file, not just its header.
                    if protocol.image_range == "float01":
                        extrema = cast(tuple[tuple[int, int], ...], rgb.getextrema())
                        if any(low < 0 or high > 1 for low, high in extrema):
                            raise ScaredA00SampleError(f"image {path} is not finite in [0,1]")
                    return rgb.height, rgb.width
            except (OSError, ValueError) as error:
                raise ScaredA00SampleError(f"unable to decode RGB image {path}: {error}") from error

        left_shape = image_shape(left_path)
        right_shape = image_shape(right_path)
        if left_shape != right_shape:
            raise ScaredA00SampleError("left/right processed images must have matching shape")
        return {
            "sample_id": normalized,
            "data_role": data_role,
            "image_hw": left_shape,
            "baseline_m": float(camera.baseline.item()),
            "ground_truth_loaded": False,
        }

    def load(
        self,
        sample_id: str,
        *,
        data_role: str | None = None,
        crop_xy_left: tuple[int, int] = (0, 0),
        crop_xy_right: tuple[int, int] = (0, 0),
        crop_hw: tuple[int, int] | None = None,
        output_hw: tuple[int, int] | None = None,
        include_supervision: bool = True,
        device: torch.device | str | None = None,
    ) -> ScaredA00Sample:
        normalized = self.roles.validate_sample_id(
            sample_id,
            role=data_role or self.roles.role_for_dataset(_safe_relative_id(sample_id)[0]),
            cache=False,
        )
        _, _, frame_id = _safe_relative_id(normalized)
        entry_role = data_role or self.roles.role_for_dataset(normalized.split("/", maxsplit=1)[0])
        left_path, right_path, frame_path, newpram_path, disparity_path, _, keyframe_dir = (
            self._paths(normalized, include_supervision=include_supervision)
        )
        frame_data = _load_json(frame_path, "frame_data")
        newpram = _load_json(newpram_path, "newpram_data")
        protocol = _load_protocol(keyframe_dir, frame_data, newpram)
        # Decode bytes on the host; optional CUDA preparation performs the
        # normalization, crop and verified resize on the input device.
        left = _read_image(left_path, image_range=protocol.image_range, device=device)
        right = _read_image(right_path, image_range=protocol.image_range, device=device)
        if left.shape != right.shape:
            raise ScaredA00SampleError("left/right processed images must have matching shape")
        calibration_dtype = left.dtype
        k_left = (
            _matrix(newpram.get("intr0"), "newpram.intr0", (3, 3))
            .to(device=left.device, dtype=calibration_dtype)
            .unsqueeze(0)
        )
        k_right = (
            _matrix(newpram.get("intr1"), "newpram.intr1", (3, 3))
            .to(device=left.device, dtype=calibration_dtype)
            .unsqueeze(0)
        )
        transform = (
            _extract_relative_transform(protocol, frame_data, newpram)
            .to(device=left.device, dtype=calibration_dtype)
            .unsqueeze(0)
        )
        camera = RectifiedStereoCamera(k_left, k_right, transform, True)
        height, width = int(left.shape[1]), int(left.shape[2])
        if (
            len(crop_xy_left) != 2
            or len(crop_xy_right) != 2
            or any(type(value) is not int or value < 0 for value in (*crop_xy_left, *crop_xy_right))
        ):
            raise ScaredA00SampleError("crop origins must be non-negative integer (x, y) pairs")
        if crop_xy_left[1] != crop_xy_right[1]:
            raise ScaredA00SampleError(
                "left/right crop y origins must match for rectified A00 input"
            )
        resolved_crop_hw = crop_hw or (height - crop_xy_left[1], width - crop_xy_left[0])
        if len(resolved_crop_hw) != 2 or any(
            type(value) is not int or value < 1 for value in resolved_crop_hw
        ):
            raise ScaredA00SampleError("crop_hw must be positive")
        if output_hw is not None and (
            len(output_hw) != 2 or any(type(value) is not int or value < 1 for value in output_hw)
        ):
            raise ScaredA00SampleError("output_hw must contain positive integer height and width")
        if (
            crop_xy_left[1] + resolved_crop_hw[0] > height
            or crop_xy_left[0] + resolved_crop_hw[1] > width
        ):
            raise ScaredA00SampleError("left crop exceeds processed image dimensions")
        if (
            crop_xy_right[1] + resolved_crop_hw[0] > height
            or crop_xy_right[0] + resolved_crop_hw[1] > width
        ):
            raise ScaredA00SampleError("right crop exceeds processed image dimensions")
        left = left[
            :,
            crop_xy_left[1] : crop_xy_left[1] + resolved_crop_hw[0],
            crop_xy_left[0] : crop_xy_left[0] + resolved_crop_hw[1],
        ]
        right = right[
            :,
            crop_xy_right[1] : crop_xy_right[1] + resolved_crop_hw[0],
            crop_xy_right[0] : crop_xy_right[0] + resolved_crop_hw[1],
        ]
        if output_hw is not None:
            left = _resize_rgb(left, output_hw)
            right = _resize_rgb(right, output_hw)
        camera = crop_resize_camera(
            camera,
            left_crop_xy=crop_xy_left,
            right_crop_xy=crop_xy_right,
            crop_hw=resolved_crop_hw,
            output_hw=output_hw or resolved_crop_hw,
        )
        if device is not None:
            left, right = left.to(device=device), right.to(device=device)
            camera = RectifiedStereoCamera(
                camera.K_L.to(device=device),
                camera.K_R.to(device=device),
                camera.T_R_from_L.to(device=device),
                True,
            )
        support = torch.ones(
            (1, 1, left.shape[1], left.shape[2]), dtype=torch.bool, device=left.device
        )
        batch = DeploymentStereoBatch(left.unsqueeze(0), right.unsqueeze(0), camera, support)
        supervision: ScaredSupervision | None = None
        if include_supervision:
            supervision = _load_supervision(
                disparity_path,
                input_hw=(height, width),
                crop_xy_left=crop_xy_left,
                crop_xy_right=crop_xy_right,
                crop_hw=resolved_crop_hw,
                output_hw=output_hw,
                device=device,
            )
        protocol_path = _protocol_path(keyframe_dir)
        calibration_metadata = {
            "protocol_schema": protocol.schema_version,
            "protocol_identity": protocol.identity_hash,
            "image_range": protocol.image_range,
            "rectified": protocol.rectified,
            "distortion_removed": protocol.distortion_removed,
            "pixel_center": protocol.pixel_center,
            "disparity_units": protocol.disparity_units,
            "disparity_sign": protocol.disparity_sign,
            "disparity_reference": protocol.disparity_reference,
            "extrinsic_source": protocol.extrinsic_source,
            "extrinsic_direction": "right_from_left",
            "extrinsic_units": "m",
            "input_hw": [height, width],
            "crop_xy_left": list(crop_xy_left),
            "crop_xy_right": list(crop_xy_right),
            "crop_hw": list(resolved_crop_hw),
            "output_hw": list(output_hw or resolved_crop_hw),
            "runtime_resampling": {
                "rgb": "bilinear_align_corners_false",
                "supervision": A00_SUPERVISION_RESAMPLING,
                "validity": "same_nearest_exact_source_index_as_disparity",
            },
            "frame_id": frame_id,
            "manifest_hash": self.manifest.manifest_hash,
            "source_file_hashes": {
                "left_rgb": self._source_hash(left_path),
                "right_rgb": self._source_hash(right_path),
                "frame_data": self._source_hash(frame_path),
                "newpram_data": self._source_hash(newpram_path),
                "a00_protocol": (
                    self._source_hash(protocol_path) if protocol_path.is_file() else None
                ),
            },
        }
        return ScaredA00Sample(normalized, entry_role, batch, calibration_metadata, supervision)

    def load_supervision(self, sample: ScaredA00Sample) -> ScaredSupervision:
        """Load separate GT [1,1,H,W] on the deployment sample's device, without RGB decoding.

        Reuse only adapter-bound crop/resampling metadata after verifying the
        manifest, source files and protocol. No GT is attached to the input sample.
        """
        normalized = self.roles.validate_sample_id(
            sample.sample_id, role=sample.data_role, cache=False
        )
        left, right, frame, calibration, disparity, _, keyframe = self._paths(
            normalized, include_supervision=True
        )
        metadata = sample.calibration_metadata
        if metadata.get("frame_id") != _safe_relative_id(normalized)[2]:
            raise ScaredA00SampleError("supervision source frame mismatch")
        if metadata.get("manifest_hash") != self.manifest.manifest_hash:
            raise ScaredA00SampleError("supervision source manifest mismatch")
        protocol = _load_protocol(
            keyframe, _load_json(frame, "frame_data"), _load_json(calibration, "newpram_data")
        )
        if metadata.get("protocol_identity") != protocol.identity_hash:
            raise ScaredA00SampleError("supervision source protocol mismatch")
        expected_resampling = {
            "rgb": "bilinear_align_corners_false",
            "supervision": A00_SUPERVISION_RESAMPLING,
            "validity": "same_nearest_exact_source_index_as_disparity",
        }
        if metadata.get("runtime_resampling") != expected_resampling:
            raise ScaredA00SampleError("supervision resampling identity mismatch")
        hashes = _mapping(metadata.get("source_file_hashes"), "source_file_hashes")
        protocol_path = _protocol_path(keyframe)
        for name, path in (
            ("left_rgb", left),
            ("right_rgb", right),
            ("frame_data", frame),
            ("newpram_data", calibration),
            ("a00_protocol", protocol_path),
        ):
            current = self._source_hash(path) if path.is_file() else None
            if hashes.get(name) != current:
                raise ScaredA00SampleError(f"supervision source file changed: {name}")

        def pair(name: str, *, positive: bool) -> tuple[int, int]:
            value = metadata.get(name)
            if (
                not isinstance(value, (tuple, list))
                or len(value) != 2
                or any(type(x) is not int or x < (1 if positive else 0) for x in value)
            ):
                raise ScaredA00SampleError(f"invalid supervision {name}")
            return int(value[0]), int(value[1])

        input_hw = pair("input_hw", positive=True)
        crop_hw = pair("crop_hw", positive=True)
        crop_left = pair("crop_xy_left", positive=False)
        crop_right = pair("crop_xy_right", positive=False)
        output_hw = pair("output_hw", positive=True)
        if output_hw != tuple(sample.batch.left.shape[-2:]) or crop_left[1] != crop_right[1]:
            raise ScaredA00SampleError("supervision crop/output grid mismatch")
        if any(
            origin[axis] + crop_hw[1 - axis] > input_hw[1 - axis]
            for origin in (crop_left, crop_right)
            for axis in (0, 1)
        ):
            raise ScaredA00SampleError("supervision crop exceeds input grid")
        return _load_supervision(
            disparity,
            input_hw=input_hw,
            crop_xy_left=crop_left,
            crop_xy_right=crop_right,
            crop_hw=crop_hw,
            output_hw=output_hw,
            device=sample.batch.left.device,
        )


class ScaredA00Dataset(Dataset[ScaredA00Sample]):
    """PyTorch dataset wrapper that keeps deployment and supervision separate."""

    def __init__(
        self,
        adapter: ScaredA00Adapter,
        *,
        include_supervision: bool = True,
        data_role: str | None = None,
        output_hw: tuple[int, int] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.adapter = adapter
        self.include_supervision = include_supervision
        if data_role is not None:
            try:
                adapter.roles.datasets_for_role(data_role)
            except ValueError as error:
                raise ScaredA00SampleError(str(error)) from error
        self.data_role = data_role
        if output_hw is not None and (
            len(output_hw) != 2 or any(type(value) is not int or value < 1 for value in output_hw)
        ):
            raise ScaredA00SampleError("output_hw must contain positive integer height and width")
        self.output_hw = output_hw
        self.device = device
        self._entries = tuple(
            entry
            for entry in adapter.manifest.entries
            if data_role is None or entry.data_role == data_role
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> ScaredA00Sample:
        if index < 0 or index >= len(self):
            raise IndexError(f"sample index {index} out of range")
        entry = self._entries[index]
        return self.adapter.load(
            entry.sample_id,
            data_role=entry.data_role,
            include_supervision=self.include_supervision,
            output_hw=self.output_hw,
            device=self.device,
        )


def collate_scared_a00(samples: Sequence[ScaredA00Sample]) -> ScaredA00Sample:
    """Collate homogeneous A00 samples for batch-size-one server execution."""

    if not samples:
        raise ScaredA00SampleError("cannot collate an empty A00 batch")
    if len(samples) != 1:
        raise ScaredA00SampleError("A00 server protocol currently requires batch_size=1")
    return samples[0]


def build_a01_manifest(
    root: Path,
    output: Path,
    *,
    roles: A01ScaredDataRoles = DEFAULT_A01_SCARED_ROLES,
    require_all_folds: bool = True,
) -> A01Manifest:
    """Index processed folds 1–9 and assign final train/test roles."""

    root = Path(root)
    entries: list[A01ManifestEntry] = []
    missing_folds: list[str] = []
    for dataset_id in roles.dataset_ids:
        dataset_root = root / dataset_id
        if not dataset_root.is_dir():
            missing_folds.append(dataset_id)
            continue
        found_fold = False
        for keyframe_root in sorted(
            path for path in dataset_root.glob("keyframe_*") if path.is_dir()
        ):
            data_dir = (
                keyframe_root / "data" if (keyframe_root / "data").is_dir() else keyframe_root
            )
            for frame_path in sorted((data_dir / "frame_data").glob("*.json")):
                sample_id = make_scared_sample_id(dataset_id, keyframe_root.name, frame_path.stem)
                entries.append(A01ManifestEntry(sample_id, roles.role_for_dataset(dataset_id)))
                found_fold = True
        if not found_fold:
            missing_folds.append(dataset_id)
    if require_all_folds and missing_folds:
        raise ScaredA00SampleError(
            "active A01 manifest is missing required SCARED folds: " + ", ".join(missing_folds)
        )
    if not entries:
        raise ScaredA00SampleError(f"no processed SCARED samples found below {root}")
    manifest = A01Manifest(roles.role_hash, tuple(entries))
    write_json_atomic(Path(output), manifest.to_dict())
    return manifest


def load_a01_manifest(
    path: Path, *, roles: A01ScaredDataRoles = DEFAULT_A01_SCARED_ROLES
) -> A01Manifest:
    """Load and validate the active role-bound manifest."""

    try:
        value: object = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScaredA00SampleError(f"unable to read A01 manifest {path}: {error}") from error
    raw = _mapping(value, "A01 manifest")
    if raw.get("schema_version") != 1 or raw.get("dataset_name") != "scared":
        raise ScaredA00SampleError("A01 manifest must be schema 1 with dataset_name=scared")
    if raw.get("role_contract_hash") != roles.role_hash:
        raise ScaredA00SampleError(
            "A01 manifest role_contract_hash does not match active role policy"
        )
    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise ScaredA00SampleError("A01 manifest entries must be a sequence")
    entries: list[A01ManifestEntry] = []
    for index, item in enumerate(raw_entries):
        entry = _mapping(item, f"A01 manifest entries[{index}]")
        sample_id = entry.get("sample_id")
        data_role = entry.get("data_role")
        if not isinstance(sample_id, str) or not isinstance(data_role, str):
            raise ScaredA00SampleError(
                "A01 manifest entries require string sample_id and data_role"
            )
        normalized = roles.validate_sample_id(sample_id, role=data_role, cache=False)
        entries.append(A01ManifestEntry(normalized, data_role))
    return A01Manifest(str(raw["role_contract_hash"]), tuple(entries))


__all__ = [
    "A00Protocol",
    "A01Manifest",
    "A01ManifestEntry",
    "ScaredA00Adapter",
    "ScaredA00Dataset",
    "ScaredA00ProtocolError",
    "ScaredA00Sample",
    "ScaredA00SampleError",
    "ScaredSupervision",
    "build_a01_manifest",
    "collate_scared_a00",
    "load_a01_manifest",
]
