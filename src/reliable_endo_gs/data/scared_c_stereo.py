"""Stereo/video and geometry primitives for the audited SCARED-C layout.

The functions in this module own the camera contract used by the lazy SCARED-C
loader.  They intentionally operate on NumPy arrays and do not retain a
dataset-wide tensor cache.  A video frame is read on demand, archive members
are read into memory only for the requested frame, and rectification maps are
cached per keyframe/calibration identity.

Array conventions
------------------
Images are ``[H, W, 3]`` RGB ``uint8`` arrays.  XYZ maps are ``[H, W, 3]``
``float32`` arrays in millimetres with ``+X`` right, ``+Y`` down, and ``+Z``
forward.  Calibration matrices use the OpenCV convention.  Pixel coordinates
are zero-based image coordinates; rasterization uses deterministic nearest
pixel assignment with a nearest-positive-Z z-buffer.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import numpy as np

from reliable_endo_gs.data.calibration import CalibrationError, parse_opencv_calibration


class StereoContractError(ValueError):
    """Raised when an input cannot satisfy the frozen SCARED-C stereo contract."""


class ArchiveReadError(OSError):
    """Raised when a requested member cannot be read from a compressed archive."""


@dataclass(frozen=True, slots=True)
class StereoCalibrationContract:
    """Raw and corrected-camera calibration for one SCARED-C keyframe."""

    M1: np.ndarray
    D1: np.ndarray
    M2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    K_colmap: np.ndarray | None
    image_size: tuple[int, int]

    @property
    def baseline(self) -> float:
        """Return ``||T||`` in millimetres."""

        return float(np.linalg.norm(self.T.reshape(3)))


@dataclass(frozen=True, slots=True)
class RectificationMaps:
    """Cached OpenCV stereo-rectification matrices and remap tables."""

    image_size: tuple[int, int]
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    left_map_x: np.ndarray
    left_map_y: np.ndarray
    right_map_x: np.ndarray
    right_map_y: np.ndarray

    @property
    def fx_rect(self) -> float:
        """Return the rectified left focal length in pixels."""

        return float(self.P1[0, 0])


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Metadata obtained without decoding a video frame."""

    width: int
    height: int
    frame_count: int
    fps: float


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """Projected image coordinates and camera-frame depth."""

    u: np.ndarray
    v: np.ndarray
    z: np.ndarray
    finite: np.ndarray


@dataclass(frozen=True, slots=True)
class RasterizedDepth:
    """Depth raster and the explicit geometry-only validity mask."""

    depth: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class DirectDisparity:
    """Directly projected left-minus-right disparity for source XYZ points."""

    disparity: np.ndarray
    valid_mask: np.ndarray
    left_rectified_xyz: np.ndarray
    right_rectified_xyz: np.ndarray


@dataclass(frozen=True, slots=True)
class RectifiedGeometry:
    """Canonical depth/disparity products made from one left XYZ map."""

    xyz_left_rect: np.ndarray
    depth_left_rect: np.ndarray
    valid_depth_mask: np.ndarray
    disparity_left_rect: np.ndarray
    valid_disparity_mask: np.ndarray


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - depends on the runtime image
        raise ImportError(
            "SCARED-C video and rectification require the OpenCV Python package"
        ) from error
    return cv2


def _as_float_array(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Convert a parsed OpenCV matrix to a finite float64 NumPy matrix."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise StereoContractError(f"{name} is not numeric") from error
    if array.shape != shape:
        raise StereoContractError(f"{name} must have shape {shape}; got {array.shape}")
    if not np.isfinite(array).all():
        raise StereoContractError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _as_distortion(value: object, *, name: str) -> np.ndarray:
    """Normalize a five-coefficient distortion vector to ``[1, 5]``."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise StereoContractError(f"{name} is not numeric") from error
    if array.size != 5:
        raise StereoContractError(f"{name} must contain five coefficients; got {array.shape}")
    array = array.reshape(1, 5)
    if not np.isfinite(array).all():
        raise StereoContractError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _as_translation(value: object, *, name: str) -> np.ndarray:
    """Normalize a three-value translation to a column vector."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise StereoContractError(f"{name} is not numeric") from error
    if array.size != 3:
        raise StereoContractError(f"{name} must contain three values; got {array.shape}")
    array = array.reshape(3, 1)
    if not np.isfinite(array).all():
        raise StereoContractError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def load_stereo_calibration(
    endoscope_calibration_path: Path,
    *,
    image_size: tuple[int, int],
    colmap_intrinsics_path: Path | None = None,
) -> StereoCalibrationContract:
    """Load the frozen raw stereo calibration and optional corrected ``K``.

    ``R`` and ``T`` retain the audited direction ``X_right = R @ X_left + T``;
    no inverse or sign correction is applied here.
    """

    try:
        raw = parse_opencv_calibration(endoscope_calibration_path)
    except CalibrationError as error:
        raise StereoContractError(f"invalid endoscope calibration: {error}") from error
    required = {"M1", "D1", "M2", "D2", "R", "T"}
    missing = sorted(name for name in required if name not in raw or raw[name] is None)
    if missing:
        raise StereoContractError(f"endoscope calibration is missing {', '.join(missing)}")

    colmap: np.ndarray | None = None
    if colmap_intrinsics_path is not None and colmap_intrinsics_path.is_file():
        try:
            colmap_raw = parse_opencv_calibration(colmap_intrinsics_path)
        except CalibrationError as error:
            raise StereoContractError(f"invalid COLMAP intrinsics: {error}") from error
        value = colmap_raw.get("K")
        if value is None:
            raise StereoContractError("COLMAP intrinsics are missing K")
        colmap = _as_float_array(value, name="K_colmap", shape=(3, 3))

    width, height = image_size
    if width <= 0 or height <= 0:
        raise StereoContractError(f"image_size must be positive; got {image_size}")
    return StereoCalibrationContract(
        M1=_as_float_array(raw["M1"], name="M1", shape=(3, 3)),
        D1=_as_distortion(raw["D1"], name="D1"),
        M2=_as_float_array(raw["M2"], name="M2", shape=(3, 3)),
        D2=_as_distortion(raw["D2"], name="D2"),
        R=_as_float_array(raw["R"], name="R", shape=(3, 3)),
        T=_as_translation(raw["T"], name="T"),
        K_colmap=colmap,
        image_size=(int(width), int(height)),
    )


def create_rectification_maps(
    calibration: StereoCalibrationContract,
    *,
    image_size: tuple[int, int] | None = None,
) -> RectificationMaps:
    """Create OpenCV rectification maps once for one calibration identity."""

    cv2 = _require_cv2()
    size = calibration.image_size if image_size is None else image_size
    if size != calibration.image_size:
        raise StereoContractError(
            f"rectification image size {size} does not match calibration {calibration.image_size}"
        )
    try:
        R1, R2, P1, P2, Q, _roi_left, _roi_right = cv2.stereoRectify(
            calibration.M1,
            calibration.D1,
            calibration.M2,
            calibration.D2,
            size,
            calibration.R,
            calibration.T,
            alpha=-1,
        )
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(
            calibration.M1,
            calibration.D1,
            R1,
            P1,
            size,
            cv2.CV_32FC1,
        )
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(
            calibration.M2,
            calibration.D2,
            R2,
            P2,
            size,
            cv2.CV_32FC1,
        )
    except cv2.error as error:
        raise StereoContractError(f"OpenCV stereo rectification failed: {error}") from error
    return RectificationMaps(
        image_size=(int(size[0]), int(size[1])),
        R1=np.ascontiguousarray(R1, dtype=np.float64),
        R2=np.ascontiguousarray(R2, dtype=np.float64),
        P1=np.ascontiguousarray(P1, dtype=np.float64),
        P2=np.ascontiguousarray(P2, dtype=np.float64),
        Q=np.ascontiguousarray(Q, dtype=np.float64),
        left_map_x=np.ascontiguousarray(left_map_x, dtype=np.float32),
        left_map_y=np.ascontiguousarray(left_map_y, dtype=np.float32),
        right_map_x=np.ascontiguousarray(right_map_x, dtype=np.float32),
        right_map_y=np.ascontiguousarray(right_map_y, dtype=np.float32),
    )


class RectificationCache:
    """Thread-safe in-memory cache keyed by keyframe and image size."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, tuple[int, int]], RectificationMaps] = {}
        self._lock = RLock()

    def get_or_create(
        self,
        keyframe_root: Path,
        calibration: StereoCalibrationContract,
    ) -> RectificationMaps:
        key = (str(keyframe_root.resolve()), calibration.image_size)
        with self._lock:
            maps = self._entries.get(key)
            if maps is None:
                maps = create_rectification_maps(calibration)
                self._entries[key] = maps
            return maps

    def clear(self) -> None:
        """Drop cached maps without touching source data."""

        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def split_stacked_stereo_frame(
    frame: np.ndarray,
    *,
    expected_width: int = 1280,
    expected_height: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a stacked RGB frame into top/left and bottom/right views."""

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise StereoContractError("stacked stereo frame must have shape [H, W, 3]")
    if frame.shape[:2] != (expected_height, expected_width):
        raise StereoContractError(
            "stacked stereo frame has unexpected dimensions: "
            f"got {(frame.shape[1], frame.shape[0])}, expected "
            f"{(expected_width, expected_height)}"
        )
    half_height = expected_height // 2
    return frame[:half_height], frame[half_height:]


class StackedStereoVideo:
    """Lazy OpenCV reader for a 1280x2048 top/bottom stereo video."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._capture: Any | None = None
        self._lock = RLock()

    def _open(self) -> Any:
        cv2 = _require_cv2()
        if self._capture is None:
            capture = cv2.VideoCapture(str(self.path))
            if not capture.isOpened():
                capture.release()
                raise OSError(f"unable to open SCARED-C video: {self.path}")
            self._capture = capture
        return self._capture

    def metadata(self) -> VideoMetadata:
        """Read dimensions/count without decoding any frame."""

        cv2 = _require_cv2()
        with self._lock:
            capture = self._open()
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        if (width, height) != (1280, 2048):
            raise StereoContractError(
                f"SCARED-C stereo video must be 1280x2048; got {(width, height)}"
            )
        if count <= 0:
            raise StereoContractError("SCARED-C stereo video has no readable frame count")
        return VideoMetadata(width, height, count, fps)

    def read(self, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Read one one-based frame and return RGB ``(left, right)`` arrays."""

        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 1:
            raise ValueError("frame_id must be a positive one-based integer")
        cv2 = _require_cv2()
        with self._lock:
            capture = self._open()
            metadata = self.metadata()
            if frame_id > metadata.frame_count:
                raise IndexError(
                    f"frame_id {frame_id} exceeds video frame count {metadata.frame_count}"
                )
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1):
                raise OSError(f"unable to seek video frame {frame_id} in {self.path}")
            ok, bgr = capture.read()
        if not ok or bgr is None:
            raise OSError(f"unable to decode video frame {frame_id} in {self.path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return split_stacked_stereo_frame(rgb)

    def close(self) -> None:
        """Release the OpenCV handle."""

        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    def __enter__(self) -> StackedStereoVideo:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def read_tar_member_bytes(archive_path: Path, member_name: str) -> bytes:
    """Read one member into memory without extracting an archive directory."""

    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            member = archive.getmember(member_name)
            if not member.isfile():
                raise ArchiveReadError(f"archive member is not a regular file: {member_name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ArchiveReadError(f"archive member cannot be read: {member_name}")
            return stream.read()
    except (OSError, KeyError, tarfile.TarError) as error:
        if isinstance(error, ArchiveReadError):
            raise
        raise ArchiveReadError(
            f"unable to read {member_name} from {archive_path}: {error}"
        ) from error


def read_archive_rgb(archive_path: Path, member_name: str) -> np.ndarray:
    """Decode one archived RGB PNG into an RGB ``uint8`` array."""

    try:
        from PIL import Image

        with Image.open(io.BytesIO(read_tar_member_bytes(archive_path, member_name))) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    except (ImportError, OSError, ValueError) as error:
        raise ArchiveReadError(f"unable to decode RGB member {member_name}: {error}") from error


def read_archive_xyz(archive_path: Path, member_name: str) -> np.ndarray:
    """Decode one archived float32 XYZ TIFF without extracting it."""

    try:
        return decode_xyz_tiff(read_tar_member_bytes(archive_path, member_name), member_name)
    except (ImportError, OSError, ValueError, tarfile.TarError) as error:
        raise ArchiveReadError(f"unable to decode XYZ member {member_name}: {error}") from error


def decode_xyz_tiff(payload: bytes, member_name: str) -> np.ndarray:
    """Decode and validate one XYZ TIFF payload obtained from an open archive."""

    try:
        import tifffile

        array = np.asarray(tifffile.imread(io.BytesIO(payload)))
    except (ImportError, OSError, ValueError) as error:
        raise ArchiveReadError(f"unable to decode XYZ member {member_name}: {error}") from error
    if array.ndim != 3 or array.shape[-1] != 3:
        raise StereoContractError(
            f"XYZ member {member_name} must have shape [H, W, 3]; got {array.shape}"
        )
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)
    return np.ascontiguousarray(array)


def read_xyz_tar_member(archive: tarfile.TarFile, member_name: str) -> np.ndarray:
    """Decode one XYZ TIFF from an already-open tar stream.

    Keeping the archive open lets the Stage1 derived-cache builder make one
    sequential pass over each selected sequence instead of restarting gzip
    decompression for every frame.
    """

    try:
        member = archive.getmember(member_name)
        if not member.isfile():
            raise ArchiveReadError(f"archive member is not a regular file: {member_name}")
        stream = archive.extractfile(member)
        if stream is None:
            raise ArchiveReadError(f"archive member cannot be read: {member_name}")
        return decode_xyz_tiff(stream.read(), member_name)
    except (KeyError, tarfile.TarError) as error:
        raise ArchiveReadError(f"unable to read {member_name} from open archive: {error}") from error


def read_static_rgb(path: Path) -> np.ndarray:
    """Decode a static RGB PNG into an RGB ``uint8`` array."""

    try:
        from PIL import Image

        with Image.open(path) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    except (ImportError, OSError, ValueError) as error:
        raise OSError(f"unable to decode static RGB image {path}: {error}") from error


def read_static_xyz(path: Path) -> np.ndarray:
    """Decode a static float32 XYZ TIFF."""

    try:
        import tifffile

        array = np.asarray(tifffile.imread(path))
    except (ImportError, OSError, ValueError) as error:
        raise OSError(f"unable to decode static XYZ map {path}: {error}") from error
    if array.ndim != 3 or array.shape[-1] != 3:
        raise StereoContractError(f"static XYZ must have shape [H, W, 3]; got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def read_frame_pose(archive_path: Path, member_name: str) -> np.ndarray:
    """Read the raw ``camera-pose`` matrix without inverting it."""

    try:
        payload = json.loads(read_tar_member_bytes(archive_path, member_name).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArchiveReadError(f"invalid frame metadata {member_name}: {error}") from error
    if not isinstance(payload, dict) or "camera-pose" not in payload:
        raise StereoContractError(f"frame metadata {member_name} has no camera-pose")
    pose = np.asarray(payload["camera-pose"], dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise StereoContractError(f"camera-pose must be finite 4x4; got {pose.shape}")
    return np.ascontiguousarray(pose)


def valid_xyz_mask(xyz: np.ndarray) -> np.ndarray:
    """Return the frozen XYZ validity mask ``finite_xyz & (Z > 0)``."""

    array = np.asarray(xyz)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise StereoContractError(f"XYZ must have shape [H, W, 3]; got {array.shape}")
    return np.isfinite(array).all(axis=-1) & (array[..., 2] > 0)


def transform_left_to_right(xyz_left: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply the audited raw stereo direction ``X_right = R @ X_left + T``."""

    points = np.asarray(xyz_left, dtype=np.float64)
    rotation = _as_float_array(R, name="R", shape=(3, 3))
    translation = np.asarray(T, dtype=np.float64).reshape(-1)
    if translation.size != 3 or not np.isfinite(translation).all():
        raise StereoContractError("T must contain three finite values")
    if points.ndim < 1 or points.shape[-1] != 3:
        raise StereoContractError("xyz_left must end in a length-three coordinate axis")
    return np.ascontiguousarray(points @ rotation.T + translation)


def rectify_xyz_pair(
    xyz_left: np.ndarray,
    calibration: StereoCalibrationContract,
    rectification: RectificationMaps,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform corrected left XYZ and its raw-stereo right counterpart."""

    left = np.asarray(xyz_left, dtype=np.float64)
    if left.ndim != 3 or left.shape[-1] != 3:
        raise StereoContractError("xyz_left must have shape [H, W, 3]")
    right_raw = transform_left_to_right(left, calibration.R, calibration.T)
    left_rect = np.ascontiguousarray(left @ rectification.R1.T)
    right_rect = np.ascontiguousarray(right_raw @ rectification.R2.T)
    return left_rect, right_rect


def project_xyz(xyz: np.ndarray, projection: np.ndarray) -> ProjectionResult:
    """Project camera-frame XYZ with a 3x4 pinhole projection matrix."""

    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise StereoContractError("xyz must end in a length-three coordinate axis")
    matrix = _as_float_array(projection, name="projection", shape=(3, 4))
    homogeneous = np.concatenate(
        (points, np.ones(points.shape[:-1] + (1,), dtype=np.float64)), axis=-1
    )
    projected = homogeneous @ matrix.T
    denominator = projected[..., 2]
    finite = np.isfinite(projected).all(axis=-1) & np.isfinite(denominator) & (denominator != 0)
    u = np.full(denominator.shape, np.nan, dtype=np.float64)
    v = np.full(denominator.shape, np.nan, dtype=np.float64)
    valid = finite
    u[valid] = projected[..., 0][valid] / denominator[valid]
    v[valid] = projected[..., 1][valid] / denominator[valid]
    return ProjectionResult(u=u, v=v, z=denominator, finite=finite)


def rasterize_nearest_depth(
    xyz_rect: np.ndarray,
    projection: np.ndarray,
    *,
    output_size: tuple[int, int],
    source_valid_mask: np.ndarray | None = None,
) -> RasterizedDepth:
    """Rasterize rectified XYZ using nearest-pixel, nearest-positive-Z semantics."""

    width, height = output_size
    if min(width, height) <= 0:
        raise StereoContractError("output_size must be positive")
    points = np.asarray(xyz_rect, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise StereoContractError("xyz_rect must have shape [H, W, 3]")
    projection_result = project_xyz(points, projection)
    valid = (
        projection_result.finite
        & np.isfinite(points).all(axis=-1)
        & (projection_result.z > 0)
        & (projection_result.u >= 0)
        & (projection_result.u < width)
        & (projection_result.v >= 0)
        & (projection_result.v < height)
    )
    if source_valid_mask is not None:
        source_mask = np.asarray(source_valid_mask, dtype=bool)
        if source_mask.shape != valid.shape:
            raise StereoContractError("source_valid_mask does not match XYZ spatial shape")
        valid &= source_mask

    depth = np.full((height, width), np.nan, dtype=np.float32)
    if not valid.any():
        return RasterizedDepth(depth=depth, valid_mask=np.zeros_like(depth, dtype=bool))

    # ``floor(x + 0.5)`` is explicit about the tie rule and avoids NumPy's
    # banker's rounding for exact half-pixel projections.
    pixel_x = np.floor(projection_result.u[valid] + 0.5).astype(np.int64)
    pixel_y = np.floor(projection_result.v[valid] + 0.5).astype(np.int64)
    inside = (pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height)
    pixel_x = pixel_x[inside]
    pixel_y = pixel_y[inside]
    z = projection_result.z[valid][inside]
    linear = pixel_y * width + pixel_x
    z_buffer = np.full(width * height, np.inf, dtype=np.float64)
    np.minimum.at(z_buffer, linear, z)
    z_buffer[z_buffer == np.inf] = np.nan
    depth = z_buffer.reshape(height, width).astype(np.float32)
    return RasterizedDepth(depth=depth, valid_mask=np.isfinite(depth) & (depth > 0))


def disparity_from_rectified_depth(
    depth_left_rect: np.ndarray,
    rectification: RectificationMaps,
    calibration: StereoCalibrationContract,
    *,
    valid_depth_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct positive left-reference disparity using ``d = fx * ||T|| / Z``."""

    depth = np.asarray(depth_left_rect, dtype=np.float64)
    if depth.ndim != 2 or depth.shape[::-1] != rectification.image_size:
        raise StereoContractError(
            f"depth must have shape {(rectification.image_size[1], rectification.image_size[0])}"
        )
    valid = np.isfinite(depth) & (depth > 0)
    if valid_depth_mask is not None:
        supplied = np.asarray(valid_depth_mask, dtype=bool)
        if supplied.shape != depth.shape:
            raise StereoContractError("valid_depth_mask does not match depth shape")
        valid &= supplied
    fx = rectification.fx_rect
    baseline = calibration.baseline
    if not np.isfinite(fx) or fx <= 0 or not np.isfinite(baseline) or baseline <= 0:
        raise StereoContractError("rectified focal length and baseline must be positive")
    disparity = np.full(depth.shape, np.nan, dtype=np.float32)
    disparity[valid] = (fx * baseline / depth[valid]).astype(np.float32)
    valid &= np.isfinite(disparity) & (disparity > 0)
    disparity[~valid] = np.nan
    return disparity, valid


def build_rectified_geometry(
    xyz_left: np.ndarray,
    calibration: StereoCalibrationContract,
    rectification: RectificationMaps,
) -> RectifiedGeometry:
    """Build canonical rectified depth and disparity from corrected left XYZ."""

    xyz = np.asarray(xyz_left, dtype=np.float32)
    source_valid = valid_xyz_mask(xyz)
    xyz_left_rect, _xyz_right_rect = rectify_xyz_pair(xyz, calibration, rectification)
    raster = rasterize_nearest_depth(
        xyz_left_rect,
        rectification.P1,
        output_size=rectification.image_size,
        source_valid_mask=source_valid,
    )
    disparity, valid_disparity = disparity_from_rectified_depth(
        raster.depth,
        rectification,
        calibration,
        valid_depth_mask=raster.valid_mask,
    )
    return RectifiedGeometry(
        xyz_left_rect=np.ascontiguousarray(xyz_left_rect, dtype=np.float32),
        depth_left_rect=raster.depth,
        valid_depth_mask=raster.valid_mask,
        disparity_left_rect=disparity,
        valid_disparity_mask=valid_disparity,
    )


def direct_projected_disparity(
    xyz_left: np.ndarray,
    calibration: StereoCalibrationContract,
    rectification: RectificationMaps,
) -> DirectDisparity:
    """Compute reference disparity by projecting both rectified 3D points."""

    source_valid = valid_xyz_mask(np.asarray(xyz_left))
    left_rect, right_rect = rectify_xyz_pair(xyz_left, calibration, rectification)
    # ``left_rect`` and ``right_rect`` are already in their respective
    # rectified camera frames.  Apply the rectified intrinsic blocks only;
    # using the fourth column of P2 here would apply T a second time.
    left_intrinsic_projection = np.concatenate(
        (rectification.P1[:, :3], np.zeros((3, 1), dtype=np.float64)), axis=1
    )
    right_intrinsic_projection = np.concatenate(
        (rectification.P2[:, :3], np.zeros((3, 1), dtype=np.float64)), axis=1
    )
    left_projection = project_xyz(left_rect, left_intrinsic_projection)
    right_projection = project_xyz(right_rect, right_intrinsic_projection)
    width, height = rectification.image_size
    valid = (
        source_valid
        & left_projection.finite
        & right_projection.finite
        & (left_projection.z > 0)
        & (right_projection.z > 0)
        & (left_projection.u >= 0)
        & (left_projection.u < width)
        & (left_projection.v >= 0)
        & (left_projection.v < height)
        & (right_projection.u >= 0)
        & (right_projection.u < width)
        & (right_projection.v >= 0)
        & (right_projection.v < height)
    )
    disparity = np.full(left_projection.u.shape, np.nan, dtype=np.float32)
    disparity[valid] = (left_projection.u[valid] - right_projection.u[valid]).astype(np.float32)
    valid &= np.isfinite(disparity) & (disparity > 0)
    disparity[~valid] = np.nan
    return DirectDisparity(
        disparity=disparity,
        valid_mask=valid,
        left_rectified_xyz=left_rect,
        right_rectified_xyz=right_rect,
    )


def reproject_with_q(
    u: np.ndarray | float,
    v: np.ndarray | float,
    disparity: np.ndarray | float,
    Q: np.ndarray,
) -> np.ndarray:
    """Reproject rectified pixels and positive disparity with OpenCV ``Q``.

    The result has shape ``[..., 3]`` and is expressed in the rectified left
    camera frame.  Invalid homogeneous denominators are returned as NaNs.
    """

    matrix = _as_float_array(Q, name="Q", shape=(4, 4))
    u_array, v_array, d_array = np.broadcast_arrays(
        np.asarray(u, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
        np.asarray(disparity, dtype=np.float64),
    )
    homogeneous = np.stack((u_array, v_array, d_array, np.ones_like(d_array)), axis=-1)
    projected = homogeneous @ matrix.T
    result = np.full(projected.shape[:-1] + (3,), np.nan, dtype=np.float64)
    valid = np.isfinite(projected).all(axis=-1) & (projected[..., 3] != 0)
    result[valid] = projected[valid, :3] / projected[valid, 3, None]
    return result


_FRAME_MEMBER_PATTERNS: dict[str, re.Pattern[str]] = {
    "frame_data": re.compile(r"^frame_data(\d+)\.json$", re.IGNORECASE),
    "rgb_frames": re.compile(r"^frame(\d+)\.png$", re.IGNORECASE),
    "scene_points": re.compile(r"^scene_points(\d+)\.tiff$", re.IGNORECASE),
}


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    """One deterministic frame member discovered in a tar archive."""

    frame_id: int
    name: str


def parse_archive_frame_member(
    name: str, kind: Literal["frame_data", "rgb_frames", "scene_points"]
) -> int | None:
    """Parse a one-based frame ID from a canonical archive member basename."""

    basename = Path(name.replace("\\", "/")).name
    match = _FRAME_MEMBER_PATTERNS[kind].fullmatch(basename)
    return None if match is None else int(match.group(1))


def index_archive_members(
    archive_path: Path,
    *,
    kind: Literal["frame_data", "rgb_frames", "scene_points"],
) -> tuple[ArchiveMember, ...]:
    """Index frame members without extracting or decoding their contents.

    Members are returned in deterministic name order.  Two members that map to
    the same frame ID are rejected even when their archive paths differ.
    """

    discovered: list[ArchiveMember] = []
    seen_frame_ids: dict[int, str] = {}
    seen_names: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                normalized_name = member.name.replace("\\", "/")
                if normalized_name in seen_names:
                    raise StereoContractError(
                        f"duplicate archive member name {normalized_name!r} in {archive_path}"
                    )
                seen_names.add(normalized_name)
                frame_id = parse_archive_frame_member(normalized_name, kind)
                if frame_id is None:
                    continue
                previous = seen_frame_ids.get(frame_id)
                if previous is not None:
                    raise StereoContractError(
                        f"duplicate {kind} frame ID {frame_id}: {previous!r} and "
                        f"{normalized_name!r}"
                    )
                if frame_id < 1:
                    raise StereoContractError(
                        f"archive frame IDs must be one-based: {normalized_name}"
                    )
                seen_frame_ids[frame_id] = normalized_name
                discovered.append(ArchiveMember(frame_id, normalized_name))
    except (OSError, tarfile.TarError) as error:
        if isinstance(error, StereoContractError):
            raise
        raise ArchiveReadError(f"unable to index archive {archive_path}: {error}") from error
    return tuple(
        sorted(discovered, key=lambda item: (item.frame_id, item.name.casefold(), item.name))
    )


def archive_member_map(
    members: tuple[ArchiveMember, ...],
) -> dict[int, str]:
    """Convert a validated archive index to a frame-ID lookup map."""

    return {member.frame_id: member.name for member in members}


def validate_archive_frame_identity(
    frame_log_ids: set[int],
    archive_members: dict[str, tuple[ArchiveMember, ...]],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Return consistently present IDs and deterministic missing-member issues."""

    if set(archive_members) != {"frame_data", "rgb_frames", "scene_points"}:
        raise StereoContractError("archive_members must contain all three corrected-video archives")
    maps = {kind: archive_member_map(members) for kind, members in archive_members.items()}
    common = (
        frame_log_ids
        & set(maps["frame_data"])
        & set(maps["rgb_frames"])
        & set(maps["scene_points"])
    )
    issues: list[str] = []
    for frame_id in sorted(frame_log_ids):
        missing = [kind for kind, members in maps.items() if frame_id not in members]
        if missing:
            issues.append(f"frame {frame_id} missing from {', '.join(missing)}")
    return tuple(sorted(common)), tuple(issues)
