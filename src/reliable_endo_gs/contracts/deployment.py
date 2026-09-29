"""Deployment-only rectified stereo, metres and integer pixel centres (v1)."""

from dataclasses import dataclass

import torch

from reliable_endo_gs.contracts.cameras import CameraBatch
from reliable_endo_gs.contracts.samples import StereoBatch


def _close(value: torch.Tensor, expected: torch.Tensor) -> bool:
    return bool(torch.allclose(value, expected, atol=1e-6, rtol=1e-6))


@dataclass(frozen=True, slots=True)
class RectifiedStereoCamera:
    """K_L/K_R [B,3,3], T_R_from_L [B,4,4]; x-right/y-down/z-forward.

    Caller explicitly certifies distortion removal/rectification. Equal focal
    lengths and scanline origins are required; horizontal principal points
    may differ. Right camera lies at positive left-camera x, in metres.
    Calibration uses FP32/FP64 even when RGB uses mixed precision.
    """

    K_L: torch.Tensor
    K_R: torch.Tensor
    T_R_from_L: torch.Tensor
    rectified: bool

    def __post_init__(self) -> None:
        if self.rectified is not True:
            raise ValueError("explicit rectified=True certification is required")
        left = self.K_L
        if left.ndim != 3 or left.shape[1:] != (3, 3) or left.shape[0] < 1:
            raise ValueError("K_L must be [B,3,3] with B > 0")
        for value, shape in (
            (left, left.shape),
            (self.K_R, left.shape),
            (self.T_R_from_L, (left.shape[0], 4, 4)),
        ):
            if value.shape != shape or value.dtype not in (torch.float32, torch.float64):
                raise ValueError("calibration requires matching shapes and FP32/FP64")
            if value.device != left.device or value.dtype != left.dtype:
                raise ValueError("calibration must share dtype/device")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("calibration must be finite")
        for k in (left, self.K_R):
            expected = torch.zeros_like(k)
            expected[:, 0, 0] = k[:, 0, 0]
            expected[:, 1, 1] = k[:, 1, 1]
            expected[:, :2, 2] = k[:, :2, 2]
            expected[:, 2, 2] = 1
            if not _close(k, expected) or not bool((k[:, (0, 1), (0, 1)] > 0).all()):
                raise ValueError("intrinsics require positive focal lengths and zero skew")
        if not _close(left[:, (0, 1), (0, 1)], self.K_R[:, (0, 1), (0, 1)]):
            raise ValueError("rectified focal lengths must match")
        if not _close(left[:, 1, 2], self.K_R[:, 1, 2]):
            raise ValueError("rectified scanline origins must match")
        expected_t = torch.eye(4, dtype=left.dtype, device=left.device).repeat(left.shape[0], 1, 1)
        expected_t[:, 0, 3] = self.T_R_from_L[:, 0, 3]
        if not _close(self.T_R_from_L, expected_t) or not bool((self.baseline > 0).all()):
            raise ValueError("expected identity rotation and strictly negative x translation")

    @property
    def baseline(self) -> torch.Tensor:
        """Positive baseline [B] in metres."""
        return -self.T_R_from_L[:, 0, 3]

    @classmethod
    def from_cameras(
        cls, left: CameraBatch, right: CameraBatch, *, rectified: bool
    ) -> "RectifiedStereoCamera":
        """Convert camera-to-world to right-from-left, without inferred units."""
        for camera in (left, right):
            t = camera.world_from_camera
            r = t[:, :3, :3]
            eye = torch.eye(3, dtype=r.dtype, device=r.device).expand_as(r)
            row = t.new_tensor([0, 0, 0, 1]).expand_as(t[:, 3])
            if not _close(r.transpose(-1, -2) @ r, eye) or not _close(t[:, 3], row):
                raise ValueError("camera-to-world must be a rigid homogeneous transform")
            if not bool((torch.linalg.det(r) > 0).all()):
                raise ValueError("camera rotation must be right handed")
        if left.intrinsics.shape != right.intrinsics.shape:
            raise ValueError("camera batches must match")
        relative = torch.linalg.solve(right.world_from_camera, left.world_from_camera)
        return cls(left.intrinsics, right.intrinsics, relative, rectified)


@dataclass(frozen=True, slots=True)
class DeploymentStereoBatch:
    """RGB [B,3,H,W] in [0,1]; support [B,1,H,W] bool, no GT/metadata.

    Support describes sensor/rectification availability, never GT validity.
    Tensors are borrowed; callers must not mutate a live request.
    """

    left: torch.Tensor
    right: torch.Tensor
    camera: RectifiedStereoCamera
    support: torch.Tensor

    def __post_init__(self) -> None:
        if self.left.ndim != 4 or self.left.shape[1] != 3 or min(self.left.shape) < 1:
            raise ValueError("RGB must be [B,3,H,W]")
        for image in (self.left, self.right):
            if image.shape != self.left.shape or not image.is_floating_point():
                raise ValueError("paired RGB shapes must match and be floating point")
            if image.device != self.left.device or image.dtype != self.left.dtype:
                raise ValueError("RGB dtype/device must match")
            if not bool((torch.isfinite(image) & (image >= 0) & (image <= 1)).all()):
                raise ValueError("RGB must be finite in [0,1]")
        b, _, h, w = self.left.shape
        if self.support.shape != (b, 1, h, w) or self.support.dtype != torch.bool:
            raise ValueError("support must be bool [B,1,H,W]")
        if self.support.device != self.left.device or self.camera.K_L.device != self.left.device:
            raise ValueError("camera/support must share image device")
        if self.camera.K_L.shape[0] != b:
            raise ValueError("camera batch mismatch")


def deployment_from_stereo(
    sample: StereoBatch, *, rectified: bool, sensor_support: torch.Tensor | None = None
) -> DeploymentStereoBatch:
    """Allowlist images/calibration only; ignore ALL sample GT, masks and metadata.

    Default all-true support is appropriate only for fully supported images.
    Rectification padding requires an explicit sensor_support supplied by the
    preprocessing caller. Its provenance cannot be inferred from tensor values.
    """
    camera = RectifiedStereoCamera.from_cameras(
        sample.left_camera, sample.right_camera, rectified=rectified
    )
    b, _, h, w = sample.left.shape
    support = sensor_support
    if support is None:
        support = torch.ones((b, 1, h, w), dtype=torch.bool, device=sample.left.device)
    return DeploymentStereoBatch(sample.left, sample.right, camera, support)
