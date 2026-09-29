"""Teacher disparity alignment to the A00 rectified input-grid contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
import torch.nn.functional as functional

from reliable_endo_gs.teachers.base import TeacherContractError, TeacherPrediction


def _canonicalize(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"metadata value {type(value).__name__} is not JSON compatible")


def _check_hw(value: tuple[int, int], name: str) -> None:
    if len(value) != 2 or any(isinstance(item, bool) or item < 1 for item in value):
        raise TeacherContractError(f"{name} must be a positive (height, width) pair")


@dataclass(frozen=True, slots=True)
class RectifiedAlignment:
    """Describe the model-grid to A00 input-grid transform.

    ``input_hw`` is the full A00 rectified left/right grid.  The teacher sees a
    crop of ``crop_hw`` resized to ``teacher_hw``.  Crop coordinates are in
    input-grid pixels and may differ horizontally between left and right
    images.  Raw disparity is assumed to be ``u_left - u_right`` in teacher
    pixels; the conversion rescales it and applies the crop-origin offset once.
    """

    input_hw: tuple[int, int]
    teacher_hw: tuple[int, int]
    crop_hw: tuple[int, int] | None = None
    left_crop_xy: tuple[int, int] = (0, 0)
    right_crop_xy: tuple[int, int] = (0, 0)
    image_transform_id: str = "unspecified_image_transform"
    calibration_transform_id: str = "unspecified_calibration_transform"
    interpolation: str = "bilinear"
    align_corners: bool = False

    def __post_init__(self) -> None:
        _check_hw(self.input_hw, "input_hw")
        _check_hw(self.teacher_hw, "teacher_hw")
        crop_hw = self.crop_hw if self.crop_hw is not None else self.input_hw
        _check_hw(crop_hw, "crop_hw")
        object.__setattr__(self, "crop_hw", crop_hw)
        for name, value in (
            ("left_crop_xy", self.left_crop_xy),
            ("right_crop_xy", self.right_crop_xy),
        ):
            if len(value) != 2 or any(isinstance(item, bool) or item < 0 for item in value):
                raise TeacherContractError(f"{name} must be a nonnegative (x, y) pair")
            if value[1] + crop_hw[0] > self.input_hw[0] or value[0] + crop_hw[1] > self.input_hw[1]:
                raise TeacherContractError(f"{name} and crop_hw exceed input_hw")
        if self.left_crop_xy[1] != self.right_crop_xy[1]:
            raise TeacherContractError("left/right vertical crop origins must match")
        if self.interpolation != "bilinear":
            raise TeacherContractError("A01 currently supports only bilinear image alignment")
        if self.align_corners:
            raise TeacherContractError("A00 alignment requires align_corners=False")
        for name, xform_value in (
            ("image_transform_id", self.image_transform_id),
            ("calibration_transform_id", self.calibration_transform_id),
        ):
            if not isinstance(xform_value, str) or not xform_value.strip():
                raise TeacherContractError(f"{name} must be a non-empty identity")

    @property
    def crop_scale_xy(self) -> tuple[float, float]:
        """Return ``(sx, sy)`` from crop-grid pixels to teacher-grid pixels."""

        assert self.crop_hw is not None
        return (
            self.teacher_hw[1] / self.crop_hw[1],
            self.teacher_hw[0] / self.crop_hw[0],
        )

    def to_metadata(self) -> dict[str, object]:
        """Return JSON metadata sufficient to audit the alignment."""

        assert self.crop_hw is not None
        sx, sy = self.crop_scale_xy
        return {
            "alignment_schema": "a00_rectified_input_grid_v1",
            "input_hw": list(self.input_hw),
            "teacher_hw": list(self.teacher_hw),
            "crop_hw": list(self.crop_hw),
            "left_crop_xy": list(self.left_crop_xy),
            "right_crop_xy": list(self.right_crop_xy),
            "scale_xy": [sx, sy],
            "image_transform_id": self.image_transform_id,
            "calibration_transform_id": self.calibration_transform_id,
            "interpolation": self.interpolation,
            "align_corners": self.align_corners,
            "disparity_sign": "u_left_minus_right",
            "source_disparity_units": "teacher_grid_pixels",
            "target_disparity_units": "a00_input_grid_pixels",
        }


@dataclass(frozen=True, slots=True)
class AlignedTeacherTarget:
    """A01 target on the A00 grid, with validity/support kept separate.

    Tensor shapes are ``[B,1,H_input,W_input]``.  ``support`` describes the
    image/preprocessing support; ``validity`` describes usable teacher
    disparity.  ``predicted_confidence`` is optional and explicitly distinct
    from ``calibrated_risk``.  A01 never creates calibrated risk, so that field
    is always ``None`` unless a future explicit calibration stage is added.
    """

    disparity: torch.Tensor
    validity: torch.Tensor
    support: torch.Tensor
    metadata: Mapping[str, object] = field(default_factory=dict)
    predicted_confidence: torch.Tensor | None = None
    calibrated_risk: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.disparity.ndim != 4 or self.disparity.shape[1] != 1:
            raise TeacherContractError("aligned disparity must have shape [B,1,H,W]")
        for value, name in ((self.validity, "validity"), (self.support, "support")):
            if value.shape != self.disparity.shape or value.dtype != torch.bool:
                raise TeacherContractError(f"{name} must be bool with disparity shape")
        if self.predicted_confidence is not None and (
            self.predicted_confidence.shape != self.disparity.shape
            or not self.predicted_confidence.is_floating_point()
        ):
            raise TeacherContractError("predicted_confidence must match disparity shape")
        if self.calibrated_risk is not None and (
            self.calibrated_risk.shape != self.disparity.shape
            or not self.calibrated_risk.is_floating_point()
        ):
            raise TeacherContractError("calibrated_risk must match disparity shape")
        if (
            self.disparity.device != self.validity.device
            or self.disparity.device != self.support.device
        ):
            raise TeacherContractError("aligned target tensors must share device")
        _canonicalize(self.metadata)

    @property
    def valid_mask(self) -> torch.Tensor:
        """Alias used by downstream cache/geometry callers."""

        return self.validity

    @property
    def supported_validity(self) -> torch.Tensor:
        """Return the intersection without conflating the source semantics."""

        return self.validity & self.support

    def cache_metadata(self) -> dict[str, object]:
        """Return only initial-schema metadata; confidence maps are excluded."""

        metadata = dict(self.metadata)
        metadata["predicted_confidence_present"] = self.predicted_confidence is not None
        metadata["calibrated_risk_present"] = self.calibrated_risk is not None
        metadata["confidence_semantics"] = "predicted_confidence_not_calibrated_risk"
        return metadata


def _resize_map(value: torch.Tensor, size: tuple[int, int], *, mode: str) -> torch.Tensor:
    source = value.to(dtype=torch.float32) if not value.is_floating_point() else value
    if mode == "nearest":
        return functional.interpolate(source, size=size, mode=mode)
    return functional.interpolate(source, size=size, mode=mode, align_corners=False)


def _embed(
    crop: torch.Tensor,
    *,
    input_hw: tuple[int, int],
    origin_xy: tuple[int, int],
    fill_value: float | bool,
) -> torch.Tensor:
    result = torch.full(
        (crop.shape[0], crop.shape[1], input_hw[0], input_hw[1]),
        fill_value,
        dtype=crop.dtype,
        device=crop.device,
    )
    x, y = origin_xy
    result[..., y : y + crop.shape[-2], x : x + crop.shape[-1]] = crop
    return result


def align_teacher_prediction(
    prediction: TeacherPrediction,
    alignment: RectifiedAlignment,
    *,
    input_support: torch.Tensor | None = None,
    calibration_metadata: Mapping[str, object] | None = None,
) -> AlignedTeacherTarget:
    """Convert one raw teacher output into the A00 rectified input grid.

    The x-disparity correction is explicit:

    ``d_input = d_teacher / sx + left_crop_x - right_crop_x``.

    Validity and support use nearest-neighbour masks so invalid pixels cannot
    bleed into valid neighbours.  Calibration metadata is recorded, never
    inferred from teacher output, and no confidence signal is called risk.
    """

    if tuple(prediction.disparity.shape[-2:]) != alignment.teacher_hw:
        raise TeacherContractError(
            "teacher disparity spatial shape does not match alignment.teacher_hw"
        )
    if prediction.metadata.get("disparity_units") != "teacher_grid_pixels":
        raise TeacherContractError(
            "teacher disparity units must be explicitly declared as teacher_grid_pixels"
        )
    if prediction.metadata.get("disparity_sign") != "u_left_minus_right":
        raise TeacherContractError(
            "teacher disparity sign must be explicitly declared as u_left_minus_right"
        )
    batch = prediction.disparity.shape[0]
    if input_support is None:
        input_support = torch.ones(
            (batch, 1, alignment.input_hw[0], alignment.input_hw[1]),
            dtype=torch.bool,
            device=prediction.disparity.device,
        )
    elif (
        input_support.shape != (batch, 1, alignment.input_hw[0], alignment.input_hw[1])
        or input_support.dtype != torch.bool
        or input_support.device != prediction.disparity.device
    ):
        raise TeacherContractError("input_support must be bool [B,1,H_input,W_input]")

    assert alignment.crop_hw is not None
    crop_h, crop_w = alignment.crop_hw
    sx, _ = alignment.crop_scale_xy
    raw = prediction.disparity.detach()
    resized = _resize_map(raw, (crop_h, crop_w), mode="bilinear") / sx
    resized = resized + float(alignment.left_crop_xy[0] - alignment.right_crop_xy[0])

    finite = torch.isfinite(raw)
    source_valid = finite & (raw > 0)
    if prediction.validity is not None:
        source_valid &= prediction.validity
    source_support = (
        prediction.support
        if prediction.support is not None
        else torch.ones_like(source_valid, dtype=torch.bool)
    )
    valid_crop = _resize_map(source_valid, (crop_h, crop_w), mode="nearest") > 0.5
    support_crop = _resize_map(source_support, (crop_h, crop_w), mode="nearest") > 0.5

    x, y = alignment.left_crop_xy
    input_support_crop = input_support[..., y : y + crop_h, x : x + crop_w]
    support_crop &= input_support_crop

    disparity = _embed(
        resized,
        input_hw=alignment.input_hw,
        origin_xy=alignment.left_crop_xy,
        fill_value=float("nan"),
    )
    validity = _embed(
        valid_crop,
        input_hw=alignment.input_hw,
        origin_xy=alignment.left_crop_xy,
        fill_value=False,
    )
    support = _embed(
        support_crop,
        input_hw=alignment.input_hw,
        origin_xy=alignment.left_crop_xy,
        fill_value=False,
    )

    confidence: torch.Tensor | None = None
    if prediction.predicted_confidence is not None:
        confidence_crop = _resize_map(
            prediction.predicted_confidence.detach(), (crop_h, crop_w), mode="bilinear"
        )
        confidence = _embed(
            confidence_crop,
            input_hw=alignment.input_hw,
            origin_xy=alignment.left_crop_xy,
            fill_value=float("nan"),
        )

    metadata: dict[str, object] = dict(prediction.metadata)
    metadata.update(alignment.to_metadata())
    metadata["support_semantics"] = "preprocessing_or_sensor_support"
    metadata["validity_semantics"] = "teacher_disparity_finite_positive_and_declared_validity"
    metadata["predicted_confidence_present"] = confidence is not None
    metadata["calibrated_risk_present"] = False
    metadata["confidence_semantics"] = "predicted_confidence_not_calibrated_risk"
    metadata["calibration"] = {} if calibration_metadata is None else dict(calibration_metadata)
    _canonicalize(metadata)
    return AlignedTeacherTarget(
        disparity=disparity,
        validity=validity,
        support=support,
        metadata=metadata,
        predicted_confidence=confidence,
        calibrated_risk=None,
    )


# A descriptive alias for callers that prefer the conversion wording.
convert_teacher_disparity = align_teacher_prediction
align_teacher_disparity = align_teacher_prediction


__all__ = [
    "AlignedTeacherTarget",
    "RectifiedAlignment",
    "align_teacher_disparity",
    "align_teacher_prediction",
    "convert_teacher_disparity",
]
