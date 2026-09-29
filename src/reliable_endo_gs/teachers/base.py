"""Baseline-independent stereo teacher contracts and lazy adapters.

The module deliberately contains no candidate imports, checkpoint reads,
network access, CUDA setup, or model construction at import time.  Integrators
inject a loader and an inference callable when a candidate is explicitly
requested.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, TypeAlias

import torch


class TeacherError(RuntimeError):
    """Base error for teacher adapter failures."""


class TeacherUnavailableError(TeacherError):
    """Raised when an optional teacher cannot be loaded in the current environment."""


class TeacherContractError(ValueError):
    """Raised when an injected teacher does not satisfy the target contract."""


def _canonicalize(value: object) -> object:
    """Convert identity metadata into deterministic JSON-compatible values."""

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


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _canonicalize(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class TeacherIdentity:
    """Pinned identity and provenance for one candidate implementation.

    ``candidate_id`` identifies the model family, while API, checkpoint,
    license, preprocessing and output fields remain explicit even when a
    candidate is not yet configured.  A digest is used in cache keys; it does
    not imply that the candidate has been scientifically selected.
    """

    candidate_id: str
    variant: str
    api_version: str
    checkpoint_id: str
    checkpoint_sha256: str | None
    license: str
    preprocessing: Mapping[str, object]
    output: Mapping[str, object]
    source: str
    config_hash: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "variant",
            "api_version",
            "checkpoint_id",
            "license",
            "source",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TeacherContractError(f"teacher identity field {name!r} must be non-empty")
        if self.checkpoint_sha256 is not None and (
            not isinstance(self.checkpoint_sha256, str) or not self.checkpoint_sha256.strip()
        ):
            raise TeacherContractError("checkpoint_sha256 must be null or non-empty")
        if self.config_hash is not None and (
            not isinstance(self.config_hash, str)
            or len(self.config_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in self.config_hash)
        ):
            raise TeacherContractError("config_hash must be null or a 64-character digest")
        _canonical_json(self.preprocessing)
        _canonical_json(self.output)

    @property
    def candidate(self) -> str:
        """Compatibility-friendly name for the candidate family."""

        return self.candidate_id

    def to_dict(self) -> dict[str, object]:
        """Return the complete identity used for provenance and hashing."""

        return {
            "candidate_id": self.candidate_id,
            "variant": self.variant,
            "api_version": self.api_version,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "license": self.license,
            "preprocessing": _canonicalize(self.preprocessing),
            "output": _canonicalize(self.output),
            "source": self.source,
            "config_hash": self.config_hash,
        }

    @property
    def identity_hash(self) -> str:
        """Return a stable SHA-256 identity digest."""

        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()

    def runtime_validation_error(self) -> str | None:
        """Return an error when this identity is unsafe for live inference."""

        if self.api_version.startswith("unverified"):
            return "teacher API identity is unverified"
        if self.checkpoint_id == "unconfigured":
            return "teacher checkpoint identity is unconfigured"
        if (
            self.checkpoint_sha256 is None
            or len(self.checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdefABCDEF" for character in self.checkpoint_sha256
            )
        ):
            return "teacher checkpoint_sha256 must be a verified 64-character digest"
        if self.license.startswith("unverified"):
            return "teacher license identity is unverified"
        if self.preprocessing.get("status") == "unverified":
            return "teacher preprocessing identity is unverified"
        if self.output.get("status") == "unverified":
            return "teacher output identity is unverified"
        if self.output.get("disparity_units") != "teacher_grid_pixels":
            return "teacher output disparity units must be teacher_grid_pixels"
        if self.output.get("disparity_sign") != "u_left_minus_right":
            return "teacher output disparity sign must be u_left_minus_right"
        return None


@dataclass(frozen=True, slots=True)
class CandidateAvailability:
    """Report candidate availability without claiming a performance ranking."""

    candidate_id: str
    available: bool
    reason: str | None = None
    checked: bool = False

    @property
    def status(self) -> Literal["available", "configured", "unavailable"]:
        if not self.available:
            return "unavailable"
        return "available" if self.checked else "configured"

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "available": self.available,
            "reason": self.reason,
            "checked": self.checked,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class TeacherPrediction:
    """Raw output from one teacher in its declared model grid.

    Disparity and optional maps have shape ``[B, 1, H_teacher, W_teacher]``.
    ``predicted_confidence`` is retained as a distinct optional signal; it is
    never silently re-labelled as calibrated risk.
    """

    disparity: torch.Tensor
    validity: torch.Tensor | None = None
    support: torch.Tensor | None = None
    predicted_confidence: torch.Tensor | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_map(self.disparity, "disparity", floating=True)
        for value, name in (
            (self.validity, "validity"),
            (self.support, "support"),
        ):
            if value is not None:
                _validate_map(value, name, floating=False, expected_shape=self.disparity.shape)
        if self.predicted_confidence is not None:
            _validate_map(
                self.predicted_confidence,
                "predicted_confidence",
                floating=True,
                expected_shape=self.disparity.shape,
            )
        _canonical_json(self.metadata)


def _validate_map(
    value: torch.Tensor,
    name: str,
    *,
    floating: bool,
    expected_shape: torch.Size | tuple[int, ...] | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TeacherContractError(f"{name} must be a torch.Tensor")
    if value.ndim != 4 or value.shape[1] != 1 or min(value.shape) < 1:
        raise TeacherContractError(f"{name} must have shape [B,1,H,W]")
    if expected_shape is not None and value.shape != expected_shape:
        raise TeacherContractError(f"{name} shape must match disparity shape")
    if floating and not value.is_floating_point():
        raise TeacherContractError(f"{name} must be floating point")
    if not floating and value.dtype != torch.bool:
        raise TeacherContractError(f"{name} must be boolean")


TeacherLoader: TypeAlias = Callable[[], object]
TeacherInference: TypeAlias = Callable[[object, torch.Tensor, torch.Tensor], object]
TeacherPreprocess: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]
TeacherOutputConverter: TypeAlias = Callable[[object], object]
AvailabilityProbe: TypeAlias = Callable[[], bool | str | CandidateAvailability]


class Teacher(Protocol):
    """Narrow explicit inference protocol consumed by later target code."""

    @property
    def identity(self) -> TeacherIdentity:
        """Return immutable candidate provenance."""

    def availability(self) -> CandidateAvailability:
        """Report whether the optional candidate can be used."""

    def infer(self, left: torch.Tensor, right: torch.Tensor) -> TeacherPrediction:
        """Infer raw disparity under an explicit no-gradient boundary."""


def _freeze_and_eval(model: object) -> None:
    """Put an injected model in eval/frozen mode without assuming a framework."""

    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()
    parameters_method = getattr(model, "parameters", None)
    if callable(parameters_method):
        for parameter in parameters_method():
            if isinstance(parameter, torch.Tensor):
                parameter.requires_grad_(False)


@dataclass(slots=True)
class LazyTeacherAdapter:
    """Lazy, frozen adapter around an injected model loader and API callable.

    Construction performs no model import, checkpoint access, download, device
    selection or CUDA initialization.  The loader is called only by ``load``
    or ``infer``.  The injected inference callable receives ``(model, left,
    right)`` and may return a tensor, a mapping, or ``TeacherPrediction``.
    """

    identity: TeacherIdentity
    loader: TeacherLoader | None
    infer_fn: TeacherInference | None
    availability_probe: AvailabilityProbe | None = None
    preprocess_fn: TeacherPreprocess | None = None
    output_converter: TeacherOutputConverter | None = None
    _model: object | None = field(init=False, default=None, repr=False)
    _availability: CandidateAvailability | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.loader is None and self.infer_fn is not None:
            raise TeacherContractError("infer_fn cannot be set without a lazy loader")
        if self.loader is not None and self.infer_fn is None:
            raise TeacherContractError("an infer_fn is required for an injected loader")

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def availability(self) -> CandidateAvailability:
        """Probe optional dependencies without loading a checkpoint by default."""

        if self._availability is not None:
            return self._availability
        if self.loader is None:
            self._availability = CandidateAvailability(
                self.identity.candidate_id,
                available=False,
                reason="no lazy loader was injected",
                checked=True,
            )
            return self._availability
        if self.availability_probe is None:
            self._availability = CandidateAvailability(
                self.identity.candidate_id,
                available=True,
                reason="loader configured; dependency/checkpoint not loaded",
                checked=False,
            )
            return self._availability
        try:
            result = self.availability_probe()
        except Exception as error:  # noqa: BLE001 - probe must become a report
            self._availability = CandidateAvailability(
                self.identity.candidate_id,
                available=False,
                reason=f"availability probe failed: {error}",
                checked=True,
            )
            return self._availability
        if isinstance(result, CandidateAvailability):
            self._availability = result
        elif isinstance(result, bool):
            self._availability = CandidateAvailability(
                self.identity.candidate_id,
                available=result,
                reason=None if result else "availability probe reported unavailable",
                checked=True,
            )
        else:
            self._availability = CandidateAvailability(
                self.identity.candidate_id,
                available=False,
                reason=result,
                checked=True,
            )
        return self._availability

    def load(self) -> object:
        """Load, freeze and switch the candidate to evaluation mode explicitly."""

        if self._model is not None:
            return self._model
        if self.loader is None:
            raise TeacherUnavailableError(
                f"{self.identity.candidate_id} is unavailable: no lazy loader was injected"
            )
        try:
            model = self.loader()
        except Exception as error:  # noqa: BLE001 - preserve optional dependency failures
            self._availability = CandidateAvailability(
                self.identity.candidate_id,
                available=False,
                reason=f"loader failed: {error}",
                checked=True,
            )
            raise TeacherUnavailableError(
                f"unable to load {self.identity.candidate_id}: {error}"
            ) from error
        if model is None:
            raise TeacherUnavailableError(f"loader for {self.identity.candidate_id} returned None")
        _freeze_and_eval(model)
        self._model = model
        self._availability = CandidateAvailability(
            self.identity.candidate_id, available=True, checked=True
        )
        return model

    def freeze(self) -> None:
        """Load and re-assert frozen/evaluation state."""

        _freeze_and_eval(self.load())

    def infer(self, left: torch.Tensor, right: torch.Tensor) -> TeacherPrediction:
        """Run one explicit frozen inference under ``torch.inference_mode``."""

        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TeacherContractError("left and right must be torch.Tensor values")
        if left.ndim != 4 or left.shape[1] != 3:
            raise TeacherContractError("left must have shape [B,3,H,W]")
        if right.shape != left.shape:
            raise TeacherContractError("right must match left shape [B,3,H,W]")
        if not left.is_floating_point() or not right.is_floating_point():
            raise TeacherContractError("teacher RGB inputs must be floating point")
        if left.device != right.device or left.dtype != right.dtype:
            raise TeacherContractError("teacher RGB inputs must share dtype and device")
        model = self.load()
        if self.infer_fn is None:
            raise TeacherUnavailableError("teacher inference callable is not configured")
        identity_error = self.identity.runtime_validation_error()
        if identity_error is not None:
            raise TeacherContractError(
                f"{self.identity.candidate_id} cannot run with unverified identity: {identity_error}"
            )
        _freeze_and_eval(model)
        move_to = getattr(model, "to", None)
        if callable(move_to):
            try:
                move_to(left.device)
            except (RuntimeError, TypeError, ValueError) as error:
                raise TeacherContractError(
                    f"{self.identity.candidate_id} model cannot move to {left.device}: {error}"
                ) from error
        teacher_left, teacher_right = left, right
        if self.preprocess_fn is not None:
            try:
                teacher_left, teacher_right = self.preprocess_fn(left, right)
            except Exception as error:  # noqa: BLE001 - preserve adapter boundary
                raise TeacherContractError(
                    f"{self.identity.candidate_id} preprocessing failed: {error}"
                ) from error
            if (
                not isinstance(teacher_left, torch.Tensor)
                or not isinstance(teacher_right, torch.Tensor)
                or teacher_left.ndim != 4
                or teacher_right.shape != teacher_left.shape
                or teacher_left.shape[1] != 3
                or teacher_left.device != left.device
            ):
                raise TeacherContractError(
                    f"{self.identity.candidate_id} preprocessing must return paired [B,3,H,W] tensors on the input device"
                )
        with torch.inference_mode():
            raw = self.infer_fn(model, teacher_left, teacher_right)
        if self.output_converter is not None:
            try:
                raw = self.output_converter(raw)
            except Exception as error:  # noqa: BLE001 - preserve adapter boundary
                raise TeacherContractError(
                    f"{self.identity.candidate_id} output conversion failed: {error}"
                ) from error
        prediction = coerce_teacher_prediction(raw)
        if prediction.disparity.device != left.device:
            raise TeacherContractError(
                f"{self.identity.candidate_id} disparity output is on {prediction.disparity.device}; "
                f"expected {left.device}"
            )
        declared_units = self.identity.output.get("disparity_units")
        declared_sign = self.identity.output.get("disparity_sign")
        metadata = dict(prediction.metadata)
        for key, declared in (
            ("disparity_units", declared_units),
            ("disparity_sign", declared_sign),
        ):
            if metadata.get(key, declared) != declared:
                raise TeacherContractError(
                    f"teacher prediction {key} conflicts with its pinned output identity"
                )
            metadata[key] = declared
        return replace(prediction, metadata=metadata)


def convert_teacher_output_to_a00(
    value: object,
    *,
    disparity_units: str = "teacher_grid_pixels",
    disparity_sign: str = "u_left_minus_right",
    disparity_scale: float = 1.0,
) -> TeacherPrediction:
    """Convert a common upstream output shape/sign into the A00 convention.

    The conversion is explicit and metadata-driven.  A rank-three disparity is
    promoted to ``[B,1,H,W]``; a declared right-minus-left sign is negated; and
    an explicit pixel scale is applied once.  No range, unit, or sign is
    inferred from tensor values.
    """

    normalized_value = value
    if isinstance(value, torch.Tensor) and value.ndim == 3:
        normalized_value = value.unsqueeze(1)
    elif isinstance(value, Mapping) and isinstance(value.get("disparity"), torch.Tensor):
        raw_disparity = value["disparity"]
        if raw_disparity.ndim == 3:
            normalized_value = dict(value)
            normalized_value["disparity"] = raw_disparity.unsqueeze(1)
    prediction = coerce_teacher_prediction(normalized_value)
    disparity = prediction.disparity
    if disparity_units != "teacher_grid_pixels":
        raise TeacherContractError("teacher output conversion requires teacher_grid_pixels units")
    if disparity_sign not in ("u_left_minus_right", "u_right_minus_left"):
        raise TeacherContractError("teacher output sign must be explicitly declared")
    if not isinstance(disparity_scale, (float, int)) or isinstance(disparity_scale, bool):
        raise TeacherContractError("teacher disparity_scale must be numeric")
    if not torch.isfinite(torch.tensor(float(disparity_scale))) or float(disparity_scale) <= 0:
        raise TeacherContractError("teacher disparity_scale must be finite and positive")
    converted = disparity * float(disparity_scale)
    if disparity_sign == "u_right_minus_left":
        converted = -converted
    declared_validity = prediction.validity
    validity = (
        declared_validity
        if declared_validity is not None
        else torch.isfinite(converted) & (converted > 0)
    )
    declared_support = prediction.support
    support = declared_support if declared_support is not None else torch.isfinite(converted)
    metadata = dict(prediction.metadata)
    metadata.update(
        {
            "disparity_units": "teacher_grid_pixels",
            "disparity_sign": "u_left_minus_right",
            "conversion_scale": float(disparity_scale),
            "validity_semantics": "declared_or_finite_positive_disparity",
            "support_semantics": "declared_or_finite_teacher_output",
        }
    )
    return replace(
        prediction, disparity=converted, validity=validity, support=support, metadata=metadata
    )


def coerce_teacher_prediction(value: object) -> TeacherPrediction:
    """Normalize common injected API outputs without importing any candidate."""

    if isinstance(value, TeacherPrediction):
        return value
    if isinstance(value, torch.Tensor):
        return TeacherPrediction(value)
    if isinstance(value, Mapping):
        disparity = value.get("disparity")
        if not isinstance(disparity, torch.Tensor):
            raise TeacherContractError("teacher mapping output must contain tensor 'disparity'")
        validity = value.get("validity", value.get("valid_mask"))
        support = value.get("support", value.get("support_mask"))
        confidence = value.get("predicted_confidence", value.get("confidence"))
        if validity is not None and not isinstance(validity, torch.Tensor):
            raise TeacherContractError("teacher validity output must be a tensor")
        if support is not None and not isinstance(support, torch.Tensor):
            raise TeacherContractError("teacher support output must be a tensor")
        if confidence is not None and not isinstance(confidence, torch.Tensor):
            raise TeacherContractError("teacher confidence output must be a tensor")
        metadata_value = value.get("metadata", {})
        if not isinstance(metadata_value, Mapping):
            raise TeacherContractError("teacher metadata output must be a mapping")
        return TeacherPrediction(
            disparity,
            validity=validity,
            support=support,
            predicted_confidence=confidence,
            metadata=metadata_value,
        )
    if isinstance(value, (tuple, list)) and value:
        if len(value) > 2 or not isinstance(value[0], torch.Tensor):
            raise TeacherContractError(
                "tuple teacher output must be (disparity,) or (disparity, confidence)"
            )
        confidence = value[1] if len(value) == 2 else None
        if confidence is not None and not isinstance(confidence, torch.Tensor):
            raise TeacherContractError("tuple teacher confidence output must be a tensor")
        return TeacherPrediction(value[0], predicted_confidence=confidence)
    raise TeacherContractError(
        "teacher output must be a tensor, mapping, tuple/list, or TeacherPrediction"
    )


__all__ = [
    "AvailabilityProbe",
    "CandidateAvailability",
    "LazyTeacherAdapter",
    "Teacher",
    "TeacherContractError",
    "TeacherError",
    "TeacherIdentity",
    "TeacherInference",
    "TeacherOutputConverter",
    "TeacherPreprocess",
    "TeacherLoader",
    "TeacherPrediction",
    "TeacherUnavailableError",
    "coerce_teacher_prediction",
    "convert_teacher_output_to_a00",
]
