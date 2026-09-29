"""Lazy candidate descriptors for FoundationStereo and DEFOM-Stereo.

These factories intentionally do not import either external project.  Their
callers inject a loader and API adapter after checking the pinned release,
checkpoint access, license and preprocessing contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from reliable_endo_gs.teachers.base import (
    AvailabilityProbe,
    LazyTeacherAdapter,
    TeacherIdentity,
    TeacherInference,
    TeacherLoader,
    TeacherOutputConverter,
    TeacherPreprocess,
)


def _identity(
    *,
    candidate_id: str,
    variant: str,
    api_version: str,
    checkpoint_id: str,
    checkpoint_sha256: str | None,
    license_name: str,
    preprocessing: Mapping[str, object] | None,
    output: Mapping[str, object] | None,
    source: str,
    config_hash: str | None,
) -> TeacherIdentity:
    return TeacherIdentity(
        candidate_id=candidate_id,
        variant=variant,
        api_version=api_version,
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        license=license_name,
        preprocessing=(
            dict(preprocessing)
            if preprocessing is not None
            else {
                "status": "unverified",
                "image_range": "candidate-specific; caller must declare",
                "rectification": "A00_rectified_input_required",
            }
        ),
        output=(
            dict(output)
            if output is not None
            else {
                "status": "unverified",
                "disparity_units": "teacher_grid_pixels",
                "disparity_sign": "u_left_minus_right",
                "confidence": "optional_predicted_only",
            }
        ),
        source=source,
        config_hash=config_hash,
    )


def make_foundation_stereo_candidate(
    *,
    loader: TeacherLoader | None = None,
    infer_fn: TeacherInference | None = None,
    availability_probe: AvailabilityProbe | None = None,
    variant: str = "primary_offline",
    api_version: str = "unverified_external_api",
    checkpoint_id: str = "unconfigured",
    checkpoint_sha256: str | None = None,
    license_name: str = "unverified_external_license",
    preprocessing: Mapping[str, object] | None = None,
    output: Mapping[str, object] | None = None,
    source: str = "FoundationStereo external repository; verify before use",
    config_hash: str | None = None,
    preprocess_fn: TeacherPreprocess | None = None,
    output_converter: TeacherOutputConverter | None = None,
) -> LazyTeacherAdapter:
    """Create a lazy FoundationStereo candidate descriptor.

    FoundationStereo is the primary strong offline candidate by protocol, not
    a selected winner.  ``loader`` and ``infer_fn`` are optional so an absent
    external dependency is represented explicitly instead of imported at
    package load time.
    """

    return LazyTeacherAdapter(
        identity=_identity(
            candidate_id="FoundationStereo",
            variant=variant,
            api_version=api_version,
            checkpoint_id=checkpoint_id,
            checkpoint_sha256=checkpoint_sha256,
            license_name=license_name,
            preprocessing=preprocessing,
            output=output,
            source=source,
            config_hash=config_hash,
        ),
        loader=loader,
        infer_fn=infer_fn,
        availability_probe=availability_probe,
        preprocess_fn=preprocess_fn,
        output_converter=output_converter,
    )


def make_defom_stereo_candidate(
    *,
    loader: TeacherLoader | None = None,
    infer_fn: TeacherInference | None = None,
    availability_probe: AvailabilityProbe | None = None,
    variant: str = "independent_comparator",
    api_version: str = "unverified_external_api",
    checkpoint_id: str = "unconfigured",
    checkpoint_sha256: str | None = None,
    license_name: str = "unverified_external_license",
    preprocessing: Mapping[str, object] | None = None,
    output: Mapping[str, object] | None = None,
    source: str = "DEFOM-Stereo external repository; verify before use",
    config_hash: str | None = None,
    preprocess_fn: TeacherPreprocess | None = None,
    output_converter: TeacherOutputConverter | None = None,
) -> LazyTeacherAdapter:
    """Create a lazy DEFOM-Stereo comparator descriptor.

    DEFOM-Stereo remains an independently identified comparator.  Availability
    or quality of this candidate is never inferred from FoundationStereo.
    """

    return LazyTeacherAdapter(
        identity=_identity(
            candidate_id="DEFOM-Stereo",
            variant=variant,
            api_version=api_version,
            checkpoint_id=checkpoint_id,
            checkpoint_sha256=checkpoint_sha256,
            license_name=license_name,
            preprocessing=preprocessing,
            output=output,
            source=source,
            config_hash=config_hash,
        ),
        loader=loader,
        infer_fn=infer_fn,
        availability_probe=availability_probe,
        preprocess_fn=preprocess_fn,
        output_converter=output_converter,
    )


# Short aliases keep call sites readable while retaining the explicit factory names.
foundation_stereo_candidate = make_foundation_stereo_candidate
defom_stereo_candidate = make_defom_stereo_candidate


def default_teacher_candidates(
    *,
    foundation_loader: TeacherLoader | None = None,
    foundation_infer_fn: TeacherInference | None = None,
    defom_loader: TeacherLoader | None = None,
    defom_infer_fn: TeacherInference | None = None,
    foundation_config_hash: str | None = None,
    defom_config_hash: str | None = None,
) -> tuple[LazyTeacherAdapter, LazyTeacherAdapter]:
    """Return both protocol candidates in stable order, without selecting one."""

    return (
        make_foundation_stereo_candidate(
            loader=foundation_loader,
            infer_fn=foundation_infer_fn,
            config_hash=foundation_config_hash,
        ),
        make_defom_stereo_candidate(
            loader=defom_loader,
            infer_fn=defom_infer_fn,
            config_hash=defom_config_hash,
        ),
    )


def sha256_file(path: Path) -> str:
    """Hash an injected checkpoint without opening it during module import."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_config_hash(config: Mapping[str, object]) -> str:
    """Return a stable hash for preprocessing/API configuration."""

    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "default_teacher_candidates",
    "defom_stereo_candidate",
    "foundation_stereo_candidate",
    "make_defom_stereo_candidate",
    "make_foundation_stereo_candidate",
    "sha256_file",
    "teacher_config_hash",
]
