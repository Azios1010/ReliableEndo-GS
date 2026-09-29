"""A01 teacher candidates, target alignment and external cache contracts."""

from importlib import import_module
from typing import Any

from reliable_endo_gs.teachers.base import (
    CandidateAvailability,
    LazyTeacherAdapter,
    Teacher,
    TeacherContractError,
    TeacherError,
    TeacherIdentity,
    TeacherPrediction,
    TeacherUnavailableError,
    coerce_teacher_prediction,
    convert_teacher_output_to_a00,
)
from reliable_endo_gs.teachers.cache import (
    DEFAULT_TARGET_SCHEMA,
    TargetCache,
    TargetCacheError,
    TargetCacheKey,
    TargetCacheResult,
    TargetSchema,
)
from reliable_endo_gs.teachers.candidates import (
    default_teacher_candidates,
    defom_stereo_candidate,
    foundation_stereo_candidate,
    make_defom_stereo_candidate,
    make_foundation_stereo_candidate,
)
from reliable_endo_gs.teachers.roles import (
    A01_CACHE_ROLES,
    A01_DATASET_IDS,
    A01_DATASET_NAME,
    A01_FINAL_DATASET_IDS,
    DEFAULT_A01_SCARED_ROLES,
    A01RoleError,
    A01ScaredDataRoles,
    load_a01_roles,
)
from reliable_endo_gs.teachers.targets import (
    AlignedTeacherTarget,
    RectifiedAlignment,
    align_teacher_disparity,
    align_teacher_prediction,
    convert_teacher_disparity,
)
_WORKFLOW_EXPORTS = {
    name: ("reliable_endo_gs.teachers.workflow", name)
    for name in (
        "A01CacheReport",
        "A01TargetResult",
        "A01WorkflowError",
        "collect_one_target",
        "collect_targets",
        "infer_and_align",
        "load_teacher_factory",
        "target_cache_key",
        "validate_cache",
    )
}


def __getattr__(name: str) -> Any:
    target = _WORKFLOW_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "A01_CACHE_ROLES",
    "A01_DATASET_IDS",
    "A01_DATASET_NAME",
    "A01_FINAL_DATASET_IDS",
    "A01RoleError",
    "A01ScaredDataRoles",
    "AlignedTeacherTarget",
    "CandidateAvailability",
    "A01CacheReport",
    "A01TargetResult",
    "A01WorkflowError",
    "DEFAULT_A01_SCARED_ROLES",
    "DEFAULT_TARGET_SCHEMA",
    "LazyTeacherAdapter",
    "RectifiedAlignment",
    "TargetCache",
    "TargetCacheError",
    "TargetCacheKey",
    "TargetCacheResult",
    "TargetSchema",
    "Teacher",
    "TeacherContractError",
    "TeacherError",
    "TeacherIdentity",
    "TeacherPrediction",
    "TeacherUnavailableError",
    "align_teacher_disparity",
    "align_teacher_prediction",
    "coerce_teacher_prediction",
    "convert_teacher_disparity",
    "convert_teacher_output_to_a00",
    "default_teacher_candidates",
    "defom_stereo_candidate",
    "foundation_stereo_candidate",
    "load_a01_roles",
    "make_defom_stereo_candidate",
    "make_foundation_stereo_candidate",
    "collect_one_target",
    "collect_targets",
    "infer_and_align",
    "load_teacher_factory",
    "target_cache_key",
    "validate_cache",
]
