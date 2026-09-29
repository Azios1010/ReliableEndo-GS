"""External dataset identity, resolution, validation, and split contracts."""

from importlib import import_module
from typing import Any

from reliable_endo_gs.data.base import DatasetAdapter
from reliable_endo_gs.data.paths import DATA_ROOT_ENV_VAR, resolve_data_root
from reliable_endo_gs.data.registry import get_dataset_adapter, list_dataset_names
from reliable_endo_gs.data.scared_a00 import (
    A00Protocol,
    A01Manifest,
    ScaredA00Adapter,
    ScaredA00Dataset,
    ScaredA00ProtocolError,
    ScaredA00Sample,
    ScaredA00SampleError,
    ScaredSupervision,
    build_a01_manifest,
    collate_scared_a00,
    load_a01_manifest,
)
from reliable_endo_gs.data.scared_contract import (
    KeyframeValidationReport,
    validate_keyframe_processed_root,
)
from reliable_endo_gs.data.scared_manifest import (
    SCARED_FOUR_KEYFRAME_SPLIT,
    SCARED_FOUR_KEYFRAME_TRAIN,
    SCARED_FOUR_KEYFRAME_VALIDATION,
    load_scared_split_manifest,
    make_scared_sample_id,
    parse_keyframe_id,
    parse_scared_keyframe_id,
    validate_scared_split_manifest,
    write_scared_split_manifest,
)
from reliable_endo_gs.data.schema import DatasetConfig, hash_dataset_config, load_dataset_config
from reliable_endo_gs.data.splits import (
    SplitManifest,
    hash_split_manifest,
    load_split_manifest,
    validate_grouped_membership,
    validate_sequence_groups,
)
from reliable_endo_gs.data.validation import DatasetValidationReport

__all__ = [
    "DATA_ROOT_ENV_VAR",
    "A00Protocol",
    "A01Manifest",
    "DatasetAdapter",
    "DatasetConfig",
    "DatasetValidationReport",
    "DataDecodeError",
    "CORRECTED_VIDEO_FRAME",
    "CORRECTED_VIDEO_MODE",
    "FinalDatasetGuardError",
    "KeyframeValidationReport",
    "LazyScaredCDataset",
    "LazyScaredCIndex",
    "ScaredCRecord",
    "Stage1FrameSplit",
    "Stage1FrameSplitError",
    "ScaredCLazyDataset",
    "ScaredCRoleManifest",
    "ScaredCSample",
    "ScaredCSampleRecord",
    "ScaredKeyframeDataset",
    "ScaredA00Adapter",
    "ScaredA00Dataset",
    "ScaredA00ProtocolError",
    "ScaredA00Sample",
    "ScaredA00SampleError",
    "ScaredSupervision",
    "ScaredMultiSequenceDataset",
    "ScaredStage1Sample",
    "ScaredStage2Dataset",
    "ScaredStage2Sample",
    "STAGE1_FORBIDDEN_DATASET_IDS",
    "STAGE1_FULL_SPLIT_NAME",
    "STAGE1_TRAIN_DATASET_IDS",
    "ScaredUpstreamWrapper",
    "SampleIndex",
    "StereoCalibration",
    "SplitManifest",
    "SCARED_FOUR_KEYFRAME_SPLIT",
    "SCARED_FOUR_KEYFRAME_TRAIN",
    "SCARED_FOUR_KEYFRAME_VALIDATION",
    "STATIC_KEYFRAME",
    "STATIC_MODE",
    "DEFAULT_ZFAR",
    "DEFAULT_ZNEAR",
    "compute_camera_center",
    "compute_disp_const",
    "compute_fov",
    "compute_projection_matrix",
    "compute_safe_depth",
    "compute_world_view_transform",
    "stage1_to_stage2_sample",
    "stage2_batch_to_native_input",
    "stage2_collate_fn",
    "validate_stage2_sample",
    "get_dataset_adapter",
    "build_scared_c_index",
    "build_a01_manifest",
    "collate_scared_a00",
    "build_scared_c_lazy_index",
    "adjust_intrinsics_for_resize_and_crop",
    "crop_intrinsics",
    "hash_dataset_config",
    "hash_split_manifest",
    "list_dataset_names",
    "load_dataset_config",
    "load_a01_manifest",
    "load_scared_c_sample",
    "load_stage1_frame_split",
    "load_scared_c_role_manifest",
    "load_scared_split_manifest",
    "load_split_manifest",
    "make_sample_id",
    "make_scared_sample_id",
    "parse_keyframe_id",
    "parse_opencv_calibration",
    "parse_scared_c_stereo_calibration",
    "parse_scared_keyframe_id",
    "resize_intrinsics",
    "resolve_data_root",
    "resolve_scared_c_root",
    "materialize_scared_c_sample",
    "materialize_scared_c_rectified_rgb",
    "make_stage1_frame_split",
    "normalize_dataset_ids",
    "normalize_keyframe_entries",
    "normalize_sample_ids",
    "require_active_scared_c_role",
    "select_stage1_records",
    "validate_grouped_membership",
    "validate_keyframe_processed_root",
    "validate_scared_split_manifest",
    "validate_sequence_groups",
    "validate_stage1_sample",
    "write_scared_split_manifest",
    "write_stage1_frame_split",
]


_LEGACY_STAGE2_EXPORTS = frozenset(
    [
        "DEFAULT_ZFAR",
        "DEFAULT_ZNEAR",
        "ScaredStage2Dataset",
        "ScaredStage2Sample",
        "compute_camera_center",
        "compute_disp_const",
        "compute_fov",
        "compute_projection_matrix",
        "compute_safe_depth",
        "compute_world_view_transform",
        "stage1_to_stage2_sample",
        "stage2_batch_to_native_input",
        "stage2_collate_fn",
        "validate_stage2_sample",
    ]
)

_HISTORICAL_SCARED_C_EXPORTS = {
    name: ("reliable_endo_gs.data.scared_c", name)
    for name in (
        "CORRECTED_VIDEO_FRAME",
        "CORRECTED_VIDEO_MODE",
        "STATIC_KEYFRAME",
        "STATIC_MODE",
        "FinalDatasetGuardError",
        "LazyScaredCDataset",
        "ScaredCLazyDataset",
        "ScaredCRoleManifest",
        "ScaredCSample",
        "ScaredCSampleRecord",
        "load_scared_c_role_manifest",
        "materialize_scared_c_rectified_rgb",
        "materialize_scared_c_sample",
        "require_active_scared_c_role",
        "resolve_scared_c_root",
    )
}
_HISTORICAL_SCARED_C_EXPORTS.update(
    {
        "LazyScaredCIndex": ("reliable_endo_gs.data.scared_c", "ScaredCIndex"),
        "build_scared_c_lazy_index": (
            "reliable_endo_gs.data.scared_c",
            "build_scared_c_index",
        ),
    }
)
_HISTORICAL_STAGE1_EXPORTS = {
    name: ("reliable_endo_gs.data.scared_c_stage1_split", name)
    for name in (
        "STAGE1_FORBIDDEN_DATASET_IDS",
        "STAGE1_FULL_SPLIT_NAME",
        "STAGE1_TRAIN_DATASET_IDS",
        "Stage1FrameSplit",
        "Stage1FrameSplitError",
        "load_stage1_frame_split",
        "make_stage1_frame_split",
        "normalize_dataset_ids",
        "normalize_keyframe_entries",
        "normalize_sample_ids",
        "select_stage1_records",
        "write_stage1_frame_split",
    )
}

# These modules belong to the historical indexed/SCARED-C data path. Keep
# their public compatibility names, but do not import them while an active
# SCARED A00 consumer is being imported.
_HISTORICAL_DATA_EXPORTS = {
    name: ("reliable_endo_gs.data.calibration", name)
    for name in (
        "StereoCalibration",
        "adjust_intrinsics_for_resize_and_crop",
        "crop_intrinsics",
        "parse_opencv_calibration",
        "parse_scared_c_stereo_calibration",
        "resize_intrinsics",
    )
}
_HISTORICAL_DATA_EXPORTS.update(
    {
        name: ("reliable_endo_gs.data.index", name)
        for name in ("SampleIndex", "ScaredCRecord", "build_scared_c_index", "make_sample_id")
    }
)
_HISTORICAL_DATA_EXPORTS.update(
    {
        name: ("reliable_endo_gs.data.loaders", name)
        for name in ("DataDecodeError", "load_scared_c_sample")
    }
)
_HISTORICAL_DATA_EXPORTS.update(
    {
        name: ("reliable_endo_gs.data.scared_multi", name)
        for name in (
            "ScaredKeyframeDataset",
            "ScaredMultiSequenceDataset",
            "ScaredStage1Sample",
            "ScaredUpstreamWrapper",
            "validate_stage1_sample",
        )
    }
)


def __getattr__(name: str) -> Any:
    """Preserve explicit historical imports through lazy module boundaries."""

    historical = (
        _HISTORICAL_SCARED_C_EXPORTS.get(name)
        or _HISTORICAL_STAGE1_EXPORTS.get(name)
        or _HISTORICAL_DATA_EXPORTS.get(name)
    )
    if historical is not None:
        module_name, attribute = historical
        return getattr(import_module(module_name), attribute)

    if name not in _LEGACY_STAGE2_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module("reliable_endo_gs.baseline.data.stage2"), name)
