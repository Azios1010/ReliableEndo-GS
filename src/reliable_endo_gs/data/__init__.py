"""External dataset identity, resolution, validation, and split contracts."""

from reliable_endo_gs.data.base import DatasetAdapter
from reliable_endo_gs.data.calibration import (
    StereoCalibration,
    adjust_intrinsics_for_resize_and_crop,
    crop_intrinsics,
    parse_opencv_calibration,
    parse_scared_c_stereo_calibration,
    resize_intrinsics,
)
from reliable_endo_gs.data.index import (
    SampleIndex,
    ScaredCRecord,
    build_scared_c_index,
    make_sample_id,
)
from reliable_endo_gs.data.loaders import DataDecodeError, load_scared_c_sample
from reliable_endo_gs.data.paths import DATA_ROOT_ENV_VAR, resolve_data_root
from reliable_endo_gs.data.registry import get_dataset_adapter, list_dataset_names
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
    "DatasetAdapter",
    "DatasetConfig",
    "DatasetValidationReport",
    "DataDecodeError",
    "ScaredCRecord",
    "SampleIndex",
    "StereoCalibration",
    "SplitManifest",
    "get_dataset_adapter",
    "build_scared_c_index",
    "adjust_intrinsics_for_resize_and_crop",
    "crop_intrinsics",
    "hash_dataset_config",
    "hash_split_manifest",
    "list_dataset_names",
    "load_dataset_config",
    "load_scared_c_sample",
    "load_split_manifest",
    "make_sample_id",
    "parse_opencv_calibration",
    "parse_scared_c_stereo_calibration",
    "resize_intrinsics",
    "resolve_data_root",
    "validate_grouped_membership",
    "validate_sequence_groups",
]
