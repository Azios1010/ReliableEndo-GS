"""External dataset identity, resolution, validation, and split contracts."""

from reliable_endo_gs.data.base import DatasetAdapter
from reliable_endo_gs.data.paths import DATA_ROOT_ENV_VAR, resolve_data_root
from reliable_endo_gs.data.registry import get_dataset_adapter, list_dataset_names
from reliable_endo_gs.data.schema import DatasetConfig, hash_dataset_config, load_dataset_config
from reliable_endo_gs.data.splits import SplitManifest, hash_split_manifest, load_split_manifest
from reliable_endo_gs.data.validation import DatasetValidationReport

__all__ = [
    "DATA_ROOT_ENV_VAR",
    "DatasetAdapter",
    "DatasetConfig",
    "DatasetValidationReport",
    "SplitManifest",
    "get_dataset_adapter",
    "hash_dataset_config",
    "hash_split_manifest",
    "list_dataset_names",
    "load_dataset_config",
    "load_split_manifest",
    "resolve_data_root",
]
