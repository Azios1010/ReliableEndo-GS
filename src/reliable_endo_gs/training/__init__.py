"""Training namespace with lazy compatibility exports for historical workflows.

Importing this package does not load baseline trainers or optional Phase-I code.
New training workflows should import their concrete modules explicitly.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "PHASE1_CHECKPOINT_SCHEMA": ("reliable_endo_gs.training.phase1", "CHECKPOINT_SCHEMA"),
    "BaselineOutput": ("reliable_endo_gs.training.phase1", "BaselineOutput"),
    "BaselinePredictor": ("reliable_endo_gs.training.phase1", "BaselinePredictor"),
    "Phase1CheckpointMetadata": ("reliable_endo_gs.training.phase1", "Phase1CheckpointMetadata"),
    "Phase1Config": ("reliable_endo_gs.training.phase1", "Phase1Config"),
    "Phase1LogRecord": ("reliable_endo_gs.training.phase1", "Phase1LogRecord"),
    "Phase1StepResult": ("reliable_endo_gs.training.phase1", "Phase1StepResult"),
    "Phase1Trainer": ("reliable_endo_gs.training.phase1", "Phase1Trainer"),
    "PhaseIForward": ("reliable_endo_gs.training.phase1", "PhaseIForward"),
    "PhaseIForwardResult": ("reliable_endo_gs.training.phase1", "PhaseIForwardResult"),
    "SigmaProvider": ("reliable_endo_gs.training.phase1", "SigmaProvider"),
    "load_phase1_checkpoint": ("reliable_endo_gs.training.phase1", "load_phase1_checkpoint"),
    "save_phase1_checkpoint": ("reliable_endo_gs.training.phase1", "save_phase1_checkpoint"),
    "Stage1ScaredCCachedDataset": (
        "reliable_endo_gs.baseline.training.stage1_scared_c",
        "Stage1ScaredCCachedDataset",
    ),
    "Stage1ScaredCConfig": (
        "reliable_endo_gs.baseline.training.stage1_scared_c",
        "Stage1ScaredCConfig",
    ),
    "Stage1ScaredCConfigError": (
        "reliable_endo_gs.baseline.training.stage1_scared_c",
        "Stage1ScaredCConfigError",
    ),
    "Stage1SmokeResult": (
        "reliable_endo_gs.baseline.training.stage1_scared_c",
        "Stage1SmokeResult",
    ),
    "Stage1TrainingResult": (
        "reliable_endo_gs.baseline.training.stage1_scared_c",
        "Stage1TrainingResult",
    ),
    "benchmark_stage1_cache_data": (
        "reliable_endo_gs.baseline.training.stage1_scared_c",
        "benchmark_stage1_cache_data",
    ),
    "build_stage1_scratch_model": (
        "reliable_endo_gs.baseline.training.stage1_scared_c",
        "build_stage1_scratch_model",
    ),
    "load_stage1_scared_c_config": (
        "reliable_endo_gs.baseline.training.stage1_scared_c",
        "load_stage1_scared_c_config",
    ),
    "prepare_stage1_full_data": (
        "reliable_endo_gs.baseline.training.stage1_scared_c",
        "prepare_stage1_full_data",
    ),
    "prepare_stage1_full_split": (
        "reliable_endo_gs.baseline.training.stage1_scared_c",
        "prepare_stage1_full_split",
    ),
    "run_stage1_scratch_smoke": (
        "reliable_endo_gs.baseline.training.stage1_scared_c",
        "run_stage1_scratch_smoke",
    ),
    "run_stage1_scratch_training": (
        "reliable_endo_gs.baseline.training.stage1_scared_c",
        "run_stage1_scratch_training",
    ),
    "stage1_collate": ("reliable_endo_gs.baseline.training.stage1_scared_c", "stage1_collate"),
    "CheckpointMetadata": ("reliable_endo_gs.training.uncertainty", "CheckpointMetadata"),
    "TrainingStepResult": ("reliable_endo_gs.training.uncertainty", "TrainingStepResult"),
    "evaluate_uncertainty_head": (
        "reliable_endo_gs.training.uncertainty",
        "evaluate_uncertainty_head",
    ),
    "load_uncertainty_checkpoint": (
        "reliable_endo_gs.training.uncertainty",
        "load_uncertainty_checkpoint",
    ),
    "save_uncertainty_checkpoint": (
        "reliable_endo_gs.training.uncertainty",
        "save_uncertainty_checkpoint",
    ),
    "train_uncertainty_step": ("reliable_endo_gs.training.uncertainty", "train_uncertainty_step"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a historical export only when explicitly requested."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    return getattr(import_module(module), attribute)
