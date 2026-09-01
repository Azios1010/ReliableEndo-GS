"""Isolated integration boundary for the pinned Endo-E2E-GS baseline."""

from reliable_endo_gs.baseline.adapter import (
    EndoE2EGSAdapter,
    TensorParity,
    compare_tensor_translation,
)
from reliable_endo_gs.baseline.config import BaselineConfig, BaselineConfigError
from reliable_endo_gs.baseline.native import (
    NativeBaselineRunner,
    NativeInferenceInput,
    NativeInferenceResult,
    NativeInputError,
    NativeParityCapture,
    NativeRunnerError,
)
from reliable_endo_gs.baseline.provenance import BaselineProvenance
from reliable_endo_gs.baseline.upstream import BaselineCapabilities, inspect_capabilities

__all__ = [
    "BaselineCapabilities",
    "BaselineConfig",
    "BaselineConfigError",
    "BaselineProvenance",
    "EndoE2EGSAdapter",
    "NativeBaselineRunner",
    "NativeInferenceInput",
    "NativeInferenceResult",
    "NativeInputError",
    "NativeParityCapture",
    "NativeRunnerError",
    "TensorParity",
    "compare_tensor_translation",
    "inspect_capabilities",
]
