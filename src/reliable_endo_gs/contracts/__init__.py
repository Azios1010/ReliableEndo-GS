"""Stable PyTorch tensor contracts for future scientific components."""

from reliable_endo_gs.contracts.cameras import CameraBatch
from reliable_endo_gs.contracts.gaussians import GaussianField
from reliable_endo_gs.contracts.reconstruction import ReconstructionState
from reliable_endo_gs.contracts.rendering import RenderOutput
from reliable_endo_gs.contracts.samples import StereoBatch
from reliable_endo_gs.contracts.stereo import StereoPrediction

__all__ = [
    "CameraBatch",
    "GaussianField",
    "ReconstructionState",
    "RenderOutput",
    "StereoBatch",
    "StereoPrediction",
]
