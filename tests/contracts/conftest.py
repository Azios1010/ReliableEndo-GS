"""Tiny CPU fixtures for structural contract tests."""

import pytest
import torch

from reliable_endo_gs.contracts import (
    CameraBatch,
    GaussianField,
    ReconstructionState,
    RenderOutput,
    StereoBatch,
    StereoPrediction,
)

BATCH = 2
HEIGHT = 4
WIDTH = 5
GAUSSIANS = 6


@pytest.fixture
def camera_batch() -> CameraBatch:
    return CameraBatch(
        intrinsics=torch.eye(3).repeat(BATCH, 1, 1),
        world_from_camera=torch.eye(4).repeat(BATCH, 1, 1),
    )


@pytest.fixture
def stereo_batch(camera_batch: CameraBatch) -> StereoBatch:
    return StereoBatch(
        left=torch.zeros(BATCH, 3, HEIGHT, WIDTH),
        right=torch.ones(BATCH, 3, HEIGHT, WIDTH),
        left_camera=camera_batch,
        right_camera=camera_batch,
        sample_ids=["sample-0", "sample-1"],
        sequence_ids=["sequence-0", "sequence-1"],
        gt_disparity=torch.zeros(BATCH, 1, HEIGHT, WIDTH),
        gt_depth=torch.ones(BATCH, 1, HEIGHT, WIDTH),
        masks={"valid": torch.ones(BATCH, 1, HEIGHT, WIDTH, dtype=torch.bool)},
    )


@pytest.fixture
def stereo_prediction() -> StereoPrediction:
    return StereoPrediction(
        disparity=torch.ones(BATCH, 1, HEIGHT, WIDTH),
        valid_mask=torch.ones(BATCH, 1, HEIGHT, WIDTH, dtype=torch.bool),
        disparity_iterations=torch.ones(BATCH, 3, HEIGHT, WIDTH),
        sigma_d=torch.full((BATCH, 1, HEIGHT, WIDTH), 0.25),
        diagnostics={"source": "synthetic"},
    )


@pytest.fixture
def gaussian_field() -> GaussianField:
    return GaussianField(
        means3d=torch.zeros(BATCH, GAUSSIANS, 3),
        colors=torch.zeros(BATCH, GAUSSIANS, 3),
        rotations=torch.zeros(BATCH, GAUSSIANS, 4),
        scales=torch.ones(BATCH, GAUSSIANS, 3),
        opacities=torch.ones(BATCH, GAUSSIANS, 1),
        valid_mask=torch.ones(BATCH, GAUSSIANS, dtype=torch.bool),
    )


@pytest.fixture
def render_output() -> RenderOutput:
    return RenderOutput(
        image=torch.zeros(BATCH, 3, HEIGHT, WIDTH),
        depth=torch.ones(BATCH, 1, HEIGHT, WIDTH),
        visibility=torch.ones(BATCH, GAUSSIANS),
    )


@pytest.fixture
def reconstruction_state(
    stereo_batch: StereoBatch,
    stereo_prediction: StereoPrediction,
    gaussian_field: GaussianField,
    render_output: RenderOutput,
) -> ReconstructionState:
    return ReconstructionState(
        batch=stereo_batch,
        stereo=stereo_prediction,
        gaussians=gaussian_field,
        render_left=render_output,
        diagnostics={"valid_pixels": BATCH * HEIGHT * WIDTH},
        provenance={"artifact_id": "synthetic"},
    )
