"""External rendering evaluation: row builder and the scale-free disparity agreement (synthetic, CPU)."""

import torch

from reliable_endo_gs.contracts import RenderOutput
from reliable_endo_gs.evaluation.external_runner import (
    disparity_affine_agreement,
    render_row_builder,
)
from reliable_endo_gs.evaluation.multi_model import Prediction


def test_affine_agreement_is_perfect_for_an_affine_function_of_inverse_depth() -> None:
    depth = torch.rand(1, 1, 40, 50) * 80 + 30
    disparity = 569.0 * 4.0 / depth + 1.7
    valid = torch.ones_like(depth, dtype=torch.bool)
    result = disparity_affine_agreement(disparity, depth, valid)
    assert abs(result["disp_affine_r"].value - 1.0) < 1e-9
    assert result["disp_affine_rmse_px"].value < 1e-3  # float32 inputs


def test_affine_agreement_is_unavailable_with_too_few_pixels() -> None:
    depth = torch.full((1, 1, 10, 10), 50.0)
    result = disparity_affine_agreement(depth, depth, torch.ones_like(depth, dtype=torch.bool))
    assert result["disp_affine_r"].value is None and result["disp_affine_rmse_px"].value is None


class _FakeAdapter:
    def __init__(self, sample: object) -> None:
        self._sample = sample

    def load(self, sample_id: str, device: object = None) -> object:
        return self._sample

    def tool_mask(self, sample_id: str, device: object = None) -> torch.Tensor:
        mask = torch.zeros(1, 1, 40, 50, dtype=torch.bool)
        mask[..., :5] = True
        return mask

    def reference_depth(self, sample_id: str, device: object = None) -> torch.Tensor:
        return torch.rand(1, 1, 40, 50) * 80 + 30


def test_row_contains_both_views_and_the_no_tool_psnr() -> None:
    from types import SimpleNamespace

    left, right = torch.rand(1, 3, 40, 50), torch.rand(1, 3, 40, 50)
    sample = SimpleNamespace(batch=SimpleNamespace(left=left, right=right), supervision=None)
    prediction = Prediction(
        torch.full((1, 1, 40, 50), 10.0),
        RenderOutput(image=left.clone()),
        RenderOutput(image=right * 0.9),
    )
    make_row = render_row_builder(
        lambda s: prediction, _FakeAdapter(sample), device=torch.device("cpu"), perceptual=None
    )  # type: ignore[arg-type]
    row = make_row("scene/000000")
    metrics = row["metrics"]
    assert row["sample_id"] == "scene/000000"
    assert metrics["left_psnr"]["value"] == float("inf")  # identical left render
    assert metrics["right_psnr"]["value"] < 40
    assert {"left_ssim", "right_ssim", "left_psnr_no_tool", "disp_affine_r"} <= set(metrics)
    assert metrics["left_psnr_no_tool"]["valid_count"] == 3 * 40 * 45
