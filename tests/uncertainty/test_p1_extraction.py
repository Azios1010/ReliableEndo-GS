"""Synthetic P1 extraction and baseline-equivalence tests."""

from pathlib import Path

import pytest
import torch

from reliable_endo_gs.evaluation.p1_runner import _verify_no_final_access
from reliable_endo_gs.uncertainty.extraction import extract_model_iterations, extract_proxy_outputs
from reliable_endo_gs.uncertainty.proxies import photometric_residual_score


def test_photometric_residual_has_zero_valid_error_for_known_integer_shift() -> None:
    left = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]]).repeat(1, 3, 1, 1)
    right = torch.tensor([[[[1.0, 2.0, 3.0, 3.0]]]]).repeat(1, 3, 1, 1)
    disparity = torch.ones(1, 1, 1, 4)

    result = photometric_residual_score(left, right, disparity)

    assert torch.equal(result.valid_mask, torch.tensor([[[[False, True, True, True]]]]))
    assert torch.allclose(result.score[result.valid_mask], torch.zeros(3))


def test_extract_proxy_outputs_keeps_u5_unavailable_and_uses_inference_mask() -> None:
    iterations = torch.tensor(
        [[[[1.0, 1.0]], [[1.5, 1.0]], [[2.0, 1.0]]]], dtype=torch.float32
    )
    left = torch.zeros(1, 3, 1, 2)
    right = torch.zeros_like(left)

    output = extract_proxy_outputs(iterations, left, right)

    assert output.disparity.shape == (1, 1, 1, 2)
    assert output.inference_valid_mask.all()
    assert output.lr_consistency is None
    assert output.correlation_entropy is None
    assert set(output.scores()) == {
        "final_update_magnitude",
        "iteration_disagreement",
        "photometric_residual",
    }


def test_fake_model_instrumentation_preserves_final_disparity() -> None:
    class FakeEncoder:
        def __call__(self, images: torch.Tensor) -> tuple[None, None, torch.Tensor]:
            return None, None, images

    class FakeRaft:
        def __call__(self, features: torch.Tensor, *, iters: int, test_mode: bool) -> object:
            del test_mode
            batch = features.shape[0] // 2
            base = features[:batch, :1]
            return [base * 0.0 + float(index) for index in range(1, iters + 1)]

    class FakeModel:
        img_encoder = FakeEncoder()
        raft_stereo = FakeRaft()
        val_iters = 3

    model = FakeModel()
    left = torch.zeros(1, 3, 2, 3)
    right = torch.zeros_like(left)
    extracted = extract_model_iterations(model, left, right, iterations=3)
    baseline_final = torch.full((1, 1, 2, 3), 3.0)

    assert torch.equal(extracted[:, -1:], baseline_final)


def test_final_evaluation_config_is_rejected_by_development_guard() -> None:
    with pytest.raises(RuntimeError, match="final"):
        _verify_no_final_access(
            config_path=Path("scared_phase1_final_v1.yaml"),
            split_path=Path("phase1_final_untouched_v1.json"),
            manifest_path=Path("scared_phase1_final_untouched_v1.json"),
            sequences=("dataset_5/keyframe_1",),
        )


def test_final_evaluation_manifest_is_rejected_by_development_guard() -> None:
    with pytest.raises(RuntimeError, match="final"):
        _verify_no_final_access(
            config_path=Path("p1_proxy.yaml"),
            split_path=Path("five_keyframe_v3.json"),
            manifest_path=Path("scared_phase1_final_untouched_v1.json"),
            sequences=("dataset_1/keyframe_1",),
        )
