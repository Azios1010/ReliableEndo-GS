"""StereoPrediction structural contract tests."""

import pytest
import torch

from reliable_endo_gs.contracts import StereoPrediction


def _prediction(**overrides: object) -> StereoPrediction:
    values: dict[str, object] = {
        "disparity": torch.ones(2, 1, 4, 5),
        "valid_mask": torch.ones(2, 1, 4, 5, dtype=torch.bool),
    }
    values.update(overrides)
    return StereoPrediction(**values)  # type: ignore[arg-type]


def test_valid_prediction_supports_tensor_and_sequence_iterations() -> None:
    tensor_form = _prediction(disparity_iterations=torch.zeros(2, 3, 4, 5))
    sequence_form = _prediction(
        disparity_iterations=[torch.zeros(2, 1, 4, 5), torch.ones(2, 1, 4, 5)],
        sigma_d=torch.ones(2, 1, 4, 5),
        diagnostics={"source": "test"},
    )

    assert isinstance(tensor_form.disparity_iterations, torch.Tensor)
    assert isinstance(sequence_form.disparity_iterations, tuple)
    assert sequence_form.batch_size == 2
    assert sequence_form.spatial_shape == (4, 5)
    with pytest.raises(TypeError):
        sequence_form.diagnostics["new"] = 1  # type: ignore[index]


def test_prediction_rejects_rank_channel_and_mask_errors() -> None:
    with pytest.raises(ValueError, match="disparity must have rank 4"):
        _prediction(disparity=torch.zeros(2, 4, 5))
    with pytest.raises(ValueError, match="disparity must have shape"):
        _prediction(disparity=torch.zeros(2, 2, 4, 5))
    with pytest.raises(ValueError, match="valid_mask must have shape"):
        _prediction(valid_mask=torch.ones(2, 1, 3, 5, dtype=torch.bool))
    with pytest.raises(TypeError, match="dtype torch.bool"):
        _prediction(valid_mask=torch.ones(2, 1, 4, 5))


def test_prediction_rejects_optional_shape_dtype_and_device_mismatch() -> None:
    with pytest.raises(ValueError, match="sigma_d must have shape"):
        _prediction(sigma_d=torch.zeros(2, 1, 3, 5))
    with pytest.raises(TypeError, match="sigma_d must have the same dtype"):
        _prediction(sigma_d=torch.zeros(2, 1, 4, 5, dtype=torch.float64))
    with pytest.raises(ValueError, match="disparity_iterations must have shape"):
        _prediction(disparity_iterations=torch.zeros(2, 3, 3, 5))
    with pytest.raises(ValueError, match=r"disparity_iterations\[0\] must have shape"):
        _prediction(disparity_iterations=[torch.zeros(2, 1, 3, 5)])
    with pytest.raises(ValueError, match="same device"):
        _prediction(sigma_d=torch.zeros(2, 1, 4, 5, device="meta"))


def test_prediction_accepts_nonfinite_values_for_later_scientific_validation() -> None:
    disparity = torch.full((2, 1, 4, 5), torch.nan)
    prediction = _prediction(disparity=disparity)
    assert prediction.disparity is disparity
