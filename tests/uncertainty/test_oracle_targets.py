"""Oracle target leakage and validity tests."""

import pytest
import torch

from reliable_endo_gs.uncertainty.oracle import build_oracle_target
from reliable_endo_gs.uncertainty.records import UncertaintyLeakageError


def test_oracle_target_uses_only_jointly_valid_residuals_and_rejects_inference() -> None:
    predicted = torch.tensor([[[[2.0, 3.0]]]])
    ground_truth = torch.tensor([[[[1.5, 10.0]]]])
    prediction_valid = torch.tensor([[[[True, True]]]])
    target_valid = torch.tensor([[[[True, False]]]])

    target = build_oracle_target(predicted, ground_truth, prediction_valid, target_valid)

    assert torch.equal(target.absolute_error, torch.tensor([[[[0.5, 7.0]]]]))
    assert torch.equal(target.valid_mask, torch.tensor([[[[True, False]]]]))
    with pytest.raises(UncertaintyLeakageError, match="cannot be used"):
        target.as_inference_feature()


def test_oracle_metadata_must_be_nonempty_and_non_deployable() -> None:
    values = torch.zeros(1, 1, 1, 1)
    valid = torch.ones_like(values, dtype=torch.bool)

    with pytest.raises(ValueError, match="target_id"):
        from reliable_endo_gs.uncertainty.oracle import OracleUncertaintyTarget

        OracleUncertaintyTarget(values, valid, target_id="")
    with pytest.raises(ValueError, match="non_deployable"):
        from reliable_endo_gs.uncertainty.oracle import OracleUncertaintyTarget

        OracleUncertaintyTarget(values, valid, non_deployable=False)
