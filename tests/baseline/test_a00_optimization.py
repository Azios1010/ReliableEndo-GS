"""AMP overflow skips must not be counted as successful optimizer updates."""

import pytest
import torch
from torch import nn

from reliable_endo_gs.baseline.training.a00_optimization import finite_update


def test_amp_overflow_retries_reduce_scale_before_exactly_one_update():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
    result = finite_update(
        lambda: (model.weight.sum() * 1e38, {}),
        model,
        optimizer,
        scaler,
        device=torch.device("cpu"),
        max_overflow_retries=8,
    )
    assert result["amp_overflow_retries"]
    assert all(row["scale_after"] < row["scale_before"] for row in result["amp_overflow_retries"])
    assert torch.isfinite(model.weight).all()
    assert torch.allclose(model.weight, torch.tensor([[-0.1]]))


def test_nonfinite_fp32_gradient_fails_without_model_update():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    with pytest.raises(ValueError, match="nonfinite FP32 gradient"):
        finite_update(
            lambda: (model.weight.sqrt().sum(), {}),
            model,
            optimizer,
            scaler,
            device=torch.device("cpu"),
            max_overflow_retries=0,
        )
    assert (model.weight == 0).all()
