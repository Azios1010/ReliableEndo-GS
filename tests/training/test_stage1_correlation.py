"""Focused equivalence check for the PyTorch 1-D correlation contract."""

import torch

from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D


def test_pytorch_correlation_matches_independent_one_level_reference() -> None:
    torch.manual_seed(7)
    fmap_left = torch.randn(1, 8, 3, 11)
    fmap_right = torch.randn(1, 8, 3, 13)
    coords = torch.zeros(1, 2, 3, 11)
    coords[:, 0] = 4.25
    coords[:, 1] = torch.arange(3, dtype=coords.dtype).view(1, 3, 1)

    fallback = CorrBlock1D(fmap_left, fmap_right, num_levels=1, radius=2)(coords)
    reference = PytorchAlternateCorrBlock1D(
        fmap_left, fmap_right, num_levels=1, radius=2
    )(coords)

    assert fallback.shape == reference.shape == (1, 5, 3, 11)
    assert torch.allclose(fallback, reference, atol=1e-5, rtol=1e-5)
