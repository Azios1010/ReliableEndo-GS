"""CPU-safe scientific seeding tests."""

import random

import pytest
import torch

from reliable_endo_gs.utils.seed import set_seed


def test_same_seed_repeats_python_and_torch_sequences() -> None:
    set_seed(123)
    python_first = [random.random() for _ in range(3)]
    torch_first = torch.rand(3)

    set_seed(123)
    python_second = [random.random() for _ in range(3)]
    torch_second = torch.rand(3)

    assert python_first == python_second
    assert torch.equal(torch_first, torch_second)


def test_different_seed_changes_torch_sequence() -> None:
    set_seed(123)
    first = torch.rand(4)
    set_seed(456)
    second = torch.rand(4)
    assert not torch.equal(first, second)


@pytest.mark.parametrize("invalid", [True, 1.5, "42", None])
def test_seed_rejects_non_integer_values(invalid: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        set_seed(invalid)  # type: ignore[arg-type]
