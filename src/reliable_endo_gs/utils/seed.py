"""Explicit scientific seeding without import-time device initialization."""

import random


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch random number generators.

    CUDA generators are seeded only when CUDA is available. This function does
    not enable deterministic algorithms or otherwise change backend behavior.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)

    # Keep the data-only CLI import path lightweight. PyTorch is a required
    # runtime dependency, but is imported only when scientific seeding occurs.
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
