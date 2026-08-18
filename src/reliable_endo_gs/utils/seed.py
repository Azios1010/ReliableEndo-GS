"""Seed utilities for dependencies currently present in the project."""

import random


def set_seed(seed: int) -> None:
    """Seed Python's random module.

    Future scientific dependencies may extend this function when introduced.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
