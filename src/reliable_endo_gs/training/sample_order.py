"""Reproducible ordering of an unchanged training manifest."""

from __future__ import annotations

import hashlib
import json
import random

from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.training.student_run import StudentSamples


class SeededSampleOrder:
    """One fixed permutation per experiment; resume cursors index that order.

    No samples are added or removed. A local RNG avoids changing model or
    optimizer randomness. The manifest identity remains separate from ordering.
    """

    def __init__(self, samples: StudentSamples, seed: int) -> None:
        self.samples = samples
        indices = list(range(len(samples)))
        random.Random(seed).shuffle(indices)
        self.indices = tuple(indices)
        self.order_hash = hashlib.sha256(json.dumps(indices).encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> ScaredA00Sample:
        return self.samples[self.indices[index]]
