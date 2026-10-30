"""Canonical model-Adam LR schedule, indexed by the committed sample cursor."""

from __future__ import annotations

import math

from torch.optim import Optimizer

from reliable_endo_gs.training.student_config import StudentLearningConfig


def model_learning_rate(config: StudentLearningConfig, epoch: int) -> float:
    """Return scalar LR for a zero-based epoch; decay horizon is independent of run length.

    Cosine runs from the initial LR in epoch 0 to the declared minimum ratio in
    epoch ``model_lr_decay_epochs - 1``. Later epochs retain that floor. Keeping
    the horizon separate preserves the same LR prefix when validation selects
    a shorter full-data run. Only the model optimizer uses this schedule.
    """
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    if config.model_lr_schedule == "constant":
        return config.learning_rate
    progress = min(epoch / (config.model_lr_decay_epochs - 1), 1.0)
    factor = (
        config.model_lr_min_ratio
        + (1 - config.model_lr_min_ratio) * (1 + math.cos(math.pi * progress)) / 2
    )
    return config.learning_rate * factor


def checkpoint_model_learning_rate(
    config: StudentLearningConfig, samples_seen: int, samples_per_epoch: int
) -> float:
    """LR stored after the last committed update, including an exact epoch boundary.

    The next epoch's LR is applied immediately before its first update, so a
    boundary checkpoint still contains the previous epoch's LR. An untouched
    initialization uses epoch 0. No mutable scheduler state is needed.
    """
    if type(samples_seen) is not int or samples_seen < 0:
        raise ValueError("samples_seen must be a nonnegative integer")
    if type(samples_per_epoch) is not int or samples_per_epoch < 1:
        raise ValueError("samples_per_epoch must be a positive integer")
    return model_learning_rate(config, max(0, samples_seen - 1) // samples_per_epoch)


def apply_model_learning_rate(
    optimizer: Optimizer, config: StudentLearningConfig, epoch: int
) -> float:
    """Set only model-optimizer scalar group LRs; leave Adam moments/steps intact."""
    rate = model_learning_rate(config, epoch)
    for group in optimizer.param_groups:
        group["lr"] = rate
    return rate
