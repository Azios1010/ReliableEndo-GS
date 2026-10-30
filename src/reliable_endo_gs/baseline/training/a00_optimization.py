"""Finite optimizer updates with explicit, measured AMP overflow retries."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch


def finite_update(
    closure: Callable[[], tuple[torch.Tensor, dict[str, float | None]]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    device: torch.device,
    max_overflow_retries: int,
) -> dict[str, Any]:
    """Count only real updates; GradScaler handles overflow before retrying.

    Retrying changes no loss or precision. Every extra presentation and its
    runtime are recorded. A nonfinite FP32 gradient or persistent AMP failure
    raises instead of silently pretending an optimizer update happened.
    """
    retries = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for attempt in range(max_overflow_retries + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = closure()
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(grad_norm)):
            if not scaler.is_enabled():
                raise ValueError("nonfinite FP32 gradient")
            before = scaler.get_scale()
            # unscale_ has already recorded found_inf; step must skip the update.
            scaler.step(optimizer)
            scaler.update()
            after = scaler.get_scale()
            if after >= before:
                raise ValueError("nonfinite norm without GradScaler overflow detection")
            retries.append({"attempt": attempt + 1, "scale_before": before, "scale_after": after})
            continue
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            "loss": float(loss.detach()),
            **metrics,
            "gradient_norm_before_clip": float(grad_norm),
            "step_seconds": time.perf_counter() - started,
            "amp_overflow_retries": retries,
        }
    raise ValueError(f"AMP gradients remained nonfinite after {max_overflow_retries} retries")
