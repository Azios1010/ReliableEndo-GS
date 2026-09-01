"""Thin entry point for contract-level evaluation of serialized tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reliable_endo_gs.evaluation import evaluate_depth, evaluate_rendering, evaluate_stereo


def evaluate_baseline_tensors(
    *,
    disparity: torch.Tensor,
    target_disparity: torch.Tensor,
    disparity_mask: torch.Tensor,
    depth: torch.Tensor | None = None,
    target_depth: torch.Tensor | None = None,
    depth_mask: torch.Tensor | None = None,
    image: torch.Tensor | None = None,
    target_image: torch.Tensor | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "stereo": {
            name: metric.to_dict()
            for name, metric in evaluate_stereo(disparity, target_disparity, disparity_mask).items()
        }
    }
    if depth is not None or target_depth is not None:
        if depth is None or target_depth is None:
            raise ValueError("depth and target_depth must be supplied together")
        result["depth"] = {
            name: metric.to_dict()
            for name, metric in evaluate_depth(depth, target_depth, depth_mask).items()
        }
    if image is not None or target_image is not None:
        if image is None or target_image is None:
            raise ValueError("image and target_image must be supplied together")
        result["rendering"] = {
            name: metric.to_dict()
            for name, metric in evaluate_rendering(image, target_image).items()
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="torch file containing contract tensors")
    args = parser.parse_args(argv)
    values = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(values, dict):
        raise ValueError("input must be a mapping of named tensors")
    result = evaluate_baseline_tensors(**values)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
