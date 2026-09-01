"""Run a CPU-safe analytic audit of Plan 06 geometry uncertainty propagation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from reliable_endo_gs.geometry import (
    GeometryConvention,
    backproject_depth,
    center_covariance_from_disparity,
    depth_uncertainty_from_disparity,
    disparity_to_depth,
    pixel_rays,
)


def main() -> int:
    """Execute the configured analytic fixture and write a provenance-bearing JSON report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/geometry/stereo_geometry.yaml"),
        help="Path to the standalone geometry audit YAML configuration.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing --output report.",
    )
    arguments = parser.parse_args()
    if arguments.output is not None and arguments.output.exists() and not arguments.overwrite:
        parser.error("--output already exists; pass --overwrite to replace it")

    config_text = arguments.config.read_text(encoding="utf-8")
    raw = yaml.safe_load(config_text)
    config = _as_mapping(raw, "configuration")
    convention = GeometryConvention(**_as_mapping(config["convention"], "convention"))
    numerical = _as_mapping(config["numerical"], "numerical")
    fixture = _as_mapping(config["audit_fixture"], "audit_fixture")
    report = _run_fixture(convention, numerical, fixture)
    report["config_path"] = arguments.config.as_posix()
    report["config_sha256"] = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(rendered)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


def _run_fixture(
    convention: GeometryConvention,
    numerical: dict[str, Any],
    fixture: dict[str, Any],
) -> dict[str, Any]:
    disparities = _float_list(fixture["disparity_values_px"], "audit_fixture.disparity_values_px")
    sigmas = _float_list(fixture["sigma_d_px"], "audit_fixture.sigma_d_px")
    if len(disparities) != len(sigmas) or not disparities:
        raise ValueError(
            "fixture disparity_values_px and sigma_d_px must be non-empty equal-length lists"
        )
    focal_length_x = _float(fixture["focal_length_x_px"], "audit_fixture.focal_length_x_px")
    baseline = _float(fixture["baseline_m"], "audit_fixture.baseline_m")
    principal_point = _float_list(fixture["principal_point_px"], "audit_fixture.principal_point_px")
    if len(principal_point) != 2:
        raise ValueError("audit_fixture.principal_point_px must contain [cx, cy]")

    dtype = torch.float64
    disparity = torch.tensor(disparities, dtype=dtype).reshape(1, 1, 1, -1)
    sigma_d = torch.tensor(sigmas, dtype=dtype).reshape_as(disparity)
    valid_mask = torch.ones_like(disparity, dtype=torch.bool)
    focal = torch.tensor([focal_length_x], dtype=dtype)
    baseline_tensor = torch.tensor([baseline], dtype=dtype)
    intrinsics = torch.tensor(
        [
            [
                [focal_length_x, 0.0, principal_point[0]],
                [0.0, focal_length_x, principal_point[1]],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=dtype,
    )
    minimum = _float(numerical["min_abs_disparity_px"], "numerical.min_abs_disparity_px")
    epsilon = _float(numerical["epsilon_m2"], "numerical.epsilon_m2")
    cap_raw = numerical.get("max_eigenvalue_m2")
    cap = None if cap_raw is None else _float(cap_raw, "numerical.max_eigenvalue_m2")
    rank_tolerance = _float(numerical["rank_tolerance_m2"], "numerical.rank_tolerance_m2")

    depth = disparity_to_depth(
        disparity,
        focal,
        baseline_tensor,
        valid_mask,
        min_abs_disparity=minimum,
        convention=convention,
    )
    rays = pixel_rays(intrinsics, height=1, width=disparity.shape[-1], convention=convention)
    centers = backproject_depth(depth.depth, intrinsics, depth.valid_mask, convention=convention)
    sigma_depth = depth_uncertainty_from_disparity(
        disparity,
        sigma_d,
        focal,
        baseline_tensor,
        valid_mask,
        min_abs_disparity=minimum,
        convention=convention,
    )
    covariance = center_covariance_from_disparity(
        disparity,
        sigma_d,
        focal,
        baseline_tensor,
        rays.rays,
        valid_mask,
        min_abs_disparity=minimum,
        epsilon=epsilon,
        max_eigenvalue=cap,
        rank_tolerance=rank_tolerance,
        convention=convention,
    )
    return {
        "formula_schema_version": "1.0",
        "dtype": str(dtype).replace("torch.", ""),
        "device": "cpu",
        "convention": asdict(convention),
        "depth_m": depth.depth.flatten().tolist(),
        "sigma_depth_m": sigma_depth.sigma_depth.flatten().tolist(),
        "centers_left_camera_m": centers.centers.permute(0, 2, 3, 1).reshape(-1, 3).tolist(),
        "cov_center_left_camera_m2": covariance.cov_center.reshape(-1, 3, 3).tolist(),
        "depth_diagnostics": asdict(depth.diagnostics),
        "backprojection_diagnostics": asdict(centers.diagnostics),
        "covariance_diagnostics": asdict(covariance.diagnostics),
    }


def _as_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a float")
    return float(value)


def _float_list(value: object, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of floats")
    return [_float(item, name) for item in value]


if __name__ == "__main__":
    raise SystemExit(main())
