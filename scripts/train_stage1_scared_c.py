"""Thin guarded entry point for Stage1 SCARED-C.

The inactive remediation config permits only ``--smoke-steps 3``.  A separate
full-training task must activate the reviewed sequence split before requesting
any longer run.
"""

from __future__ import annotations

import argparse

import torch

from reliable_endo_gs.training.stage1_scared_c import (
    DEFAULT_CONFIG_PATH,
    benchmark_stage1_cache_data,
    load_stage1_scared_c_config,
    run_stage1_scratch_smoke,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_stage1_scared_c_config(args.config)
    if args.smoke_steps != 3:
        raise SystemExit("remediation entry point permits exactly --smoke-steps 3")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Stage1 SCARED-C smoke requires a visible CUDA device")
    data_seconds = benchmark_stage1_cache_data(config, samples=8)
    result = run_stage1_scratch_smoke(config, device=device, steps=3)
    print(f"mean_data_seconds={data_seconds:.6f}")
    print(f"mean_compute_seconds={result.mean_compute_seconds:.6f}")
    print(f"mean_total_seconds={result.mean_total_seconds:.6f}")
    print(f"losses={','.join(f'{value:.8f}' for value in result.losses)}")
    print(f"allocations_gib={','.join(f'{value:.6f}' for value in result.allocations_gib)}")
    print(f"peak_allocated_gib={result.peak_allocated_gib:.6f}")
    print(f"checkpoint_loaded={result.checkpoint_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
