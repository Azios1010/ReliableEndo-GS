"""Stage1 SCARED-C preparation, smoke, and full-training entry point."""

from __future__ import annotations

import argparse

import torch

from reliable_endo_gs.training.stage1_scared_c import (
    DEFAULT_CONFIG_PATH,
    benchmark_stage1_cache_data,
    load_stage1_scared_c_config,
    prepare_stage1_full_data,
    run_stage1_scratch_smoke,
    run_stage1_scratch_training,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="run the fixed 60000-step protocol")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="resolve and persist the deterministic full-data split without training",
    )
    parser.add_argument(
        "--build-cache",
        action="store_true",
        help="with --prepare-only, materialize the selected corrected-video GT cache",
    )
    parser.add_argument(
        "--artifact-root",
        default="outputs/stage1_scared_c_full_v1",
    )
    parser.add_argument(
        "--resume-ckpt",
        default=None,
        help="resume the full-data run from a validated Stage1 checkpoint",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_stage1_scared_c_config(args.config)
    if args.prepare_only and (args.full or args.smoke_steps is not None):
        raise SystemExit("--prepare-only cannot be combined with --full or --smoke-steps")
    if args.build_cache and not args.prepare_only:
        raise SystemExit("--build-cache requires --prepare-only")
    if args.resume_ckpt is not None and not args.full:
        raise SystemExit("--resume-ckpt requires --full")
    if args.prepare_only:
        split = prepare_stage1_full_data(config, build_cache=args.build_cache)
        print(f"split_manifest={config.frame_split_path}")
        print(f"split_sha256={split.manifest_sha256}")
        print(f"keyframes={len(split.keyframe_entries)}")
        print(f"source_samples={len(split.source_sample_ids)}")
        print(f"train_samples={split.train_count}")
        print(f"validation_samples={split.validation_count}")
        print(f"cache_built={args.build_cache}")
        return 0
    if args.full and args.smoke_steps is not None:
        raise SystemExit("--full and --smoke-steps are mutually exclusive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Stage1 SCARED-C requires a visible CUDA device")
    if args.full:
        result = run_stage1_scratch_training(
            config,
            device=device,
            artifact_root=args.artifact_root,
            resume_ckpt=args.resume_ckpt,
        )
        print(f"steps_completed={result.steps_completed}")
        print(f"best_step={result.best_step}")
        print(f"best_validation_epe={result.best_validation_epe:.8f}")
        print(f"selected_checkpoint={result.selected_checkpoint}")
        print(f"selected_checkpoint_sha256={result.selected_checkpoint_sha256}")
        return 0
    smoke_steps = 3 if args.smoke_steps is None else args.smoke_steps
    config.require_safe_mode(steps=smoke_steps)
    data_seconds = benchmark_stage1_cache_data(config, samples=8)
    result = run_stage1_scratch_smoke(config, device=device, steps=smoke_steps)
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
