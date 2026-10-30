"""Explicit target-quality preflight and read-only native selection evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("quality", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--roles", type=Path, default=Path("configs/teachers/a01_scared.yaml"))
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--max-samples", type=int, default=112 if name == "evaluate" else 64)
        if name == "quality":
            command.add_argument("--cache-root", type=Path, required=True)
            command.add_argument(
                "--config", type=Path, default=Path("configs/routing/intervention_selection.yaml")
            )
            command.add_argument("--sample-id", action="append", default=[])
        else:
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--training-manifest", type=Path, required=True)
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--cost-calibration", type=Path)
            command.add_argument("--target-mask-index", type=Path)
            command.add_argument("--metric-config", type=Path)
            command.add_argument("--study-protocol-path", type=Path)
            command.add_argument("--selector-refit-path", type=Path)
            command.add_argument("--lpips-backbone", type=Path)
            command.add_argument("--lpips-calibration", type=Path)
            command.add_argument("--require-lpips", action="store_true")
            command.add_argument(
                "--sample-worklist", type=Path, help="Use the sealed development IDs"
            )
            command.add_argument("--sample-id", action="append", default=[])
            command.add_argument(
                "--checkpoint-sha256",
                help="Explicit pinned historical read-only evaluation; records old/current sources",
            )
    sweep = commands.add_parser("weight-sweep")
    sweep.add_argument("--pilot-dir", type=Path, required=True)
    sweep.add_argument("--output-dir", type=Path, required=True)
    sweep.add_argument("--weights", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    geometry = commands.add_parser("geometry-screen")
    geometry.add_argument("--checkpoint", type=Path, required=True)
    geometry.add_argument("--data-root", type=Path, required=True)
    geometry.add_argument("--manifest", type=Path, required=True)
    geometry.add_argument("--training-manifest", type=Path, required=True)
    geometry.add_argument("--roles", type=Path, default=Path("configs/teachers/a01_scared.yaml"))
    geometry.add_argument("--candidate-config", type=Path, action="append")
    geometry.add_argument("--output-dir", type=Path, required=True)
    geometry.add_argument("--per-dataset", type=int, default=2)
    native = commands.add_parser("geometry-native")
    native.add_argument("--checkpoint", type=Path, required=True)
    native.add_argument("--probe-report", type=Path, required=True)
    native.add_argument("--data-root", type=Path, required=True)
    native.add_argument("--manifest", type=Path, required=True)
    native.add_argument("--training-manifest", type=Path, required=True)
    native.add_argument("--roles", type=Path, default=Path("configs/teachers/a01_scared.yaml"))
    native.add_argument("--output-dir", type=Path, required=True)
    table = commands.add_parser("collect-outcomes")
    table.add_argument("--checkpoint", type=Path, required=True)
    table.add_argument("--probe-report", type=Path, required=True)
    table.add_argument("--data-root", type=Path, required=True)
    table.add_argument("--manifest", type=Path, required=True)
    table.add_argument("--training-manifest", type=Path, required=True)
    table.add_argument("--roles", type=Path, default=Path("configs/teachers/a01_scared.yaml"))
    table.add_argument("--config", type=Path, required=True)
    table.add_argument("--collection-role", choices=("train", "development"), required=True)
    table.add_argument("--per-dataset", type=int, default=4)
    table.add_argument("--timing-repeats", type=int, default=3)
    table.add_argument("--output-dir", type=Path, required=True)
    pilot = commands.add_parser("outcome-pilot")
    pilot.add_argument("--training-table", type=Path, required=True)
    pilot.add_argument("--development-table", type=Path, required=True)
    pilot.add_argument("--config", type=Path, required=True)
    pilot.add_argument("--output-dir", type=Path, required=True)
    pilot.add_argument("--device", default="cpu")
    pilot.add_argument(
        "--training-policy",
        type=Path,
        help="Opt-in grouped early stopping and regularization controls",
    )
    pilot.add_argument("--budgets-ms", type=float, nargs="+", default=[250.0, 500.0, 1000.0])
    pilot.add_argument(
        "--relabel-policy",
        action="store_true",
        help="Explicit v2 replay: preserve collection identities and bind new risk/utility labels",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "quality":
            from reliable_endo_gs.evaluation.target_quality import audit_target_quality

            result = audit_target_quality(
                data_root=args.data_root,
                manifest_path=args.manifest,
                roles_path=args.roles,
                cache_root=args.cache_root,
                config_path=args.config,
                output_dir=args.output_dir,
                sample_ids=args.sample_id,
                max_samples=args.max_samples,
            )
        elif args.command in ("geometry-native", "collect-outcomes"):
            from reliable_endo_gs.evaluation.geometry_native import (
                evaluate_native_geometry_candidates,
            )

            result = evaluate_native_geometry_candidates(
                checkpoint=args.checkpoint,
                probe_report=args.probe_report,
                data_root=args.data_root,
                manifest_path=args.manifest,
                training_manifest=args.training_manifest,
                roles_path=args.roles,
                output_dir=args.output_dir,
                **(
                    dict(
                        tradeoff_config_path=args.config,
                        collection_role=args.collection_role,
                        per_dataset=args.per_dataset,
                        timing_repeats=args.timing_repeats,
                    )
                    if args.command == "collect-outcomes"
                    else {}
                ),
            )
        elif args.command == "outcome-pilot":
            from reliable_endo_gs.training.outcome_learning import run_outcome_pilot

            result = run_outcome_pilot(
                training_table=args.training_table,
                development_table=args.development_table,
                config_path=args.config,
                output_dir=args.output_dir,
                device_name=args.device,
                budgets_ms=tuple(args.budgets_ms),
                relabel_policy=args.relabel_policy,
                training_policy_path=args.training_policy,
            )
        elif args.command == "geometry-screen":
            from reliable_endo_gs.evaluation.geometry_candidates import screen_geometry_candidates

            result = screen_geometry_candidates(
                checkpoint=args.checkpoint,
                data_root=args.data_root,
                manifest_path=args.manifest,
                training_manifest=args.training_manifest,
                roles_path=args.roles,
                candidate_configs=tuple(
                    args.candidate_config
                    or [
                        Path("configs/routing/geometry_correspondence_candidate.yaml"),
                        Path("configs/routing/geometry_support_candidate.yaml"),
                    ]
                ),
                output_dir=args.output_dir,
                per_dataset=args.per_dataset,
            )
        elif args.command == "weight-sweep":
            from reliable_endo_gs.evaluation.stereo_weight_sweep import sweep_stereo_weights

            result = sweep_stereo_weights(
                pilot_dir=args.pilot_dir, output_dir=args.output_dir, weights=tuple(args.weights)
            )
        else:
            from reliable_endo_gs.evaluation.intervention_selection import (
                evaluate_intervention_selection,
            )

            if args.sample_worklist is not None:
                if args.sample_id:
                    raise ValueError("choose --sample-id or --sample-worklist")
                work = json.loads(args.sample_worklist.read_text())
                ids = work.get("development")
                if not isinstance(ids, list) or not ids or any(not isinstance(s, str) for s in ids):
                    raise ValueError("worklist must declare development sample IDs")
                args.sample_id = ids
            result = evaluate_intervention_selection(
                checkpoint=args.checkpoint,
                data_root=args.data_root,
                manifest_path=args.manifest,
                training_manifest=args.training_manifest,
                roles_path=args.roles,
                output_dir=args.output_dir,
                max_samples=args.max_samples,
                device_name=args.device,
                cost_calibration_path=args.cost_calibration,
                target_mask_index=args.target_mask_index,
                metric_config_path=args.metric_config,
                sample_ids=tuple(args.sample_id) if args.sample_id else None,
                expected_checkpoint_sha256=args.checkpoint_sha256,
                selector_refit_path=args.selector_refit_path,
                study_protocol_path=args.study_protocol_path,
                lpips_backbone=args.lpips_backbone,
                lpips_calibration=args.lpips_calibration,
                require_lpips=args.require_lpips,
            )
    except (OSError, ValueError, RuntimeError, KeyError, AssertionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
