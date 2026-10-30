"""External quarter-cohort RGB review and ROI finalization; no training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("review", "finalize", "automatic"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--development-manifest", type=Path)
    parser.add_argument("--roles-path", type=Path)
    parser.add_argument("--refit-config-path", type=Path)
    parser.add_argument("--previous-roi-policy", type=Path)
    parser.add_argument("--study-protocol-path", type=Path)
    parser.add_argument("--per-keyframe", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    from reliable_endo_gs.config.selector_refit import load_selector_refit_config
    from reliable_endo_gs.data.selector_cohort import (
        finalize_selector_rois,
        prepare_automatic_selector_cohort,
        prepare_selector_review,
    )

    if args.mode == "automatic":
        if any(
            getattr(args, name) is None
            for name in (
                "training_manifest",
                "development_manifest",
                "roles_path",
                "refit_config_path",
                "study_protocol_path",
            )
        ):
            parser.error("automatic requires manifests, roles, refit config and study protocol")
        from reliable_endo_gs.config.selector_study import load_selector_study_config
        from reliable_endo_gs.routing.support_grid import automatic_study_regions

        study = load_selector_study_config(args.study_protocol_path)
        _, regions = automatic_study_regions(study)
        result = prepare_automatic_selector_cohort(
            data_root=args.data_root,
            output_dir=args.output_dir,
            training_manifest=args.training_manifest,
            development_manifest=args.development_manifest,
            roles_path=args.roles_path,
            policy=load_selector_refit_config(args.refit_config_path),
            candidate_identity=regions.identity,
            study_config_hash=study.config_hash,
            per_keyframe=args.per_keyframe,
        )
    elif args.mode == "finalize":
        result = finalize_selector_rois(
            data_root=args.data_root,
            output_dir=args.output_dir,
            resume=args.resume,
            workers=args.workers,
        )
    else:
        if any(
            getattr(args, name) is None
            for name in (
                "training_manifest",
                "development_manifest",
                "roles_path",
                "refit_config_path",
                "previous_roi_policy",
            )
        ):
            parser.error("review requires manifests, roles, refit config and prior ROI policy")
        result = prepare_selector_review(
            data_root=args.data_root,
            output_dir=args.output_dir,
            training_manifest=args.training_manifest,
            development_manifest=args.development_manifest,
            roles_path=args.roles_path,
            policy=load_selector_refit_config(args.refit_config_path),
            previous_roi_policy=args.previous_roi_policy,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
