"""External quarter-cohort RGB review and ROI finalization; no training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("review", "finalize"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--development-manifest", type=Path)
    parser.add_argument("--roles-path", type=Path)
    parser.add_argument("--refit-config-path", type=Path)
    parser.add_argument("--previous-roi-policy", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    from reliable_endo_gs.config.selector_refit import load_selector_refit_config
    from reliable_endo_gs.data.selector_cohort import (
        finalize_selector_rois,
        prepare_selector_review,
    )

    if args.mode == "finalize":
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
