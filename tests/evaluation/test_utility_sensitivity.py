"""Common physical preference, raw-unit safety, unknowns and TRAIN isolation."""

from dataclasses import replace
from pathlib import Path

import torch

from reliable_endo_gs.config.intervention import load_intervention_config
from reliable_endo_gs.evaluation.utility_sensitivity import quality_oracle_tradeoff
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES
from reliable_endo_gs.training.selector_outcomes import SelectorOutcomes
from tests.routing.test_metric_tradeoff import vector
from tests.routing.test_selector_refit import policy


def test_common_utility_pilot_preserves_guards_and_has_separate_identity():
    previous = load_intervention_config(
        Path("configs/routing/intervention_metric_selector_fit.yaml")
    )
    pilot = load_intervention_config(
        Path("configs/routing/intervention_metric_common_utility.yaml")
    )
    expected = replace(
        previous,
        stereo_utility_multiplier=1.0,
        metric_quality=dict(previous.metric_quality, geometry_priority=1.5),
    )
    assert pilot == expected
    assert pilot.config_hash != previous.config_hash
    assert not pilot.selector_gain_only_refinement


def test_geometry_weight_changes_safe_tradeoff_without_rewarding_harm_or_unknown():
    ids = ("dataset_1/keyframe_1/a", "dataset_2/keyframe_1/b")
    records = []
    vectors = [
        (
            vector(local_depth_mae_gain_mm=2),
            vector(
                left_psnr_gain_db=0.1,
                right_psnr_gain_db=0.1,
                left_ssim_gain=0.001,
                right_ssim_gain=0.001,
            ),
        ),
        (vector(local_depth_mae_gain_mm=100, left_psnr_gain_db=-1), vector(left_psnr_gain_db=0.01)),
    ]
    for sid, pair in zip(ids, vectors, strict=True):
        records.append(
            {
                "sample_id": sid,
                "interventions": [
                    {
                        "actions": [
                            {"selector_label_available": True, "metric_changes": None},
                            *[
                                {"selector_label_available": True, "metric_changes": v}
                                for v in pair
                            ],
                        ]
                    }
                ],
            }
        )
    records.append(
        {
            "sample_id": "dataset_3/keyframe_1/c",
            "interventions": [
                {
                    "actions": [
                        {"selector_label_available": True},
                        {"selector_label_available": False},
                        {"selector_label_available": False},
                    ]
                }
            ],
        }
    )
    x = torch.zeros(2, len(BALANCED_FEATURE_NAMES))
    y = torch.zeros(2, 3)
    table = SelectorOutcomes({"collection_role": "train"}, records, x, y, y, y, ids)
    cfg = replace(
        policy(),
        stereo_utility_multiplier=1.0,
        metric_quality=dict(
            policy().metric_quality,
            geometry_priority=1.0,
            geometry_weight=1.0,
            local_geometry_weight=1.0,
        ),
    )
    one = quality_oracle_tradeoff(table, cfg)
    four = quality_oracle_tradeoff(
        table, replace(cfg, metric_quality=dict(cfg.metric_quality, geometry_priority=4.0))
    )
    assert one["actions"] == {"GAUSSIAN": 2}
    assert four["actions"] == {"STEREO": 1, "GAUSSIAN": 1}
    assert one["excluded_images_without_complete_repair_labels"] == 1
    assert four["mean_changes_keyframe_macro"]["left_psnr_gain_db"] >= 0
    assert four["mean_changes_keyframe_macro"]["local_depth_mae_gain_mm"] == 1
