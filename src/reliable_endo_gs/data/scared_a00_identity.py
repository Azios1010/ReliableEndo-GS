"""Reviewed compatibility for immutable A00 numeric outputs.

The empty-target remediation preserves numeric outputs for samples with GT;
the correspondence remediation preserves both positive and empty GT outputs.
Only the exact before/after processor pairs below may share those outputs.
Unknown code changes require regeneration or a new compatibility review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

POSITIVE_GT_PROCESSOR_PAIR = (
    "2b00968de6cbfaf46ca0635b10e017ebe8819f655e165c57d5f2e0438ebc6278",
    "742b568f6a193c2c0889058746e3f947fd098195fcae518379014912604ab00d",
)
EMPTY_GT_PROCESSOR_PAIR = (
    "4c736d25fae1eb172ea8c4f1c1e9a4ad7a62f46b19b8af48b574cd2366d43243",
    "e18cdb13b4532029918b20fa4154128f338e777f16b49a85f9d23150775c8378",
)

CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR = (
    "b60aaf12325e3e5d54c24bdd39f8ad36fa756e49a793c4e1edbab394f070c902",
    "7e73d55412b17166ce26b7367edf786ff194233041d4c952237cade6c87bfe7e",
)

TYPED_FORMATTED_PROCESSOR_PAIR = (
    "fcb3190893f3d75bef23ba068824ca57ed1065464396741eb962fc289650439d",
    "7f60dd5a7197d0c44ceee27db3bc96570569e10aabaed685d97bbb4b233acf3c",
)


def a00_processor_compatibility(
    recorded_pair: tuple[object, object],
    current_pair: tuple[str, str],
    metric_summaries: Sequence[Mapping[str, object]],
) -> str | None:
    """Return an explicit compatibility decision; never rewrite producer hashes."""
    if recorded_pair == current_pair:
        return "exact_code_identity"
    if current_pair == TYPED_FORMATTED_PROCESSOR_PAIR:
        # Reviewed against CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR:
        # only formatting, type annotations/imports and identity typing.cast
        # calls changed. The executable AST is identical after removing those
        # typing-only constructs. Keep historical producer hashes immutable.
        if recorded_pair == CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR:
            return "reviewed_typing_format_compatibility"
        current_pair = CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR
    if (
        recorded_pair == EMPTY_GT_PROCESSOR_PAIR
        and current_pair == CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR
    ):
        # Only evidence correspondence changed. Rectification, disparity/GT
        # construction, masks, streaming and resume verification are unchanged.
        # Empty-target outputs from the reviewed empty-GT producer stay valid.
        if not metric_summaries:
            return None
        for summary in metric_summaries:
            count = summary.get("valid_pixel_count")
            if type(count) is not int or count < 0:
                return None
            if count == 0:
                if summary.get("status") != "unavailable" or any(
                    key not in summary or summary[key] is not None
                    for key in ("q_vs_a00_abs_error_mm", "q_vs_metric_gt_abs_error_mm")
                ):
                    return None
            elif summary.get("status") == "unavailable":
                return None
        return "reviewed_calibration_evidence_compatibility"
    if recorded_pair != POSITIVE_GT_PROCESSOR_PAIR or current_pair not in (
        EMPTY_GT_PROCESSOR_PAIR,
        CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR,
    ):
        return None
    if not metric_summaries or any(
        type(summary.get("valid_pixel_count")) is not int
        or cast(int, summary["valid_pixel_count"]) <= 0
        or summary.get("status") == "unavailable"
        for summary in metric_summaries
    ):
        return None
    return "reviewed_positive_gt_compatibility"
