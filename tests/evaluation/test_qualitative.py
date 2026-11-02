from __future__ import annotations

import pytest

from reliable_endo_gs.evaluation.qualitative import file_stem


def test_file_stem_flattens_a_sample_id() -> None:
    assert (
        file_stem("dataset_8/keyframe_0/frame_data000307")
        == "dataset_8__keyframe_0__frame_data000307"
    )


@pytest.mark.parametrize("bad", ["dataset_8/frame", "../x/y/z", "a/b/c/d"])
def test_file_stem_rejects_unexpected_ids(bad: str) -> None:
    with pytest.raises(ValueError):
        file_stem(bad)
