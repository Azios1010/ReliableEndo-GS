"""The opt-in right-view visibility mask of the student loss: identity of existing configs is unchanged."""

from __future__ import annotations

import pytest

from reliable_endo_gs.training.student_config import StudentLearningConfig


def test_default_config_identity_is_unchanged_by_the_flag() -> None:
    config = StudentLearningConfig()
    assert config.right_visibility_mask is False
    assert "right_visibility_mask" not in config.to_dict()


def test_enabled_flag_is_part_of_the_identity() -> None:
    config = StudentLearningConfig(right_visibility_mask=True)
    assert config.to_dict()["right_visibility_mask"] is True
    assert config.config_hash != StudentLearningConfig().config_hash


def test_flag_must_be_a_boolean() -> None:
    with pytest.raises(ValueError):
        StudentLearningConfig(right_visibility_mask=1)  # type: ignore[arg-type]
