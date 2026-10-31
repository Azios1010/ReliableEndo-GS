"""The registered success bar: seeded bootstrap and one failing case per condition."""

import numpy as np
import pytest

from reliable_endo_gs.evaluation.confirmatory import (
    FrameRows,
    bootstrap_interval,
    cluster_bootstrap_interval,
    confirmatory_verdict,
    dataset_verdict,
)


def _rows(
    n=40, gain=1.0, ssim=0.02, lpips=-0.03, ratio=1.0, keyframes=("k0", "k1", "k2", "k3"), seed=0
):
    rng = np.random.default_rng(seed)
    kf = [keyframes[i % len(keyframes)] for i in range(n)]
    base = 0.01 * np.ones(n)
    return FrameRows(
        kf,
        gain + 0.1 * rng.standard_normal(n),
        ssim + 0.001 * rng.standard_normal(n),
        lpips + 0.001 * rng.standard_normal(n),
        base * ratio,
        base,
    )


def test_bootstrap_is_seeded_and_brackets_the_mean():
    v = np.random.default_rng(1).standard_normal(200) + 2
    a, b = bootstrap_interval(v, seed=3), bootstrap_interval(v, seed=3)
    assert a == b and a[1] < a[0] < a[2] and bootstrap_interval(v, seed=4) != a
    c = cluster_bootstrap_interval(v, ["x"] * 100 + ["y"] * 100, seed=3)
    assert c[1] < c[0] < c[2]
    with pytest.raises(ValueError):
        bootstrap_interval(np.array([]))


def test_a_clear_gain_on_both_datasets_is_supported():
    out = confirmatory_verdict({"dataset_8": _rows(seed=1), "dataset_9": _rows(seed=2)})
    assert out["verdict"] == "supported" and out["failing"] == {"dataset_8": [], "dataset_9": []}


@pytest.mark.parametrize(
    ("kwargs", "condition"),
    [
        ({"gain": 0.3}, "P1_mean_gain_at_least_0.5dB"),
        ({"ssim": -0.02}, "N1_ssim_not_worse"),
        ({"lpips": 0.02}, "N2_lpips_not_worse"),
        ({"ratio": 1.2}, "N3_last80_columns_within_1.05x"),
    ],
)
def test_each_registered_condition_can_fail_on_its_own(kwargs, condition):
    out = confirmatory_verdict({"dataset_8": _rows(**kwargs), "dataset_9": _rows(seed=2)})
    assert (
        out["verdict"] == "not supported"
        and condition in out["failing"]["dataset_8"]
        and out["failing"]["dataset_9"] == []
    )


def test_positive_share_keyframe_and_finiteness_conditions():
    rows = _rows(n=40)
    gain = rows.right_psnr_gain.copy()
    gain[:10] = -0.5  # 25% of the frames negative: the mean stays above 0.5 dB
    worse = FrameRows(
        rows.keyframe,
        gain,
        rows.right_ssim_gain,
        rows.right_lpips_change,
        rows.last80_mse_refined,
        rows.last80_mse_stop,
    )
    checks = dataset_verdict(worse)["checks"]
    assert not checks["P3_positive_on_80pct_of_frames"]
    bad_keyframe = np.where(np.array(rows.keyframe) == "k0", -2.0, rows.right_psnr_gain)
    checks = dataset_verdict(
        FrameRows(
            rows.keyframe,
            bad_keyframe,
            rows.right_ssim_gain,
            rows.right_lpips_change,
            rows.last80_mse_refined,
            rows.last80_mse_stop,
        )
    )["checks"]
    assert not checks["P4_every_keyframe_mean_positive"]
    nan = rows.right_ssim_gain.copy()
    nan[0] = np.nan
    assert not dataset_verdict(
        FrameRows(
            rows.keyframe,
            rows.right_psnr_gain,
            nan,
            rows.right_lpips_change,
            rows.last80_mse_refined,
            rows.last80_mse_stop,
        )
    )["checks"]["N4_finite"]


def test_both_final_test_datasets_are_required_and_shapes_checked():
    with pytest.raises(ValueError, match="missing"):
        confirmatory_verdict({"dataset_8": _rows()})
    with pytest.raises(ValueError):
        FrameRows(["a"], np.zeros(2), np.zeros(1), np.zeros(1), np.ones(1), np.ones(1))
