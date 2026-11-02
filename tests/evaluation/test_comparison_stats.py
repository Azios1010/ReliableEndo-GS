import numpy as np

from reliable_endo_gs.evaluation.comparison_stats import (
    cluster_bootstrap,
    frame_bootstrap,
    higher_is_better,
    kd_vs_gt_verdict,
    paired_summary,
)


def _ids(dataset, keyframes=4, per=5):
    return [
        f"{dataset}/keyframe_{k}/frame_data{i:06d}" for k in range(keyframes) for i in range(per)
    ]


def test_polarity_and_intervals():
    assert (
        higher_is_better("left_psnr")
        and higher_is_better("disparity_1px_accuracy")
        and not higher_is_better("right_lpips_alex")
    )
    delta = np.full(50, 0.5) + np.random.default_rng(0).normal(0, 0.05, 50)
    mean, lo, hi = frame_bootstrap(delta, 1000)
    assert lo < mean < hi and lo > 0.4
    clusters = [f"k{i % 5}" for i in range(50)]
    assert cluster_bootstrap(delta, clusters, 1000)[1] > 0.3


def test_paired_summary_direction():
    ids = _ids("dataset_8")
    a = {i: 30.0 + 0.1 * (n % 3) for n, i in enumerate(ids)}
    b = {i: 28.0 for i in ids}
    s = paired_summary(a, b, "left_psnr", 500)
    assert s["a_better_frame_ci"] and s["win_rate_a"] == 1.0
    worse = paired_summary(a, b, "disparity_epe", 500)  # lower is better: a is larger => worse
    assert not worse["a_better_frame_ci"] and worse["win_rate_a"] == 0.0


def test_kd_vs_gt_rule():
    gt, kd = {}, {}
    for dataset in ("dataset_8", "dataset_9"):
        for i in _ids(dataset):
            gt[i], kd[i] = 4.0, 2.4  # r = 1.67 everywhere
    assert kd_vs_gt_verdict(gt, kd, gt, kd)["verdict"] == "KD better"
    near = {i: 4.0 for i in gt}
    same = {i: 3.8 for i in gt}
    assert kd_vs_gt_verdict(near, same, near, same)["verdict"] == "no meaningful difference"
    mid = {i: 3.3 for i in gt}  # r = 1.21: neither better nor no-difference
    assert kd_vs_gt_verdict(near, mid, near, mid)["verdict"] == "mixed"
