"""Synthetic checks for benchmark metric definitions and evaluation isolation."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from reliable_endo_gs.cli.main import build_parser
from reliable_endo_gs.contracts.deployment import RectifiedStereoCamera
from reliable_endo_gs.evaluation.a04_benchmark import select_samples, summarize_metrics
from reliable_endo_gs.evaluation.benchmark_geometry import cloud_metrics, evaluate_geometry
from reliable_endo_gs.evaluation.benchmark_protocol import BenchmarkProtocol
from reliable_endo_gs.evaluation.perceptual import LPIPSAlexNet
from reliable_endo_gs.evaluation.rendering import windowed_ssim
from reliable_endo_gs.evaluation.stereo import MetricInputError


def _camera() -> RectifiedStereoCamera:
    k = torch.tensor([[[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]])
    t = torch.eye(4)[None].clone()
    t[:, 0, 3] = -0.1
    return RectifiedStereoCamera(k, k.clone(), t, True)


def test_ssim_matches_pinned_upstream_formula() -> None:
    import importlib.util

    # Read-only parity with pinned source; never add baseline imports to core.
    path = Path(__file__).resolve().parents[2] / "third_party/endo_e2e_gs/lib/loss.py"
    spec = importlib.util.spec_from_file_location("benchmark_test_upstream_loss", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gen = torch.Generator().manual_seed(41)
    x, y = torch.rand(1, 3, 32, 40, generator=gen), torch.rand(1, 3, 32, 40, generator=gen)
    assert windowed_ssim(x, y).value == pytest.approx(float(module.ssim(x, y)), abs=2e-6)
    assert windowed_ssim(x, x).value == pytest.approx(1.0)
    with pytest.raises(MetricInputError):
        windowed_ssim(x + 2, y)
    with pytest.raises(MetricInputError):
        windowed_ssim(x * float("nan"), y)


def test_geometry_units_invalid_prediction_and_no_render_depth() -> None:
    target = torch.full((1, 1, 5, 5), 10.0)
    predicted = torch.full_like(target, 20.0)
    mask = torch.ones_like(target, dtype=torch.bool)
    before = target.clone()
    metrics, details = evaluate_geometry(
        predicted, target, mask, mask, _camera(), cloud_max_points=25, cloud_chunk_size=4
    )
    assert metrics["disparity_epe"].value == 10
    assert metrics["stereo_depth_mae_mm"].value == pytest.approx(500)
    assert metrics["stereo_depth_rmse_mm"].value == pytest.approx(500)
    assert metrics["stereo_depth_absrel"].value == pytest.approx(0.5)
    assert metrics["rendered_depth_mae_mm"].value is None
    assert details["gt_valid_pixels"] == 25 and torch.equal(target, before)
    predicted[0, 0, 0, 0] = float("nan")
    predicted[0, 0, 0, 1] = 10
    metrics, _ = evaluate_geometry(
        predicted, target, mask, mask, _camera(), cloud_max_points=25, cloud_chunk_size=4
    )
    assert metrics["geometry_coverage"].value == pytest.approx(24 / 25)
    assert metrics["disparity_1px_accuracy"].value == pytest.approx(1 / 25)


def test_principal_point_correction_owned_by_rectified_geometry() -> None:
    c = _camera()
    kr = c.K_R.clone()
    kr[:, 0, 2] -= 2
    camera = RectifiedStereoCamera(c.K_L, kr, c.T_R_from_L, True)
    target = torch.full((1, 1, 5, 5), 12.0)
    mask = torch.ones_like(target, dtype=torch.bool)
    metrics, _ = evaluate_geometry(
        target, target, mask, mask, camera, cloud_max_points=25, cloud_chunk_size=5
    )
    assert metrics["stereo_depth_mae_mm"].value == 0


def test_cloud_unsquared_mm_and_chunk_invariance() -> None:
    p = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    t = p + torch.tensor([0.0, 0.0, 0.01])
    normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    for chunk in (1, 2, 8):
        result = cloud_metrics(
            p,
            t,
            max_points=20,
            chunk_size=chunk,
            predicted_normals=normals,
            target_normals=-normals,
        )
        assert result["chamfer_mm"].value == pytest.approx(10.0, abs=0.001)
        assert result["normal_consistency"].value == pytest.approx(1.0)
    empty = cloud_metrics(
        p[:0], t, max_points=20, chunk_size=1, predicted_normals=normals[:0], target_normals=normals
    )
    assert empty["chamfer_mm"].value is None and empty["normal_consistency"].value is None


def test_empty_gt_has_unavailable_geometry_not_zero_error() -> None:
    d = torch.ones(1, 1, 5, 5)
    m = torch.zeros_like(d, dtype=torch.bool)
    metrics, details = evaluate_geometry(
        d, d, m, ~m, _camera(), cloud_max_points=25, cloud_chunk_size=4
    )
    assert metrics["disparity_epe"].value is None
    assert metrics["stereo_depth_rmse_mm"].value is None
    assert details["gt_valid_pixels"] == 0


def test_protocol_rejects_unknown_fields_training_role_and_bad_counts(tmp_path: Path) -> None:
    for text in ("data_role: final_train", "warmup: -1", "extra: 1", "resolution_hw: [16,32]"):
        p = tmp_path / "protocol.yaml"
        p.write_text(text)
        with pytest.raises(ValueError):
            BenchmarkProtocol.load(p)


def test_selection_is_final_test_unique_and_spans_sequences() -> None:
    entries = [
        SimpleNamespace(sample_id=f"dataset_{d}/keyframe_{k}/frame_data{i:06}", data_role=role)
        for d, role in ((1, "final_train"), (8, "final_test"), (9, "final_test"))
        for k in range(2)
        for i in range(10)
    ]
    adapter = SimpleNamespace(manifest=SimpleNamespace(entries=entries))
    ids = select_samples(adapter, max_samples=8)
    assert len(ids) == len(set(ids)) == 8
    assert len({i.rsplit("/", 1)[0] for i in ids}) == 4
    assert all(i.endswith(("000000", "000009")) for i in ids)
    with pytest.raises(ValueError):
        select_samples(adapter, sample_ids=[entries[0].sample_id])


def test_aggregation_frame_mean_and_pooled_rmse_are_distinct() -> None:
    rows = [
        {
            "metrics": {
                "rmse_mm": {"value": v, "valid_count": n, "total_count": n, "definition": "test"}
            }
        }
        for v, n in ((1.0, 1), (3.0, 3))
    ]
    value = summarize_metrics(rows)["rmse_mm"]
    assert value["frame_mean"] == 2
    assert value["valid_weighted"] == pytest.approx(7**0.5)


def test_cli_is_independent_from_train_harness() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "a04",
            "--checkpoint",
            "ckpt.pt",
            "--data-root",
            "data",
            "--manifest",
            "manifest.json",
            "--output-dir",
            "out",
            "--lpips-backbone",
            "alexnet.pth",
            "--lpips-calibration",
            "alex.pth",
            "--max-samples",
            "8",
        ]
    )
    assert args.command == "benchmark" and args.max_samples == 8
    assert not hasattr(args, "epochs") and not hasattr(args, "output_checkpoint")


def test_lpips_requires_real_local_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LPIPSAlexNet(tmp_path / "missing", tmp_path / "missing2")


def test_lpips_rejects_random_or_unpinned_artifacts(tmp_path: Path) -> None:
    backbone, calibration = tmp_path / "backbone", tmp_path / "calibration"
    backbone.write_bytes(b"not pretrained AlexNet")
    calibration.write_bytes(b"not official LPIPS")
    with pytest.raises(MetricInputError, match="pinned"):
        LPIPSAlexNet(backbone, calibration)


def test_gaussian_cloud_restricts_to_gt_source_pixels() -> None:
    d = torch.full((1, 1, 5, 5), 10.0)
    mask = torch.ones_like(d, dtype=torch.bool)
    centres = torch.tensor([[[0.0, 0.0, 1.0], [20.0, 20.0, 30.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    indices = torch.tensor([[12, -1]])
    metrics, _ = evaluate_geometry(
        d,
        d,
        mask,
        mask,
        _camera(),
        cloud_max_points=25,
        cloud_chunk_size=5,
        gaussian_centres=centres,
        gaussian_valid=valid,
        gaussian_source_index=indices,
    )
    # The unmapped Gaussian is excluded; GT still has all 25 valid points.
    assert metrics["gaussian_centres_chamfer_mm"].total_count == 26
