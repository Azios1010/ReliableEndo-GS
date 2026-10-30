"""Opt-in inference transforms: numerics, identity, exact undo and CUDA-graph exactness."""

import pytest
import torch
from torch import nn

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera
from reliable_endo_gs.reconstruction.deployment import (
    optimize_student_for_inference,
    use_parallel_group_norm,
)
from reliable_endo_gs.reconstruction.stereo_features import ParallelMomentsGroupNorm
from reliable_endo_gs.reconstruction.student import StereoStudent
from tests.training.test_student_architecture import _stereo_batch


class _Holder(nn.Module):
    def __init__(self, student: StereoStudent) -> None:
        super().__init__()
        self.stereo_student = student


def _frozen_student(seed: int = 3) -> StereoStudent:
    torch.manual_seed(seed)
    return StereoStudent(8, 8, architecture="multiscale").eval().requires_grad_(False)


@pytest.mark.parametrize(
    ("shape", "shift"), [((1, 32, 24, 30), 0.0), ((2, 8, 16, 20), 40.0), ((1, 5, 7, 11), -25.0)]
)
def test_parallel_group_norm_matches_native_values_and_gradients(shape, shift):
    torch.manual_seed(1)
    native = nn.GroupNorm(1, shape[1])
    native.weight.data.normal_(1.0, 0.3)
    native.bias.data.normal_(0.0, 0.3)
    fast = ParallelMomentsGroupNorm.from_group_norm(native)
    x = (torch.randn(*shape) * 2.5 + shift).requires_grad_(True)
    y_native, y_fast = native(x), fast(x)
    # Documented float32 approximation: agreement to rounding, not bitwise.
    assert torch.allclose(y_native, y_fast, atol=2e-5, rtol=1e-5)
    upstream = torch.randn_like(y_native)
    native_grads = torch.autograd.grad(y_native, (x, native.weight, native.bias), upstream)
    fast_grads = torch.autograd.grad(y_fast, (x, fast.weight, fast.bias), upstream)
    for expected, actual in zip(native_grads, fast_grads, strict=True):
        assert torch.allclose(expected, actual, atol=5e-4, rtol=1e-3)


def test_parallel_group_norm_keeps_state_modes_and_rejects_other_shapes():
    native = nn.GroupNorm(1, 6, eps=1e-3)
    native.weight.data.normal_()
    native.bias.data.normal_()
    native.bias.requires_grad_(False)
    native.eval()
    fast = ParallelMomentsGroupNorm.from_group_norm(native)
    assert list(fast.state_dict()) == list(native.state_dict()) == ["weight", "bias"]
    assert all(
        torch.equal(a, b)
        for a, b in zip(native.state_dict().values(), fast.state_dict().values(), strict=True)
    )
    assert not fast.training and fast.eps == 1e-3 and fast.bias.requires_grad is False
    plain = ParallelMomentsGroupNorm.from_group_norm(nn.GroupNorm(1, 4, affine=False))
    x = torch.randn(2, 4, 5, 5)
    assert torch.allclose(plain(x), nn.GroupNorm(1, 4, affine=False)(x), atol=1e-5)
    with pytest.raises(ValueError, match="exactly one group"):
        ParallelMomentsGroupNorm.from_group_norm(nn.GroupNorm(2, 4))
    with pytest.raises(ValueError, match="channel count"):
        fast(torch.randn(1, 5, 4, 4))


def test_parallel_group_norm_reduces_low_precision_input_in_float32():
    torch.manual_seed(2)
    native = nn.GroupNorm(1, 8)
    fast = ParallelMomentsGroupNorm.from_group_norm(native)
    x = torch.randn(1, 8, 12, 12) * 3 + 5
    expected = native(x)
    actual = fast(x.to(torch.bfloat16))
    assert actual.dtype is torch.bfloat16
    assert torch.allclose(actual.float(), expected, atol=0.1, rtol=0.05)


def test_swap_only_single_group_layers_and_undo_is_bit_exact():
    student = _frozen_student()
    batch = _stereo_batch(hw=(19, 27))
    before = student(batch)
    keys = list(student.state_dict())
    singles = sum(type(m) is nn.GroupNorm for m in student.modules())
    count, undo = use_parallel_group_norm(student)
    assert count == singles > 0
    assert not any(type(m) is nn.GroupNorm for m in student.modules())
    assert list(student.state_dict()) == keys
    assert torch.allclose(student(batch), before, atol=1e-3)
    undo()
    assert sum(type(m) is nn.GroupNorm for m in student.modules()) == singles
    assert torch.equal(student(batch), before)


def test_optimize_record_reports_exactness_and_undoes_everything():
    holder = _Holder(_frozen_student(5))
    batch = _stereo_batch(hw=(19, 27))
    before = holder.stereo_student(batch)
    idle = optimize_student_for_inference(holder)
    assert idle.to_dict()["bit_exact"] is True and idle.numerical_approximations == ()
    applied = optimize_student_for_inference(holder, parallel_group_norm=True)
    info = applied.to_dict()
    assert info["bit_exact"] is False and info["numerical_approximations"] == [
        "parallel_moments_group_norm"
    ]
    assert info["parallel_group_norm_layers"] > 0
    applied.undo()
    assert torch.equal(holder.stereo_student(batch), before)
    with pytest.raises(TypeError, match="StereoStudent"):
        optimize_student_for_inference(nn.Linear(1, 1))


def test_cuda_graph_request_rejects_cpu_and_unfrozen_students():
    holder = _Holder(_frozen_student())
    batch = _stereo_batch(hw=(19, 27))
    with pytest.raises(ValueError, match="CUDA tensors"):
        optimize_student_for_inference(holder, stereo_cuda_graph=True, example_batches=[batch])
    with pytest.raises(ValueError, match="at least one example"):
        optimize_student_for_inference(holder, stereo_cuda_graph=True)


def _to_cuda(batch: DeploymentStereoBatch) -> DeploymentStereoBatch:
    c = batch.camera
    camera = RectifiedStereoCamera(c.K_L.cuda(), c.K_R.cuda(), c.T_R_from_L.cuda(), c.rectified)
    return DeploymentStereoBatch(
        batch.left.cuda(), batch.right.cuda(), camera, batch.support.cuda()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a CUDA device")
def test_stereo_cuda_graph_is_bit_identical_and_falls_back_to_eager():
    holder = _Holder(_frozen_student(7).cuda())
    student = holder.stereo_student
    grid = (32, 40)
    batches = [_to_cuda(_stereo_batch(hw=grid)) for _ in range(3)]
    other_grid = _to_cuda(_stereo_batch(hw=(24, 30)))
    with torch.no_grad():
        eager = [student(b) for b in batches]
        eager_other = student(other_grid)
        eager_evidence = student.predict_with_evidence(batches[0], include_evidence=True)
        record = optimize_student_for_inference(
            holder, stereo_cuda_graph=True, example_batches=[batches[0]]
        )
        assert record.stereo_cuda_graph_grids == (grid,) and record.to_dict()["bit_exact"] is True
        for batch, expected in zip(batches, eager, strict=True):
            assert torch.equal(student(batch), expected)  # replay with new frames, every time
        assert torch.equal(student(other_grid), eager_other)  # uncaptured grid -> eager
        disparity, evidence = student.predict_with_evidence(batches[0], include_evidence=True)
        assert (
            torch.equal(disparity, eager_evidence[0])
            and evidence.keys() == eager_evidence[1].keys()
        )
    record.undo()
    assert "predict_with_evidence" not in student.__dict__


class _CheckedHolder(_Holder):
    """Forward that registers one validation which fails, then keeps computing."""

    def __init__(self, student: StereoStudent) -> None:
        super().__init__(student)
        self.completed = False

    def forward(self, ok: bool) -> str:
        from reliable_endo_gs.utils.readback import raise_first_failure

        raise_first_failure([(torch.tensor(ok), lambda: ValueError("invalid frame"))])
        self.completed = True
        return "result"


def test_deferred_validation_option_wraps_forward_defers_checks_and_undoes():
    holder = _CheckedHolder(_frozen_student())
    record = optimize_student_for_inference(holder, deferred_validation=True)
    assert record.to_dict()["deferred_validation"] is True and record.to_dict()["bit_exact"] is True
    assert holder(True) == "result"
    with pytest.raises(ValueError, match="invalid frame"):
        holder(False)
    assert holder.completed  # the body ran to the end; the error surfaced at the end of the call
    record.undo()
    holder.completed = False
    with pytest.raises(ValueError, match="invalid frame"):
        holder(False)
    assert not holder.completed  # immediate again after undo
