"""Batched validation keeps fail-fast semantics: same exception, same message, same order."""

from dataclasses import replace

import pytest
import torch

from reliable_endo_gs.rendering.field import FieldRenderRequest
from reliable_endo_gs.rendering.production import NativeCameraBatch
from tests.architecture.test_deployment_foundation import camera, field


def _valid_field():
    f = field()  # primitive 1 is masked out and holds NaN / zero rotation on purpose
    return f


def test_masked_out_garbage_is_still_accepted():
    FieldRenderRequest(_valid_field(), camera(), (3, 5))


@pytest.mark.parametrize(
    ("name", "bad", "message"),
    [
        (
            "means3d",
            torch.tensor([[[float("nan"), 0, 2], [0.0, 0, 0]]], dtype=torch.float64),
            "finite",
        ),
        (
            "colors",
            torch.tensor([[[float("inf"), 1, 1], [1.0, 1, 1]]], dtype=torch.float64),
            "finite",
        ),
        (
            "rotations",
            torch.tensor([[[float("nan"), 0, 0, 0], [0.0, 0, 0, 0]]], dtype=torch.float64),
            "finite",
        ),
        (
            "scales",
            torch.tensor([[[0.01, 0.01, -0.01], [1.0, 1, 1]]], dtype=torch.float64),
            "positive",
        ),
        (
            "colors",
            torch.tensor([[[1.5, 1, 1], [1.0, 1, 1]]], dtype=torch.float64),
            r"RGB must be in \[0,1\]",
        ),
        (
            "opacities",
            torch.tensor([[[1.5], [0.5]]], dtype=torch.float64),
            r"alpha must be in \[0,1\]",
        ),
        (
            "opacities",
            torch.tensor([[[-0.1], [0.5]]], dtype=torch.float64),
            r"alpha must be in \[0,1\]",
        ),
        (
            "rotations",
            torch.tensor([[[2.0, 0, 0, 0], [0.0, 0, 0, 0]]], dtype=torch.float64),
            "normalized",
        ),
    ],
)
def test_value_violations_on_valid_primitives_raise_the_same_error(name, bad, message):
    with pytest.raises(ValueError, match=message):
        FieldRenderRequest(replace(_valid_field(), **{name: bad}), camera(), (3, 5))


def test_first_failing_check_in_the_original_order_wins():
    f = _valid_field()
    both = replace(
        f,
        means3d=torch.tensor([[[float("nan"), 0, 2], [0.0, 0, 0]]], dtype=torch.float64),
        scales=-f.scales,
    )
    with pytest.raises(ValueError, match="finite"):  # finite is tested before positivity
        FieldRenderRequest(both, camera(), (3, 5))


@pytest.mark.parametrize(
    ("norm", "accepted"), [(1.0 + 1.9e-5, True), (1.0 + 2.3e-5, False), (1.0 - 2.3e-5, False)]
)
def test_rotation_norm_tolerance_matches_allclose_bounds(norm, accepted):
    f = _valid_field()
    rotations = torch.tensor([[[norm, 0, 0, 0], [0.0, 0, 0, 0]]], dtype=torch.float64)
    request = replace(f, rotations=rotations)
    if accepted:
        FieldRenderRequest(request, camera(), (3, 5))
    else:
        with pytest.raises(ValueError, match="normalized"):
            FieldRenderRequest(request, camera(), (3, 5))


def _native_camera(**overrides):
    values = {
        "viewmatrix": torch.eye(4).repeat(1, 1, 1),
        "projmatrix": torch.eye(4).repeat(1, 1, 1),
        "campos": torch.zeros(1, 3),
        "tanfovx": torch.full((1,), 0.5),
        "tanfovy": torch.full((1,), 0.5),
        "background": torch.zeros(1, 3),
    }
    values.update(overrides)
    return NativeCameraBatch(**values)


def test_native_camera_batch_reports_the_first_nonfinite_tensor_in_declared_order():
    _native_camera()
    with pytest.raises(ValueError, match="campos must be finite"):
        _native_camera(campos=torch.tensor([[0.0, float("nan"), 0.0]]))
    with pytest.raises(ValueError, match="viewmatrix must be finite"):
        _native_camera(
            tanfovx=torch.full((1,), float("inf")),
            viewmatrix=torch.full((1, 4, 4), float("nan")),
        )
