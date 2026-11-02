"""Off-axis projection terms: a point must land on fx*X/Z + cx for the integer-centre pixel convention."""

import torch

from reliable_endo_gs.baseline.finetune.projection import off_axis_terms


def _symmetric(fx_n: float, fy_n: float, near: float, far: float) -> torch.Tensor:
    # upstream layout: row0 = [2 fx_n, 0, 0, 0] (focal part), row1 = [0, 2 fy_n, 0, 0], depth rows, w = z
    m = torch.zeros(1, 4, 4)
    m[:, 0, 0], m[:, 1, 1] = 2 * fx_n, 2 * fy_n
    m[:, 2, 2], m[:, 2, 3] = far / (far - near), -(far * near) / (far - near)
    m[:, 3, 2] = 1.0
    return m


def test_projected_pixel_matches_the_pinhole_model_with_an_off_centre_principal_point():
    w, h = 640, 512
    fx, fy, cx, cy = (
        560.2,
        560.2,
        320.84,
        260.11,
    )  # integer-centre pixel convention (A00 datasets 1-3)
    k = torch.tensor(
        [[[(fx) / w, 0.0, (cx + 0.5) / w], [0.0, fy / h, (cy + 0.5) / h], [0.0, 0.0, 1.0]]]
    )
    m = off_axis_terms(_symmetric(fx / w, fy / h, 0.03, 0.16), k)
    point = torch.tensor([0.011, -0.007, 0.08, 1.0])
    clip = m[0] @ point
    ndc = clip[:2] / clip[3]
    pixel = (ndc + 1.0) * torch.tensor([w, h]) / 2.0 - 0.5  # rasterizer: ((ndc + 1) * size - 1) / 2
    expected = torch.tensor([fx * 0.011 / 0.08 + cx, fy * -0.007 / 0.08 + cy])
    assert torch.allclose(pixel, expected, atol=1e-3)


def test_centred_camera_is_unchanged_and_batch_must_match():
    w, h = 640, 512
    k = torch.tensor([[[500.0 / w, 0.0, 0.5], [0.0, 500.0 / h, 0.5], [0.0, 0.0, 1.0]]])
    base = _symmetric(500.0 / w, 500.0 / h, 0.03, 0.16)
    assert torch.equal(off_axis_terms(base, k), base)
    try:
        off_axis_terms(base.repeat(2, 1, 1), k)
    except ValueError:
        pass
    else:
        raise AssertionError("batch mismatch must raise")
