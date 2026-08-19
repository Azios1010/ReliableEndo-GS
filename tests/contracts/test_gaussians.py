"""GaussianField structural and semantic-separation tests."""

import pytest
import torch

from reliable_endo_gs.contracts import GaussianField


def _field(**overrides: object) -> GaussianField:
    values: dict[str, object] = {
        "means3d": torch.zeros(2, 6, 3),
        "colors": torch.zeros(2, 6, 3),
        "rotations": torch.zeros(2, 6, 4),
        "scales": torch.ones(2, 6, 3),
        "opacities": torch.ones(2, 6, 1),
        "valid_mask": torch.ones(2, 6, dtype=torch.bool),
    }
    values.update(overrides)
    return GaussianField(**values)  # type: ignore[arg-type]


def test_valid_field_keeps_covariances_semantically_separate() -> None:
    surface = torch.eye(3).repeat(2, 6, 1, 1)
    center = torch.full((2, 6, 3, 3), 2.0)
    effective = torch.full((2, 6, 3, 3), 3.0)
    field = _field(cov_surface=surface, cov_center=center, cov_effective=effective)

    assert field.cov_surface is surface
    assert field.cov_center is center
    assert field.cov_effective is effective
    assert field.gaussian_count == 6
    assert field.SCHEMA_NAME == "gaussian_field"


def test_field_does_not_derive_missing_covariances(gaussian_field: GaussianField) -> None:
    assert gaussian_field.cov_surface is None
    assert gaussian_field.cov_center is None
    assert gaussian_field.cov_effective is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("means3d", torch.zeros(2, 6, 4), "means3d must have shape"),
        ("colors", torch.zeros(2, 5, 3), "colors must have shape"),
        ("rotations", torch.zeros(2, 6, 3), "rotations must have shape"),
        ("scales", torch.zeros(2, 6, 4), "scales must have shape"),
        ("opacities", torch.zeros(2, 6), "opacities must have rank"),
        ("valid_mask", torch.ones(2, 6), "dtype torch.bool"),
        ("cov_center", torch.zeros(2, 6, 3, 2), "cov_center must have shape"),
    ],
)
def test_field_rejects_invalid_shapes_and_mask_dtype(
    field: str, value: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _field(**{field: value})


def test_field_rejects_dtype_and_device_mismatch() -> None:
    with pytest.raises(TypeError, match="colors must have the same dtype"):
        _field(colors=torch.zeros(2, 6, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match="cov_surface must be on the same device"):
        _field(cov_surface=torch.zeros(2, 6, 3, 3, device="meta"))
