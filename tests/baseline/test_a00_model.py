"""A00 adaptation contracts independent of native CUDA and upstream weights."""

import torch
from torch import nn

from reliable_endo_gs.baseline.a00_model import A00EndoModel, stabilize_zero_scales
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch
from tests.architecture.test_a04_reference_reconstruction import _camera


class Encoder(nn.Module):
    def forward(self, image):
        return [image, image, image]


class Stereo(nn.Module):
    def __init__(self):
        super().__init__()
        self.disparity = nn.Parameter(torch.tensor(8.0))

    def forward(self, image, *, iters, test_mode):
        prediction = self.disparity.expand(image.shape[0] // 2, 1, *image.shape[-2:]).clone()
        return prediction if test_mode else [prediction] * iters


class Head(nn.Module):
    def forward(self, image, depth, features):
        self.seen_depth = depth
        b, _, h, w = image.shape
        rotation = image.new_zeros(b, 4, h, w)
        rotation[:, 0] = 1
        return rotation, depth.expand(b, 3, h, w) * 0.001, torch.ones_like(depth) * 0.5


def fixture():
    native = nn.Module()
    native.img_encoder = Encoder()
    native.raft_stereo = Stereo()
    native.gs_parm_regresser = Head()
    model = A00EndoModel(
        native, lambda predictions, gt, valid: ((predictions[-1] - gt).abs()[valid].mean(), {})
    )
    rgb = torch.full((1, 3, 5, 5), 0.25)
    batch = DeploymentStereoBatch(
        rgb, rgb, _camera(dtype=torch.float32), torch.ones(1, 1, 5, 5, dtype=torch.bool)
    )
    return model, batch


def test_metric_centers_use_principal_offset_and_integer_pixels_with_gradient():
    model, batch = fixture()
    output = model(batch, with_gaussians=True)
    assert output.field is not None
    # fx*B / (8 - principal_offset3) = 2 metres, not normalized [0,1].
    assert torch.equal(model.native.gs_parm_regresser.seen_depth, torch.full((1, 1, 5, 5), 2.0))
    assert torch.allclose(output.field.means3d[0, 0], torch.tensor([-0.04, -0.04, 2.0]))
    output.field.means3d[..., 2].mean().backward()
    assert torch.allclose(model.native.raft_stereo.disparity.grad, torch.tensor(-0.4))


def test_training_gt_validity_never_controls_inference_geometry():
    model, batch = fixture()
    gt = torch.ones(1, 1, 5, 5) * 7
    mask = torch.zeros_like(gt, dtype=torch.bool)
    mask[..., 2, 2] = True
    trained = model(batch, gt_disparity=gt, gt_valid=mask, with_gaussians=True)
    deployed = model(batch, with_gaussians=True)
    assert torch.equal(trained.field.means3d, deployed.field.means3d)
    assert deployed.field.valid_mask.all()
    assert trained.disparity_loss == 1
    assert deployed.disparity_loss is None


def test_invalid_prediction_is_masked_without_silently_changing_disparity():
    model, batch = fixture()
    model.native.raft_stereo.disparity.data.fill_(-1)
    output = model(batch, with_gaussians=True)
    assert (output.disparity == -1).all()
    assert not output.field.valid_mask.any()
    assert torch.isfinite(output.field.means3d).all()


def test_stereo_only_step_has_no_gaussian_head_call():
    model, batch = fixture()
    output = model(batch)
    assert output.field is None
    assert not hasattr(model.native.gs_parm_regresser, "seen_depth")


def test_opt_in_zero_scale_stabilization_preserves_positives_and_covariance_gradient():
    positive_subnormal = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    scale = torch.tensor([0.0, float(positive_subnormal), 0.001, 0.01], requires_grad=True)
    fixed = stabilize_zero_scales(scale)
    assert fixed[0] == torch.finfo(torch.float32).tiny
    assert torch.equal(fixed[1:], scale[1:])
    assert torch.equal(fixed.square(), scale.square())
    old_grad = torch.autograd.grad(scale.square().sum(), scale, retain_graph=True)[0]
    new_grad = torch.autograd.grad(fixed.square().sum(), scale)[0]
    assert torch.equal(old_grad, new_grad)
    invalid = torch.tensor([-1.0, float("inf"), float("nan")])
    torch.testing.assert_close(stabilize_zero_scales(invalid), invalid, equal_nan=True)


def test_default_reject_keeps_legacy_scale_outputs_and_explicit_policy_counts_zeros():
    model, batch = fixture()

    class ZeroHead(Head):
        def forward(self, image, depth, features):
            rotation, scale, alpha = super().forward(image, depth, features)
            return rotation, scale * 0, alpha

    model.native.gs_parm_regresser = ZeroHead()
    old = model(batch, with_gaussians=True)
    assert not old.field.scales.any() and old.scale_underflow_count == 0
    model.scale_underflow_policy = "zero_to_float32_tiny"
    fixed = model(batch, with_gaussians=True)
    assert fixed.scale_underflow_count == 75
    assert (fixed.field.scales == torch.finfo(torch.float32).tiny).all()
    for name in ("means3d", "colors", "rotations", "opacities", "valid_mask"):
        assert torch.equal(getattr(old.field, name), getattr(fixed.field, name))
