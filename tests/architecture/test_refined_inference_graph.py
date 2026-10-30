"""RefinedInference with a CUDA graph of the segment before the rasteriser: exact against the eager static path, same errors."""

from dataclasses import replace

import pytest
import torch

from reliable_endo_gs.actions.refinement import RefinedInference
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera
from reliable_endo_gs.contracts.rendering import RenderOutput
from reliable_endo_gs.gaussians import GeometryPriorPolicy, SamplingPolicy
from reliable_endo_gs.gaussians.refinement import FieldRefiner, FieldRefinerConfig
from reliable_endo_gs.reconstruction import ReferenceConfig
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.rendering.field import FieldRenderRequest

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture needs a CUDA device"
)

GRID = (32, 40)
META = {"calibration_identity": "graph-test", "frame_id": "a"}


class MaskedRenderer:
    """Image = a per-view scalar that reads every attribute of the valid primitives (no native rasteriser needed)."""

    identity = "graph-test-masked-renderer"

    def render(self, request: FieldRenderRequest) -> RenderOutput:
        field = request.field
        weight = field.valid_mask.to(field.means3d.dtype)
        per_slot = (
            field.means3d.sum(-1)
            + field.scales.sum(-1)
            + field.opacities[..., 0]
            + field.rotations.sum(-1)
            + field.colors.sum(-1)
        )
        value = (weight * per_slot).sum() * 1.0e-3 + (0.5 if request.view == "right" else 0.0)
        image = value.reshape(1, 1, 1, 1).expand(1, 3, *request.image_hw).contiguous()
        return RenderOutput(image, visibility=torch.ones_like(image[:, :1]))


def _camera(scale: float = 1.0) -> RectifiedStereoCamera:
    left = (
        torch.tensor([[40.0 * scale, 0.0, 20.0], [0.0, 40.0 * scale, 16.0], [0.0, 0.0, 1.0]])
        .unsqueeze(0)
        .cuda()
    )
    right = left.clone()
    transform = torch.eye(4).unsqueeze(0).cuda()
    transform[:, 0, 3] = -0.005
    return RectifiedStereoCamera(left, right, transform, True)


def _batch(seed: int, hw=GRID, scale: float = 1.0) -> DeploymentStereoBatch:
    generator = torch.Generator().manual_seed(seed)
    left = torch.rand((1, 3, *hw), generator=generator).cuda()
    right = torch.rand((1, 3, *hw), generator=generator).cuda()
    support = (
        torch.rand((1, 1, *hw), generator=generator) > 0.25
    ).cuda()  # holes in the sampling grid
    return DeploymentStereoBatch(left, right, _camera(scale), support)


@pytest.fixture(scope="module")
def parts():
    torch.manual_seed(5)
    policy = GeometryPriorPolicy(sampling=SamplingPolicy(mode="fixed", stride=2, cap=100_000))
    config = replace(ReferenceConfig(), head_hidden_dims=(8, 8), prior_policy=policy)
    model = StudentReconstructionModel(
        renderer=MaskedRenderer(), config=config, width=8, bins=8, stereo_architecture="multiscale"
    ).cuda()
    model.eval().requires_grad_(False)
    refiner = FieldRefiner(
        FieldRefinerConfig(hidden_dims=(8,), bounds=config.residual_bounds)
    ).cuda()
    for p in refiner.parameters():
        torch.nn.init.normal_(p, std=0.2)
    refiner.eval().requires_grad_(False)
    return model, refiner


def _make(parts, **kwargs) -> RefinedInference:
    model, refiner = parts
    return RefinedInference(
        model, refiner, composition_alpha_epsilon=1.0e-4, static_layout=True, **kwargs
    )


def _outputs(result):
    return {
        "left": result.left.image,
        "right": result.right.image,
        "disparity": result.disparity,
        "means3d": result.field.means3d,
        "rotations": result.field.rotations,
        "scales": result.field.scales,
        "opacities": result.field.opacities,
        "colors": result.field.colors,
        "valid": result.field.valid_mask,
    }


def test_replay_is_bit_identical_to_the_eager_static_path_and_returns_fresh_tensors(parts):
    batches = [_batch(seed) for seed in range(4)]
    with torch.no_grad():
        eager = _make(parts)
        expected = [
            {k: v.clone() for k, v in _outputs(eager(b, calibration_metadata=META)).items()}
            for b in batches
        ]
        eager.close()
        graphed = _make(parts)
        graphed.capture([(batches[0], META)])
        assert graphed.captured_grids == (GRID,)
        results = [
            graphed(b, calibration_metadata={**META, "frame_id": f"f{i}"})
            for i, b in enumerate(batches)
        ]
        graphed.close()
    assert any(int(e["valid"].sum()) < e["valid"].numel() for e in expected), (
        "the test must exercise hole slots"
    )
    for result, want in zip(
        results, expected, strict=True
    ):  # every result is still intact after later calls
        for name, tensor in _outputs(result).items():
            assert torch.equal(tensor, want[name]), name


def test_changed_camera_tensors_are_copied_and_unchanged_ones_are_not(parts):
    first, second = _batch(1), _batch(1, scale=1.1)  # same images, different intrinsics
    with torch.no_grad():
        eager = _make(parts)
        want = [_outputs(eager(b, calibration_metadata=META)) for b in (first, second)]
        want = [{k: v.clone() for k, v in w.items()} for w in want]
        eager.close()
        graphed = _make(parts)
        graphed.capture([(first, META)])
        got = [_outputs(graphed(b, calibration_metadata=META)) for b in (first, second, first)]
        graphed.close()
    assert not torch.equal(want[0]["means3d"], want[1]["means3d"])
    for name in want[0]:
        assert torch.equal(got[0][name], want[0][name]) and torch.equal(
            got[2][name], want[0][name]
        ), name
        assert torch.equal(got[1][name], want[1][name]), name


def test_in_place_camera_edit_is_seen_through_the_version_counter(parts):
    batch = _batch(2)
    with torch.no_grad():
        graphed = _make(parts)
        graphed.capture([(batch, META)])
        before = _outputs(graphed(batch, calibration_metadata=META))["means3d"].clone()
        batch.camera.K_L[:, 0, 0] *= 1.2  # in place: same tensor object, bumped version
        batch.camera.K_R[:, 0, 0] *= 1.2
        after = _outputs(graphed(batch, calibration_metadata=META))["means3d"].clone()
        eager = _make(parts)
        want = _outputs(eager(batch, calibration_metadata=META))["means3d"]
        graphed.close()
        eager.close()
    assert not torch.equal(before, after) and torch.equal(after, want)


def _corrupt_singular_intrinsics(batch):
    batch.camera.K_L[:, 0, 0] = 0.0  # set after the constructor's own validation
    batch.camera.K_R[:, 0, 0] = 0.0


def _corrupt_nan_left(batch):
    batch.left[0, :, 3:9, 4:12] = float("nan")  # non-finite colours on the RGB path


def _corrupt_inf_right(batch):
    batch.right[0, :, 5:20, 6:30] = float("inf")  # non-finite input of the stereo / disparity path


@pytest.mark.parametrize("compile_kernels", [False, True])
@pytest.mark.parametrize(
    "corrupt", [_corrupt_singular_intrinsics, _corrupt_nan_left, _corrupt_inf_right]
)
def test_invalid_input_behaves_exactly_like_the_eager_path(parts, corrupt, compile_kernels):
    """CRITERIA_BC section 3: the graph (and compiled) path raises what eager raises, or returns what eager returns."""
    batch = _batch(3)
    bad = _batch(3)
    corrupt(bad)
    with torch.no_grad():
        eager = _make(parts, compile_kernels=compile_kernels)
        try:
            want, want_error = _outputs(eager(bad, calibration_metadata=META)), None
        except Exception as error:  # noqa: BLE001  (the class and message are compared below)
            want, want_error = None, error
        graphed = _make(parts, compile_kernels=compile_kernels)
        graphed.capture([(batch, META)])
        try:
            got, got_error = _outputs(graphed(bad, calibration_metadata=META)), None
        except Exception as error:  # noqa: BLE001
            got, got_error = None, error
        # a valid frame afterwards still works and matches the eager result: the failed frame poisoned nothing
        good = _outputs(graphed(batch, calibration_metadata=META))
        good_want = _outputs(eager(batch, calibration_metadata=META))
        eager.close()
        graphed.close()
    assert (want_error is None) == (got_error is None), (want_error, got_error)
    if want_error is not None:
        assert type(got_error) is type(want_error) and str(got_error) == str(want_error)
    else:
        for name, tensor in want.items():
            assert torch.equal(got[name], tensor) or torch.isnan(tensor).any(), name
    for name, tensor in good_want.items():
        assert torch.equal(good[name], tensor), name
    assert bool(torch.isfinite(good["left"]).all())


def test_instances_of_one_model_can_be_closed_in_any_order(parts):
    model, refiner = parts
    original = model.renderer
    first = RefinedInference(model, refiner, composition_alpha_epsilon=1.0e-4, static_layout=True)
    second = RefinedInference(model, None, composition_alpha_epsilon=1.0e-4, static_layout=True)
    assert model.renderer is first._skipper is second._skipper
    first.close()
    assert model.renderer is first._skipper  # still needed by the second instance
    second.close()
    second.close()  # idempotent
    assert model.renderer is original


def test_unknown_signatures_fall_back_to_the_eager_path(parts):
    other = _batch(4, hw=(24, 30))
    with torch.no_grad():
        eager = _make(parts)
        want = _outputs(eager(other, calibration_metadata=META))
        graphed = _make(parts)
        graphed.capture([(_batch(4), META)])
        got = _outputs(graphed(other, calibration_metadata=META))  # other grid
        no_meta = _outputs(
            graphed(_batch(4))
        )  # metadata presence differs from the captured example
        keys = _outputs(graphed(_batch(4), calibration_metadata={"other": 1}))  # key set differs
        graphed.close()
        eager.close()
    assert graphed.captured_grids == (GRID,)
    for name in want:
        assert torch.equal(got[name], want[name]), name
    assert bool(torch.isfinite(no_meta["left"]).all()) and bool(torch.isfinite(keys["left"]).all())


def test_capture_requires_the_static_layout_and_a_static_grid(parts):
    model, refiner = parts
    with pytest.raises(ValueError, match="static_layout"):
        RefinedInference(model, refiner, composition_alpha_epsilon=1.0e-4).capture(
            [(_batch(0), META)]
        )
    capped_policy = GeometryPriorPolicy(
        sampling=SamplingPolicy(mode="fixed", stride=2, cap=10)
    )  # the cap truncates the grid
    dense_model = StudentReconstructionModel(
        renderer=MaskedRenderer(),
        config=replace(ReferenceConfig(), head_hidden_dims=(8, 8), prior_policy=capped_policy),
        width=8,
        bins=8,
        stereo_architecture="multiscale",
    ).cuda()
    dense_model.eval().requires_grad_(False)
    holder = RefinedInference(
        dense_model, refiner, composition_alpha_epsilon=1.0e-4, static_layout=True
    )
    try:
        with pytest.raises(ValueError, match="static grid"):
            holder.capture([(_batch(0), META)])
    finally:
        holder.close()


def test_stop_path_has_no_refinement_and_matches_the_base_field(parts):
    model, _ = parts
    batch = _batch(6)
    with torch.no_grad():
        stop = RefinedInference(model, None, composition_alpha_epsilon=1.0e-4, static_layout=True)
        stop.capture([(batch, META)])
        eager_stop = RefinedInference(
            model, None, composition_alpha_epsilon=1.0e-4, static_layout=True
        )
        eager_result = eager_stop(batch, calibration_metadata=META)
        graph_result = stop(batch, calibration_metadata=META)
        stop.close()
        eager_stop.close()
    assert graph_result.action.name == "STOP" and graph_result.changed_fields == ()
    for name, tensor in _outputs(graph_result).items():
        assert torch.equal(tensor, _outputs(eager_result)[name]), name


def test_compiled_kernels_are_a_rounding_level_approximation_and_capture_inside_the_graph(parts):
    batch = _batch(7)
    with torch.no_grad():
        eager = _make(parts)
        want = _outputs(eager(batch, calibration_metadata=META))
        eager.close()
        compiled = _make(parts, compile_kernels=True)
        compiled.capture(
            [(batch, META)]
        )  # the warm-up calls compile, the capture records the compiled kernels
        got = _outputs(compiled(batch, calibration_metadata=META))
        compiled.close()
    assert torch.equal(got["disparity"], want["disparity"])
    assert torch.equal(got["valid"], want["valid"])
    assert torch.equal(got["means3d"], want["means3d"])
    assert torch.equal(got["colors"], want["colors"])
    # Rounding-level for nearly every slot; the few slots at an exact tie of a discrete branch (best neighbour pair, quaternion
    # branch) may take the other branch, so the comparison is on the fraction of slots, not on the worst slot.
    for name in ("scales", "opacities", "rotations"):
        moved = (got[name] - want[name]).abs() > 1e-4
        moved = moved.reshape(*moved.shape[:2], -1).any(-1)
        assert float(moved.float().mean()) < 0.05, (
            name
        )  # ~2e-5 on real frames, noisier on this toy field
    assert bool(torch.isfinite(got["left"]).all()) and bool(torch.isfinite(got["right"]).all())
