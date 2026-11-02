"""Full-frame Gaussian refinement applied to the composed field, with the Stage A Gaussian-repair semantics.

``refine_gaussian_field`` is the Stage A Gaussian branch of ``RepairActions.execute`` with the architecture factored
out into :class:`FieldRefiner` and the write region defaulting to every valid primitive: residuals are predicted from
the context of the already composed field, composed once more by the canonical owner (``compose_gaussian_field``),
merged back by source index and rendered with the existing renderer (both views). Optional colour residuals are added
to the field colours and clamped to [0, 1]. Differentiable end to end for training; geometry (disparity) is untouched.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from reliable_endo_gs.actions.repair import (
    Action,
    ActionResult,
    RepairState,
    _render_outcome,
    merge_field,
    preserve_zero_gaussian_residual,
    primitive_core,
)
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch
from reliable_endo_gs.contracts.gaussians import GaussianField
from reliable_endo_gs.contracts.rendering import RenderOutput
from reliable_endo_gs.gaussians import (
    compose_gaussian_field,
    context_from_geometry_prior,
    quaternion_to_rotation_matrix,
)
from reliable_endo_gs.gaussians.refinement import FieldRefiner
from reliable_endo_gs.gaussians.sampling import fixed_grid_cells
from reliable_endo_gs.reconstruction.model import ReferenceReconstructionModel
from reliable_endo_gs.regions.grid import Region
from reliable_endo_gs.rendering.field import FieldRenderRequest
from reliable_endo_gs.utils.inference_mode import (
    compiled_kernels,
    reuse_camera_settings,
    skip_unread_surface_covariance,
    static_grid_layout,
)
from reliable_endo_gs.utils.readback import (
    PackedChecks,
    collected_checks,
    deferred_validation,
    graph_capture_diagnostics,
    pack_checks,
    raise_first_failure,
)


def full_frame_region(batch_left: torch.Tensor) -> Region:
    """Core = halo = the whole [H,W] grid."""
    height, width = int(batch_left.shape[-2]), int(batch_left.shape[-1])
    return Region(0, (0, 0, height, width), (0, 0, height, width))


def refined_gaussian_field(
    state: RepairState,
    refiner: FieldRefiner,
    reference: ReferenceReconstructionModel,
    *,
    composition_alpha_epsilon: float,
    region: Region | None = None,
    exact_zero_identity: bool = True,
) -> tuple[GaussianField, tuple[str, ...]]:
    """The refined field (every slot of ``region`` composed once more, merged back by source index) and the changed field names."""
    base, batch = state.base, state.batch
    if base.render_right is None:
        raise ValueError("refinement requires both base renders")
    if batch.left.shape[0] != 1:
        raise ValueError("refinement supports batch-size one")
    if base.gaussians.cov_center is not None or base.gaussians.cov_effective is not None:
        raise ValueError("refinement does not implement uncertainty covariance updates")
    region = region or full_frame_region(batch.left)
    field = base.gaussians
    core = primitive_core(state, region)
    prior = replace(
        base.prior,
        mu=field.means3d,
        R0=quaternion_to_rotation_matrix(field.rotations),
        rotations=field.rotations,
        S0=field.scales,
        alpha0=field.opacities,
        rgb=field.colors,
    )
    context = context_from_geometry_prior(prior, context_scale_m=reference.config.context_scale_m)
    residuals = refiner(context, core)
    bounds = replace(refiner.config.bounds, alpha_logit_epsilon=composition_alpha_epsilon)
    changed = compose_gaussian_field(prior, residuals.attributes, bounds=bounds)
    if exact_zero_identity:
        changed = preserve_zero_gaussian_residual(field, changed, residuals.attributes)
    names = ["rotation", "scale", "opacity", "cov_surface"]
    if residuals.delta_color is not None:
        changed = replace(changed, colors=(field.colors + residuals.delta_color).clamp(0.0, 1.0))
        names.append("colour")
    return merge_field(base.gaussians, changed, core), tuple(names)


EDGE_STRIP_DISPARITY_COLUMNS = 8
EDGE_STRIP_QUANTILE = 0.90


def right_edge_strip_mask(disparity: torch.Tensor) -> torch.Tensor:
    """Right-view columns the left view cannot observe: ``[1, 1, 1, W]`` float, 1 where the strip applies.

    ``disparity`` is the left-view map ``[1, 1, H, W]`` in pixels. For a rectified pair with ``x_R = x_L - d`` a
    right-view column ``x_R >= W - d`` would need ``x_L >= W``, which the left view does not contain. The strip width is
    ``d_hat``, the ``EDGE_STRIP_QUANTILE`` quantile over rows of the largest disparity in the last
    ``EDGE_STRIP_DISPARITY_COLUMNS`` columns (a conservative statistic: the outermost disparities are the least reliable).
    A non-finite ``d_hat`` makes the whole width the strip, i.e. the guard falls back to the unrefined image.
    Chosen on development frames only (analysis EDGE_GUARD_DEV_SPEC, candidate C1); a hypothesis-driven design, not an
    established result. ``d_hat`` has a heavy tail (35-221 px on development frames), so a strip can be very wide.
    """
    if disparity.ndim != 4 or disparity.shape[:2] != (1, 1):
        raise ValueError(f"right_edge_strip_mask expects a [1, 1, H, W] disparity map, got {tuple(disparity.shape)}")
    width = int(disparity.shape[-1])
    edge = disparity[0, 0, :, -EDGE_STRIP_DISPARITY_COLUMNS:].amax(dim=1).float()
    d_hat = torch.nan_to_num(torch.quantile(edge, EDGE_STRIP_QUANTILE), nan=float("inf"))
    columns = torch.arange(width, device=disparity.device, dtype=torch.float32)
    return (columns >= width - d_hat).to(torch.float32).reshape(1, 1, 1, width)


def guard_right_edge(result: ActionResult, stop_right: torch.Tensor) -> ActionResult:
    """Replace the right-view image in the strip of :func:`right_edge_strip_mask` by the unrefined (STOP) right image.

    ``stop_right`` is ``[1, C, H, W]``, the right render of the unrefined field. Only ``right.image`` is composited; the
    left image, ``right.depth``/``right.visibility`` (from the refined render), the field and the disparity are untouched.
    """
    mask = right_edge_strip_mask(result.disparity).to(result.right.image.dtype)
    image = result.right.image * (1.0 - mask) + stop_right.to(result.right.image.dtype) * mask
    return replace(result, right=replace(result.right, image=image))


def refine_gaussian_field(
    state: RepairState,
    refiner: FieldRefiner,
    reference: ReferenceReconstructionModel,
    *,
    composition_alpha_epsilon: float,
    region: Region | None = None,
    exact_zero_identity: bool = True,
    edge_guard: bool = False,
) -> ActionResult:
    """Apply ``refiner`` to the primitives of ``region`` (default: all) and render both views.

    ``edge_guard`` (opt-in, default off) leaves the strip of :func:`right_edge_strip_mask` of the right view at the
    unrefined render, which the fully rendered base state already holds.
    """
    field, names = refined_gaussian_field(
        state,
        refiner,
        reference,
        composition_alpha_epsilon=composition_alpha_epsilon,
        region=region,
        exact_zero_identity=exact_zero_identity,
    )
    result = _render_outcome(
        state, field, state.base.target.disparity, reference, Action.GAUSSIAN, names
    )
    if edge_guard:
        if state.base.render_right is None:
            raise ValueError("edge_guard needs the unrefined right render of the base state")
        result = guard_right_edge(result, state.base.render_right.image)
    return result


class _BaseRenderSkipper:
    """Delegates to the real renderer; while ``skip`` is set it returns a zero image without rasterising.

    Used only for the unrefined base pass of :class:`RefinedInference`, whose rendered images are never read.
    """

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.skip = False
        self.skipped_calls = 0
        self.users = 0  # RefinedInference instances sharing this skipper; the original renderer returns when the last one closes

    def __getattr__(self, name: str) -> Any:
        if name in {"inner", "skip", "skipped_calls", "users"}:
            raise AttributeError(name)
        return getattr(self.inner, name)  # identity / backend_id etc.

    def render(self, request: Any) -> RenderOutput:
        if self.skip:
            field = request.field
            self.skipped_calls += 1
            image = torch.zeros(
                field.means3d.shape[0],
                3,
                *request.image_hw,
                device=field.means3d.device,
                dtype=field.means3d.dtype,
            )
            return RenderOutput(image=image)
        return self.inner.render(request)  # type: ignore[attr-defined, no-any-return]


@dataclass
class _CapturedSegment:
    """One CUDA graph of the model forward and the refinement up to the refined field (everything before the rasteriser)."""

    batch: DeploymentStereoBatch  # static input buffers, overwritten before every replay
    metadata_keys: frozenset[str] | None
    graph: torch.cuda.CUDAGraph
    base: Any  # the base result built during capture; its tensors are static, only the batch geometry of it is used
    field: GaussianField
    disparity: torch.Tensor
    names: tuple[str, ...]
    checks: PackedChecks | None
    camera_key: tuple[object, ...] | None = None
    camera_refs: tuple[torch.Tensor, ...] = ()


def _segment_signature(batch: DeploymentStereoBatch) -> tuple[object, ...]:
    camera = batch.camera
    return (
        tuple(batch.left.shape),
        batch.left.dtype,
        batch.left.device,
        tuple(batch.right.shape),
        tuple(batch.support.shape),
        tuple(camera.K_L.shape),
        camera.K_L.dtype,
    )


def _camera_key(camera: Any) -> tuple[object, ...]:
    return tuple(
        (id(t), t.data_ptr(), t._version) for t in (camera.K_L, camera.K_R, camera.T_R_from_L)
    )


class RefinedInference:
    """Deployed inference with an always-applied :class:`FieldRefiner` (batch size one).

    The model forward runs with its two base renders skipped (the unrefined field is only an input to the refiner),
    then ``refine_gaussian_field`` -- the single implementation shared with training and evaluation -- renders the
    refined field. The result is bit-identical to ``refine_gaussian_field`` applied to a fully rendered base state;
    only the two rasterisations of the unrefined field are saved, and the support covariance that no consumer of
    the images reads is not composed (``result.field.cov_surface`` is None). Everything runs in one
    ``deferred_validation`` scope, so nothing is returned unchecked. ``close()`` restores the original renderer once the last
    instance of the model is closed.

    ``static_layout`` additionally keeps one slot per grid cell of a fixed-stride sampling policy and evaluates the surface
    frame only at those cells (``utils.inference_mode.static_grid_layout``): bit-identical on every valid slot.

    ``compile_kernels`` runs the pure tensor functions marked ``compiled_when_enabled`` through ``torch.compile``: a numerical
    approximation (see ``utils.inference_mode``), compiled during the warm-up calls of ``capture`` or on first use.

    ``capture(examples)`` records the segment from the input images to the refined field as one CUDA graph per example
    signature (requires ``static_layout``, whose shapes are static). A call whose image/intrinsics shapes and
    ``calibration_metadata`` key set match a captured example replays it (the metadata values are not compared, see
    ``_lookup``); any other call runs eagerly. The rasteriser stays outside the graph (its native code reads a count back to
    the host). Every validation recorded inside the graph is stacked on the device and registered with the active
    ``deferred_validation`` scope at each replay, so invalid input raises exactly like the eager path. The camera tensors are
    copied into the static buffers only when the caller's camera tensors changed (identity, address or in-place version; torch
    bumps the version of every in-place op). Count diagnostics inside the graph are placeholders, and the returned field and
    disparity are fresh copies (``copy_outputs``), so they survive the next call.

    ``edge_guard`` (opt-in, default off; no effect without a refiner) keeps the strip of :func:`right_edge_strip_mask` of the right view
    at the unrefined render. The base renders are skipped, so this costs one extra right-view rasterisation of the unrefined field
    (latency not measured). With the guard off the outputs are bit-identical to before.
    """

    def __init__(
        self,
        model: ReferenceReconstructionModel,
        refiner: FieldRefiner | None,
        *,
        composition_alpha_epsilon: float,
        region: Region | None = None,
        exact_zero_identity: bool = True,
        static_layout: bool = False,
        copy_outputs: bool = True,
        compile_kernels: bool = False,
        edge_guard: bool = False,
    ) -> None:
        if model.renderer is None:
            raise ValueError("RefinedInference requires a model with an injected renderer")
        self.model = model
        self.refiner = refiner
        self.composition_alpha_epsilon = composition_alpha_epsilon
        self.region = region
        self.exact_zero_identity = exact_zero_identity
        self.static_layout = static_layout
        self.copy_outputs = copy_outputs
        self.compile_kernels = compile_kernels
        self.edge_guard = edge_guard
        # A refined and an unrefined (STOP) instance normally share one model: they share one skipper, so closing order is irrelevant.
        if isinstance(model.renderer, _BaseRenderSkipper):
            self._skipper = model.renderer
        else:
            self._skipper = _BaseRenderSkipper(model.renderer)
            object.__setattr__(model, "_renderer", self._skipper)
        self._skipper.users += 1
        self._closed = False
        self._captured: dict[tuple[object, ...], _CapturedSegment] = {}

    @property
    def skipped_base_renders(self) -> int:
        return self._skipper.skipped_calls

    @property
    def captured_grids(self) -> tuple[tuple[int, int], ...]:
        return tuple((sig[0][-2], sig[0][-1]) for sig in self._captured)  # type: ignore[index]

    def close(self) -> None:
        """Release the shared skipper; the original renderer is restored when the last instance of the model closes."""
        if self._closed:
            return
        self._closed = True
        self._skipper.users -= 1
        if self._skipper.users == 0:
            object.__setattr__(self.model, "_renderer", self._skipper.inner)

    def _scopes(self, *, defer: bool = True) -> ExitStack:
        # Deployment-only scopes (utils.inference_mode): unread support covariance is not composed, native camera
        # settings are reused while the camera tensors are unchanged, and (optionally) the static grid layout is used.
        stack = ExitStack()
        for scope in (
            *((deferred_validation,) if defer else ()),
            skip_unread_surface_covariance,
            reuse_camera_settings,
        ):
            stack.enter_context(scope())
        if self.static_layout:
            stack.enter_context(static_grid_layout())
        if self.compile_kernels:
            stack.enter_context(compiled_kernels())
        return stack

    def _segment(
        self, batch: DeploymentStereoBatch, forward_kwargs: dict[str, Any]
    ) -> tuple[Any, GaussianField, tuple[str, ...]]:
        """Model forward with skipped base renders, then the refined field; no rasterisation."""
        self._skipper.skip = True
        try:
            base = self.model(batch, **forward_kwargs)
        finally:
            self._skipper.skip = False
        if self.refiner is None:
            return base, base.gaussians, ()
        field, names = refined_gaussian_field(
            RepairState(batch, base),
            self.refiner,
            self.model,
            composition_alpha_epsilon=self.composition_alpha_epsilon,
            region=self.region,
            exact_zero_identity=self.exact_zero_identity,
        )
        return base, field, names

    def capture(
        self, examples: Sequence[tuple[DeploymentStereoBatch, Mapping[str, object] | None]]
    ) -> None:
        """Capture one graph per distinct example signature (see the class docstring)."""
        if not self.static_layout:
            raise ValueError("CUDA graph capture requires static_layout=True (static shapes)")
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("freeze the model (eval, requires_grad_(False)) before capture")
        sampling = self.model.config.prior_policy.sampling
        for batch, metadata in examples:
            signature = _segment_signature(batch)
            if signature in self._captured or not batch.left.is_cuda:
                continue
            if (
                fixed_grid_cells(int(batch.left.shape[-2]), int(batch.left.shape[-1]), sampling)
                is None
            ):
                raise ValueError(
                    "the sampling policy has no static grid layout at this resolution; it cannot be captured"
                )
            static = copy.deepcopy(batch)
            kwargs: dict[str, Any] = (
                {} if metadata is None else {"calibration_metadata": dict(metadata)}
            )
            with torch.no_grad(), self._scopes():
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(
                        3
                    ):  # warm-up: lazy initialisation and host-side caches happen before capture
                        self._segment(static, kwargs)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), self._scopes(defer=False), collected_checks() as collected:
                with graph_capture_diagnostics(), torch.cuda.graph(graph):
                    base, field, names = self._segment(static, kwargs)
                    packed = (
                        pack_checks([c for c in collected if not isinstance(c, PackedChecks)])
                        if collected
                        else None
                    )
            self._captured[signature] = _CapturedSegment(
                static,
                None if metadata is None else frozenset(metadata),
                graph,
                base,
                field,
                base.target.disparity,
                names,
                packed,
            )

    def _lookup(
        self, batch: DeploymentStereoBatch, forward_kwargs: dict[str, Any]
    ) -> _CapturedSegment | None:
        if not self._captured:
            return None
        captured = self._captured.get(_segment_signature(batch))
        if (
            captured is None
            or torch.is_grad_enabled()
            or set(forward_kwargs) - {"calibration_metadata"}
        ):
            return None
        metadata = forward_kwargs.get("calibration_metadata")
        # The values only feed provenance records that RefinedInference does not return (they are per-frame identifiers and hashes);
        # the model requires them to be present, so presence and key set are part of the signature, the values are not.
        if (None if metadata is None else frozenset(metadata)) != captured.metadata_keys:
            return None
        return captured

    def _replay(self, captured: _CapturedSegment, batch: DeploymentStereoBatch) -> None:
        static = captured.batch
        key = _camera_key(batch.camera)
        if key != captured.camera_key:
            static.camera.K_L.copy_(batch.camera.K_L)
            static.camera.K_R.copy_(batch.camera.K_R)
            static.camera.T_R_from_L.copy_(batch.camera.T_R_from_L)
            captured.camera_key = key
            captured.camera_refs = (batch.camera.K_L, batch.camera.K_R, batch.camera.T_R_from_L)
        static.left.copy_(batch.left)
        static.right.copy_(batch.right)
        static.support.copy_(batch.support)
        captured.graph.replay()
        if captured.checks is not None:
            raise_first_failure([captured.checks])

    def _unrefined_right_image(self, batch: DeploymentStereoBatch, base: Any) -> torch.Tensor:
        """Right render of the unrefined field (the base renders were skipped)."""
        field = replace(base.gaussians, cov_surface=None, cov_center=None, cov_effective=None)
        hw = (int(batch.left.shape[-2]), int(batch.left.shape[-1]))
        return self.model.renderer.render(FieldRenderRequest(field, batch.camera, hw, "right")).image

    def __call__(self, batch: Any, **forward_kwargs: Any) -> ActionResult:
        with self._scopes():
            captured = self._lookup(batch, forward_kwargs)
            if captured is None:
                base, field, names = self._segment(batch, forward_kwargs)
                disparity = base.target.disparity
            else:
                self._replay(captured, batch)
                base, field, names, disparity = (
                    captured.base,
                    captured.field,
                    captured.names,
                    captured.disparity,
                )
            action = Action.STOP if self.refiner is None else Action.GAUSSIAN
            result = _render_outcome(
                RepairState(batch, base), field, disparity, self.model, action, names
            )
            if self.edge_guard and self.refiner is not None:
                result = guard_right_edge(result, self._unrefined_right_image(batch, base))
            if captured is not None and self.copy_outputs:
                result = replace(
                    result,
                    field=replace(
                        result.field,
                        **{
                            f.name: getattr(result.field, f.name).clone()
                            for f in fields(result.field)
                            if isinstance(getattr(result.field, f.name), torch.Tensor)
                        },
                    ),
                    disparity=result.disparity.clone(),
                )
            return result


__all__ = [
    "RefinedInference",
    "full_frame_region",
    "guard_right_edge",
    "refine_gaussian_field",
    "right_edge_strip_mask",
]
