"""Opt-in inference-time transforms for the frozen student; weights and checkpoint identity never change.

Training defaults, configurations and checkpoints are untouched: each transform is applied after loading and can
be undone. Bit-exact transforms (CUDA graph) and the one documented numerical approximation (parallel-moments
GroupNorm, see ``ParallelMomentsGroupNorm``) are reported separately in ``InferenceOptimizationRecord`` so
that a run can state exactly what was applied.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch
from reliable_endo_gs.reconstruction.stereo_features import ParallelMomentsGroupNorm
from reliable_endo_gs.reconstruction.student import StereoStudent
from reliable_endo_gs.utils.readback import deferred_validation as deferred_validation_scope

_Signature = tuple[Any, ...]


def use_parallel_group_norm(root: nn.Module) -> tuple[int, Callable[[], None]]:
    """Swap every single-group ``nn.GroupNorm`` under ``root``; return (count, undo).

    The replacement shares values, device, dtype, mode and ``requires_grad`` with the original
    and keeps ``state_dict`` keys; the original modules are retained so ``undo`` is exact.
    """
    swaps: list[tuple[nn.Module, str, nn.Module]] = []
    for parent in list(root.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is nn.GroupNorm and child.num_groups == 1:
                parent.add_module(name, ParallelMomentsGroupNorm.from_group_norm(child))
                swaps.append((parent, name, child))

    def undo() -> None:
        for parent, name, original in reversed(swaps):
            parent.add_module(name, original)

    return len(swaps), undo


def _signature(batch: DeploymentStereoBatch) -> _Signature:
    camera = batch.camera
    return (
        tuple(batch.left.shape),
        batch.left.dtype,
        batch.left.device,
        tuple(batch.right.shape),
        batch.right.dtype,
        tuple(camera.K_L.shape),
        tuple(camera.K_R.shape),
        camera.K_L.dtype,
    )


class StereoCudaGraph:
    """Replay the student stereo network as one CUDA graph, bit-for-bit equal to eager execution.

    One graph is captured per example grid (static shapes and addresses). ``predict_with_evidence`` copies
    left/right images and both intrinsics into the static buffers and replays; any request the graph cannot
    reproduce exactly (matching evidence or feature tokens, autograd, training mode, another grid) falls back to
    the eager method. Valid while the student's parameters keep their memory (in-place loads are fine);
    capture after any module swap. Only ``left``, ``right``, ``camera.K_L`` and ``camera.K_R`` are read by the
    captured network, which ``tests/architecture/test_inference_deployment.py`` pins.
    """

    def __init__(self, student: StereoStudent, examples: Sequence[DeploymentStereoBatch]) -> None:
        if not examples:
            raise ValueError("at least one example batch is required")
        if any(not example.left.is_cuda for example in examples):
            raise ValueError("CUDA graph capture requires CUDA tensors")
        if student.training or any(p.requires_grad for p in student.parameters()):
            raise ValueError("freeze the student (eval, requires_grad_(False)) before capture")
        self._student = student
        self._eager = student.predict_with_evidence
        self._captures: dict[
            _Signature, tuple[DeploymentStereoBatch, torch.cuda.CUDAGraph, torch.Tensor]
        ] = {}
        for example in examples:
            signature = _signature(example)
            if signature in self._captures:
                continue
            static = copy.deepcopy(example)
            with torch.no_grad():
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._eager(static, include_evidence=False)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self._eager(static, include_evidence=False)[0]
            self._captures[signature] = (static, graph, output)

    @property
    def grids(self) -> tuple[tuple[int, int], ...]:
        return tuple((sig[0][-2], sig[0][-1]) for sig in self._captures)

    def predict_with_evidence(
        self,
        batch: DeploymentStereoBatch,
        *,
        include_evidence: bool = True,
        include_feature_tokens: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        capture = (
            None
            if include_evidence or include_feature_tokens
            else self._captures.get(_signature(batch))
        )
        if capture is None or torch.is_grad_enabled() or self._student.training:
            return self._eager(
                batch,
                include_evidence=include_evidence,
                include_feature_tokens=include_feature_tokens,
            )
        static, graph, output = capture
        static.left.copy_(batch.left)
        static.right.copy_(batch.right)
        static.camera.K_L.copy_(batch.camera.K_L)
        static.camera.K_R.copy_(batch.camera.K_R)
        graph.replay()
        return output.clone(), {}


@dataclass
class InferenceOptimizationRecord:
    """What was applied; ``numerical_approximations`` is empty iff every applied transform is bit-exact."""

    parallel_group_norm_layers: int = 0
    stereo_cuda_graph_grids: tuple[tuple[int, int], ...] = ()
    deferred_validation: bool = False
    _undo: list[Callable[[], None]] = field(default_factory=list, repr=False)

    @property
    def numerical_approximations(self) -> tuple[str, ...]:
        return ("parallel_moments_group_norm",) if self.parallel_group_norm_layers else ()

    def to_dict(self) -> dict[str, object]:
        return {
            "parallel_group_norm_layers": self.parallel_group_norm_layers,
            "stereo_cuda_graph_grids": [list(g) for g in self.stereo_cuda_graph_grids],
            "deferred_validation": self.deferred_validation,
            "numerical_approximations": list(self.numerical_approximations),
            "bit_exact": not self.numerical_approximations,
        }

    def undo(self) -> None:
        """Restore the original modules and eager methods (in reverse application order)."""
        while self._undo:
            self._undo.pop()()


def optimize_student_for_inference(
    model: nn.Module,
    *,
    parallel_group_norm: bool = False,
    stereo_cuda_graph: bool = False,
    example_batches: Sequence[DeploymentStereoBatch] = (),
    deferred_validation: bool = False,
) -> InferenceOptimizationRecord:
    """Apply the requested deployment transforms to ``model.stereo_student`` in a fixed safe order.

    Order: GroupNorm swap first, then graph capture, so the graph records the swapped layers.
    ``deferred_validation`` wraps ``model.forward`` so every batched validation inside one frame is evaluated
    together at the end of the call (nothing is returned unchecked); see ``deferred_validation`` in
    ``utils.readback``. It is bit-exact and keeps the host from waiting for the GPU in the middle of a frame.
    """
    student = getattr(model, "stereo_student", None)
    if not isinstance(student, StereoStudent):
        raise TypeError("model must expose a StereoStudent as .stereo_student")
    record = InferenceOptimizationRecord()
    if parallel_group_norm:
        count, undo = use_parallel_group_norm(student)
        record.parallel_group_norm_layers = count
        record._undo.append(undo)
    if stereo_cuda_graph:
        runner = StereoCudaGraph(student, example_batches)
        student.predict_with_evidence = runner.predict_with_evidence  # type: ignore[method-assign]
        record.stereo_cuda_graph_grids = runner.grids
        record._undo.append(lambda: student.__dict__.pop("predict_with_evidence", None))
    if deferred_validation:
        inner = model.forward

        def forward_with_deferred_validation(*args: Any, **kwargs: Any) -> Any:
            with deferred_validation_scope():
                return inner(*args, **kwargs)

        model.forward = forward_with_deferred_validation  # type: ignore[method-assign]
        record.deferred_validation = True
        record._undo.append(lambda: model.__dict__.pop("forward", None))
    return record
