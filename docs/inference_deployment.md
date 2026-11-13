# Student inference deployment path

Opt-in, undoable transforms for the frozen student (`reliable_endo_gs.reconstruction.deployment`). Weights,
configs, checkpoints and training defaults are untouched; nothing here changes a scientific formula.

```python
from reliable_endo_gs.reconstruction.deployment import optimize_student_for_inference

record = optimize_student_for_inference(
    model,  # StudentReconstructionModel, eval(), requires_grad_(False)
    parallel_group_norm=True,  # numerical approximation, see below
    stereo_cuda_graph=True,  # bit-exact; one graph per example grid
    example_batches=[example_batch],  # one DeploymentStereoBatch per input grid
    deferred_validation=True,  # bit-exact; validation errors surface at the end of forward()
)
record.to_dict()  # states exactly what was applied and whether every transform is bit-exact
record.undo()  # restores the original modules and eager methods
```

## What each option does

| Option | Exactness | Notes |
| --- | --- | --- |
| `parallel_group_norm` | float32 rounding only (not bitwise) | `ParallelMomentsGroupNorm` (`stereo_features.py`) replaces single-group `GroupNorm`: native reduces one row per sample in one CUDA block (one SM at batch one). The approximation is documented beside the formula; `state_dict` keys are unchanged. |
| `stereo_cuda_graph` | bit-exact | Captures the stereo network per grid. Falls back to eager for matching evidence/feature tokens, autograd, training mode and uncaptured grids. Only `left`, `right`, `K_L`, `K_R` are read by the captured network. |
| `deferred_validation` | bit-exact | `utils.readback.deferred_validation()` collects the batched `raise_first_failure` checks of one forward and evaluates them with one host transfer when the call ends. Nothing is returned unchecked; the exception type and message are unchanged. If the body raises another exception first, a failed validation is raised instead, chained from it. |

Independent of these options (always on, bit-exact): diagnostics counts and validation flags are read back in
batches (`utils.readback`) instead of one synchronising `.item()` each; value checks use masked reductions
instead of boolean-mask indexing; constant index tables are cached or built on the device. Behavioural note:
when an input violates both a structural check (shape/dtype/device) and a value check, the structural error is
raised first.

## Evidence (2026-10-29, development cohort only; not Gate I/II, no `final_test`, no sealed 64-frame cohort)

Criteria were written before any change (sha256 `f40ebf2e464aac1db135c3630caa0e273410f971f187aaa440ee6993c0afdbe5`).
Bit-identical required for readback batching, validation batching and the CUDA graph; for the GroupNorm
approximation the 56-frame val_7 means had to stay within |dEPE| <= 0.01 px, |d depth MAE| <= 0.01 mm,
|dPSNR| <= 0.01 dB, |dSSIM| <= 0.001, |dLPIPS| <= 0.001. This replaces the earlier 0.001 px bit-closeness rule
for this one approximation, on the maintainer's instruction to keep performance while maximising FPS.

* All bounds passed on the 56-frame benchmark (EPE +0.00000 px, depth MAE +0.00001 mm, left PSNR -0.0009 dB,
  right PSNR -0.0013 dB, SSIM 0, LPIPS +0.00004/+0.00008); worst single-frame right-PSNR change 0.0625 dB.
* Latency (batch 1, preloaded inputs, model + geometry + Gaussians + both renders, shared host): 74.1 -> 27.6 ms
  at 512x640 (13.5 -> 36.2 FPS) and 274.7 -> 86.9 ms at 1024x1280 (3.6 -> 11.5 FPS); official benchmark protocol
  median 75.2 -> 27.6 ms. Peak memory unchanged (about 1.0 GB).
* Not adopted: `torch.compile` of the stereo network (faster but not bit-exact; 3.39 px maximum disparity
  deviation at 1024x1280 is unexplained).
* The baseline shares the render path; baseline timings measured on the pre-change code are no longer
  reproducible on this source, and no claim about the baseline is made here.

Evidence directory (scripts, hashes, results): `~/.local/share/reliableendo-gs/reports/inference-optimization-20261029/`.
Tests: `tests/architecture/test_inference_deployment.py`, `tests/unit/test_readback.py`,
`tests/rendering/test_batched_validation.py`.

## Refined inference path (field refiner): `RefinedInference`

`reliable_endo_gs.actions.refinement.RefinedInference` runs the frozen student, the L6 field refiner and both rasterisations
(batch 1). Every option below is opt-in and default off; with all of them off the output is bit-identical to part A.

| Option | Exactness | Notes |
| --- | --- | --- |
| `static_layout=True` | bit-exact on every valid slot | `utils.inference_mode.static_grid_layout`: for `SamplingPolicy(mode="fixed")` whose grid fits in `cap`, slot `k` is grid cell `k` (cells without a valid centre are holes, like padding), no `nonzero`, and the surface frame is evaluated only at the grid cells (`estimate_surface_frame_on_grid`, sharing `_surface_frame_from_neighbours` with the dense path). No-op when the cap would truncate the grid (1024x1280 with stride 2). |
| `capture(examples)` | bit-exact | One CUDA graph from the input images to the refined field (needs `static_layout`). The rasteriser stays outside (its native code reads a count to the host). Validations recorded in the graph are re-registered with `deferred_validation` at every replay; count diagnostics inside are placeholders; outputs are copied. |
| `compile_kernels=True` | numerical approximation | `torch.compile` of four pure tensor functions marked `compiled_when_enabled`; compiled during the warm-up of `capture`. Rounding-level for nearly all slots, a different but equivalent branch at exact ties for about 2e-5 of the slots. |
| `refiner=None` | - | The unrefined STOP path through the same machinery, to measure the refiner's marginal cost under any transform. |

Evidence and criteria (2026-10-30, 56 benchmark frames, no sealed/final_test data): `docs/fps_bc_status_2026-10-30.txt`,
`docs/fps_bc_criteria_2026-10-30.txt`. Tests: `tests/architecture/test_static_grid_layout.py`,
`tests/architecture/test_refined_inference_graph.py`, `tests/unit/test_readback.py`.
