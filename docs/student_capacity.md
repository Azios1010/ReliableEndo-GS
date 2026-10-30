# Stereo student capacity and KD diagnostic

The user requested a stronger student on 2026-10-20 after clarifying the
sequence: frozen-teacher KD plus RGB reconstruction, output-quality evaluation,
explicit student/head freeze, then Phase II action learning. This change adds
an opt-in candidate; it does not approve a frozen artifact or start Phase II.

## Architecture and ownership

`StudentLearningConfig.stereo_architecture` accepts `compact` (historical
default) and `multiscale`. The compact default keeps its original parameter
keys, seeded initialization and matching arithmetic. Its neutral architecture
field is omitted from serialization, preserving historical configuration hashes.
Source identity checks still apply; old checkpoints do not silently acquire
new source compatibility or a different architecture.

The multiscale candidate uses:

- A shared residual encoder with feature branches at strides 4, 8 and 16,
  fused onto the existing quarter-resolution matching grid.
- GroupNorm without running batch statistics, supporting batch-one learning
  and subsequent fixed student state.
- A small residual 3D cost regularizer. Invalid matching entries are replaced
  before convolution, validity is supplied separately, and the original hard
  epipolar mask is reapplied before softmax. Its last convolution starts at zero.
- A contextual RGB/disparity residual decoder with dilations 1, 2 and 4.

The existing matching expectation, positive effective-disparity parameterization,
camera geometry, Gaussian head, rendering and KD/RGB loss owners are retained.
Only teacher disparity is distilled. Teacher features, Gaussian attributes and
calibrated uncertainty are not added. No repair or selector optimizer is stepped
in this stage. All checkpoint consumers reconstruct the configured architecture;
warm starts reject a compact/multiscale mismatch.

The candidate configuration is
[`student_kd_multiscale.yaml`](../configs/training/student_kd_multiscale.yaml):
width 64, 64 matching candidates, 1,430,426 stereo parameters and the unchanged
19,207-parameter Gaussian head. The compact control has 12,481 stereo parameters.
Both use FP32 student/head learning and the same frozen FP16 FoundationStereo.

## Bounded diagnostic

This is development evidence from one seed, not a full-data experiment,
paper reproduction, matched-cost comparison or research gate decision.

- Seven FIT images: one metadata-order median eligible image per original
  TRAIN dataset. Fourteen evaluation images: first/last entries of the previously
  fixed development list per dataset. FIT/evaluation keyframes are disjoint.
- All 21 targets come from the same pinned FoundationStereo at 640x512 with
  32 inference iterations. No final-test 8–9 sample is accessed.
- Three separately initialized arms: compact width32/bins32, compact
  width32/bins64, and multiscale width64/bins64. Their initial Gaussian heads
  are bitwise identical. The latter comparison increases both width and
  structure; it does not isolate each architectural component.
- Each arm receives 112 optimizer updates: 16 passes over the same seven
  ordered FIT images, Adam LR 0.001, RGB and disparity-KD weights both 1.
  This bounded constant-LR schedule differs from the prepared full-run
  ten-epoch cosine schedule. Outcomes are recorded at updates 0/28/56/112.
- Teacher targets and input tensors are resident during training. Timings
  exclude file I/O, cache generation, evaluation and checkpoint writes;
  they do not establish full-data training throughput.
- RGB metrics use the same full-image implementations for both views.
  GT is separate evaluation data, never a student input or pilot training loss.
  Geometry uses the existing sampled diagnostic owner (256 cloud points).
  LPIPS and rendered-depth metrics are not introduced by this experiment.

Teacher disparity reaches 220.81 px in sampled FIT and 196.92 px in evaluation.
The compact32 coarse matcher spans 0–124 px; one FIT image has 25.67% of
teacher-valid pixels above that span. A 64-candidate grid spans 0–252 px and
covers these sampled targets. The residual can move final predictions beyond
the coarse span, so this is not a hard output ceiling or full-dataset coverage.

## Observed results after 112 updates

Means over the same 14 development images:

| Metric | Compact32 | Compact64 | Multiscale64 |
|---|---:|---:|---:|
| Teacher disparity EPE, px ↓ | 21.274 | 33.973 | 13.496 |
| Left PSNR, dB ↑ | 18.210 | 16.661 | 21.781 |
| Right PSNR, dB ↑ | 16.887 | 15.495 | 18.934 |
| Left SSIM ↑ | 0.5701 | 0.5598 | 0.6307 |
| Right SSIM ↑ | 0.5207 | 0.5238 | 0.5650 |
| Stereo depth MAE, mm ↓ | 32.573 | 524.573 | 12.114 |
| Stereo depth RMSE, mm ↓ | 144.621 | 1859.621 | 47.550 |
| Median model update after first pass, ms | 69.39 | 76.88 | 164.74 |
| Peak allocated during pilot arm, GiB | 1.110 | 1.355 | 2.430 |

The candidate improves the final render and teacher fit relative to both
controls at this update budget. Merely increasing candidate count did not
improve the compact control in this pilot. Different initial predictions and
greater candidate compute remain explicit limitations of the comparison.

The stronger student's development PSNR and teacher fit improve across the
measured checkpoints. Its depth RMSE remains large, including a transient
295.47 mm at update28 and 47.55 mm at update112. Even its final depth RMSE
exceeds the compact control's initialization (30.15 mm). This tail error and
the remaining 13.50 px teacher gap prevent treating it as a finished student
or a certified frozen Phase I artifact.

Raw targets, manifests, per-image metrics, checkpoints, source snapshot,
update logs, tests and source/config/data hashes are retained externally under
experiment `student-capacity-20261020T093057Z`. Generated metrics are not edited.
The full 22,950-image, ten-epoch candidate run remains a separate explicit
execution. No automatic freeze or Phase II continuation is enabled.
