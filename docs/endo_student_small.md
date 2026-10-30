# Small Endo-E2E-GS / KD student comparison

The user first requested configuration preparation only, then explicitly
authorized a separate tmux session and a small Endo-E2E-GS run to compare
against the existing stronger-student pilot. The bounded specification is
[`endo_student_small.yaml`](../configs/experiment/endo_student_small.yaml).
Its runner is `reliable_endo_gs.baseline.training.a00_small`, which checks
authorization, budget, manifest hashes, TRAIN-only identity and clean pinned
upstream before execution. Only Endo is trained; student metrics are read from
the immutable existing 112-update pilot, with a pinned result hash.

## Fixed task and data

Reuse `fit.json` (7 images) and `validation.json` (14 images) from external
experiment `student-capacity-20261020T093057Z`, with their exact file hashes.
Each original TRAIN dataset 1–7 contributes one FIT and two validation images;
whole keyframes are disjoint. Both manifest roles remain `final_train` under
the existing role contract: validation is held-out development evidence for
this pilot, not final TEST8–9. No new sample selection or raw-data copy is needed.

Both models receive the same 512x640 stereo pairs, calibration and sensor
support in the same FIT order, with batch one and seed 20261006. Ground truth
is supervision/evaluation only, never an inference input. Evaluate both target
cameras using full-image PSNR and Gaussian-window SSIM, plus the existing
disparity/depth/geometry owners with counts and coverage. RGB stays in [0,1]
for metrics; report stereo depth in mm from metric geometry, never from a
min/max-normalized depth image. Missing metrics remain unavailable.

## Proposed bounded budget

The prepared default is 112 total optimizer updates per model. This is an
explicit short-budget design choice, not a paper-derived training schedule.

| Model | Stereo-only updates | Joint reconstruction updates | Total |
|---|---:|---:|---:|
| Endo-E2E-GS, fresh initialization | 56 | 56 | 112 |
| Multiscale KD student + Gaussian head, fresh initialization | 0 | 112 | 112 |

Each stage traverses the same seven FIT images in fixed order. The baseline
retains the released architecture, GT-disparity sequence loss, left-view RGB
loss, AdamW and original LR horizons; only data, batch size, seed and stopping
budget are adapted explicitly. Its stage-one scheduler uses the **100100**
hardcoded horizon in `train_stage1.py`, despite YAML `num_steps: 60000`.
Stage two uses horizon 10100. This short stop is still in early warmup and
cannot establish convergence or disprove the baseline's final quality.

The student uses frozen FoundationStereo disparity targets and two-view RGB
Charbonnier loss, Adam LR 0.001 constant, with the previous pilot architecture
and 16 passes. The separate
[`student_kd_multiscale_pilot.yaml`](../configs/training/student_kd_multiscale_pilot.yaml)
keeps the ten-epoch full-data configuration unchanged. Repairs and selector
receive no updates; baseline Stage 2 does not mean research Phase II.

Different supervision, loss, precision, architectures and teacher pretraining
are disclosed. Equal updates/sample presentations are not equal compute.
Report both baseline stages' cost and the student's teacher-target generation
cost separately. Compare final update 112 without selecting checkpoints on
validation outcomes; show all metrics and per-image/per-dataset results.

## Integration checks before execution

The pinned released code is not itself a metric A00 implementation. Static
source findings motivated an explicitly named **A00 metric-geometry variant**:

- `lib/utils.py::disp2depth` multiplies by an input mask and min/max-normalizes
  depth before Gaussian construction. `depth2pc` uses half-pixel centers.
  A00 requires metric depth and integer pixel centers. Confirm the explicit
  camera/input conversion and metric geometry; do not silently replace these
  scientific transformations or label normalized depth as meters.
- The native SCARED loader derives its mask from GT disparity. Inference in
  this task must use deployment sensor support, not GT validity.
- Training scripts hardcode dataset3/keyframe1; `render.py` currently selects
  EndoNeRF. Neither is the correct entry point for the pinned A00 manifests.
- Left/right target-camera rendering, disparity sign/principal-point handling,
  finite outputs and metric geometry still require an adapter validation.

`baseline/a00_model.py` composes the untouched learned upstream image encoder,
RAFT estimator and GS regressor with `reconstruct_rectified` and the same
`ProductionFieldRenderer` used by the student. GT enters the upstream sequence
loss only. Metric depth replaces min/max depth as GS-regressor input and
center geometry; integer-pixel backprojection and calibrated right-camera
rendering are shared with the student. This changes the released pipeline's
scientific geometry, is isolated in a separate variant, and does **not** claim
native parity or paper reproduction. The original checkout stays clean.
Negative/invalid predicted disparity is not flipped, clamped, filled or
converted to GT: canonical validity excludes it and coverage is reported.

External paths are environment-bound: `RELIABLE_ENDO_DATA_ROOT`,
`RELIABLE_ENDO_SMALL_MANIFEST_ROOT`, `RELIABLE_ENDO_SMALL_TEACHER_CACHE`,
`RELIABLE_ENDO_TEACHER_RUNTIME_CONFIG` and `RELIABLE_ENDO_RUN_ROOT`.
GPU0 is pinned by UUID; GPU1 is excluded. Each run needs a new output
directory and source/config/manifest/checkpoint/hardware provenance. No full
training, freeze, gate decision or Phase II continuation is implied.
