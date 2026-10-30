# Student and joint learning experiment

## Status and authority

This implements the next user-authorized experiment from the local handoff:
student reconstruction learning followed by joint student/head/repair learning
and supervised utility selection. It does not certify Gate I or Gate II,
establish calibrated uncertainty, or replace the frozen A04 experiment.

### User correction — 2026-10-20

The requested sequence is frozen-teacher disparity KD into the stereo student
with RGB reconstruction learning for student/head, output-quality evaluation,
explicit student/head freeze, and then Phase II repair-action learning.
The teacher remains frozen during KD;
student/head weights and buffers remain fixed during subsequent action learning.
Selector training follows the action/oracle evidence and existing gate rules.
Do not automatically chain training stages or launch training without an
explicit user instruction to run it.

The opt-in stronger stereo architecture, its compact controls and bounded KD
evidence are documented in [Student capacity](student_capacity.md). Architecture
selection does not itself approve student freezing or a full-data run.

`configs/training/student_kd.yaml` prepares positive-KD student-only training
with `kd_weight: 1.0`, `joint_epochs: 0`, and the existing ten-epoch LR horizon.
Use the canonical RGB plus teacher-disparity KD path without an intervention
configuration for this first stage. The epoch horizon does not certify quality
or convergence. The full-data candidate uses all 22,950 original TRAIN 1–7
samples; final test 8–9 remains excluded from optimization and model selection.
Any former development validation frames included in that full-data manifest
are training diagnostics thereafter, not held-out evidence. A genuine held-out
development run must record its smaller FIT manifest explicitly.

The three-epoch `student_quality` run used `kd_weight: 0.0` and TRAIN-GT geometry
supervision. Preserve it as a no-KD diagnostic; it does not establish teacher
transfer, KD student readiness, or approval to freeze a KD artifact. Its earlier
asset-search failure is not evidence that the teacher is unavailable: the
external installation must be rechecked. Changing KD/objective/manifest starts
a separately identified experiment, not an implicit resume or relabeling of
the no-KD checkpoint.

### User objective clarification — 2026-10-17

The objective is better final reconstruction output on the baseline task.
Improving the stereo student is a possible means, not an acceptance criterion.
Compare the final deployed Gaussian/rendering output with a matched control,
using identical stereo inputs, target cameras/images, frame lists, resolution,
masks, metric implementations and aggregation. The original left/self-view
reconstruction remains the primary rendering task; report right/cross-view
results separately. Keep original SCARED TRAIN 1–7 and final test 8–9, and use
TRAIN-only held-out data for development decisions.

Assess final-image PSNR/SSIM and LPIPS when a verified backend is available,
alongside supported Gaussian/3D quality and coverage. Stereo depth errors are
component diagnostics; they do not substitute for Gaussian or rendered-depth
quality. Missing output metrics remain unavailable. Student freezing or
updating must be justified by final-output comparisons, preserving the existing
research gates and stage boundaries.

The 2026-10-16 no-KD qualification workflow selects eligible epochs by smallest
stereo depth MAE against initialization. That legacy rule does not establish
this output-quality objective or improvement over a reproduced baseline. Its
metrics/checkpoints remain diagnostic evidence. A matched output comparison
and a declared output-based selection rule are required before promoting that
recipe to the full-data production candidate; do not silently relabel a resume
as implementing the revised objective.

This document describes the legacy utility experiment. The user-approved
opt-in gain/harm selector, correspondence-aware repair, screened KD and
model-only initialization are specified in
[Intervention selection](intervention_selection.md). Without
`--intervention-config`, the legacy objective below remains active.

The active data contract remains original SCARED: final_train is dataset_1–7;
final_test is dataset_8–9. No split, preprocessing, baseline, teacher weights,
existing external artifacts or running server job is changed by this code.

## Executable architecture

```text
DeploymentStereoBatch: left/right RGB, rectified metric camera, sensor support
 -> shared stereo CNN: 3 -> 32 -> 32, two stride-2 convolutions
 -> horizontal negative squared feature-distance cost volume, 32 candidates
 -> masked soft expectation, stride-4 pixel scaling, bilinear upsampling
 -> RGB + normalized coarse disparity decoder: 7 -> 32 -> 1
 -> positive effective disparity parameterization + principal-point offset
 -> canonical A03 geometry/prior/source-index sampling
 -> independent Gaussian attribute head: 13 -> 128 -> 128 -> 7
 -> canonical bounded composition and full-frame left/right rendering
 -> optional fixed regions + detached 18 pre-action features + utility selector
 -> STOP / existing stereo repair / existing Gaussian repair
```

The student and Gaussian head start from their own initialization. The CLI
does not copy or fine-tune a frozen A04 head. Both RGB views supervise this
experiment; neither is advertised as a held-out supervision view. Reference
A04 retains its original left-only objective and provenance rules.

`StudentReconstructionResult` includes disparity `[B,1,H,W]` in input-grid
pixels (`u_left-u_right`), sensor/prediction masks, `GeometryPrior`,
`GaussianField`, two RGB renders `[B,3,H,W]`, and student provenance. Its
inherited `target` slot is an internal predicted-disparity adapter, not a
teacher target. Serialized provenance names `prediction_identity` and
`prediction_adapter`, uses `student_reconstruction_provenance.v1`, and states
that both views are supervised. No confidence/calibrated uncertainty is
invented. At selected inference, `prior` retains pre-action correspondence;
the field, disparity and renders contain the selected outcome, explicitly
identified in diagnostics.

The candidate disparity spacing is four input-grid pixels. The stereo model
uses a soft expectation, not a hard correspondence solver. Effective disparity
is `0.25 + softplus(coarse + residual - principal_point_offset)` by default.
This parameterization and the search range are experimental choices. Depth,
backprojection, surface construction and Gaussian composition still call the
existing canonical implementations. Hard support, sampling, region choice,
and detached stereo-repair bounds do not receive gradients.

## Parameters, losses and schedule

Training and smoke runs may opt into `--gpu-input`: PNG/TIFF decoding stays on
CPU, followed by transfer of decoded arrays to CUDA. RGB normalization, crop,
resize, disparity/mask preparation and camera crop/resize use CUDA. Byte RGB
normalization uses a device lookup rounded once to FP32, matching the canonical
CPU mapping. The CUDA RGB resize supports unchanged dimensions and exactly
twofold reduction (including 1280x1024 to 640x512). It explicitly accumulates the
four quarter-weight products from top left to bottom right to retain the
single-thread CPU RGB kernel's rounding. Use `OMP_NUM_THREADS=1`, as in the
existing run; a different reference thread count is explicitly rejected for
CUDA twofold resizing because CPU dispatch can change the rounding expression.
Other CUDA resize scales raise an error; there is no CPU image-processing fallback.
The separate offline-truth reader reuses the deployment sample's verified
source/crop binding without decoding RGB again; GT remains outside deployment
inputs. Camera, crop, nearest-exact GT resampling, FP32, losses and sample order
retain their existing definitions. CPU loading remains the default. RGB and
GT equivalence and downstream rendering must be checked on representative data
before migrating an existing run. The generic CUDA bilinear implementation was
rejected after small input roundoff exceeded the declared downstream disparity
error bound; the restricted expression above addresses that rounding difference.

The selected pipeline is recorded in checkpoint identity. A legacy checkpoint
cannot silently resume with this option: an explicit, separately recorded
identity migration must preserve every model/optimizer tensor, RNG state,
configuration and cursor. Preserve the original checkpoint and source snapshot
and record synchronized end-to-end timing before adopting the CUDA path.
The current device pipeline identity is `cuda_a00_input.v2`; it is incompatible
with the earlier CPU-RGB/CUDA-GT candidate identity. Verification does not
authorize starting or resuming training; that requires an explicit user request.

Default parameter counts (width 32, attribute head width 128):

| Owner | Parameters | Optimizer | Student stage | Joint stage |
| --- | ---: | --- | --- | --- |
| Stereo student | 12,481 | model Adam | updated | updated |
| Gaussian attribute head | 19,207 | same model Adam | updated | updated |
| Stereo repair | 1,441 | action Adam | no update | updated |
| Gaussian repair | 1,735 | same action Adam | no update | updated |
| Utility selector | 1,730 | selector Adam | no update | updated |
| Teacher / frozen reference | external | none | unchanged | unchanged |

All three optimizer parameter sets are disjoint. The model optimizer contains
only student and its independent attribute head. Selector MSE never owns or
updates student, reference, teacher or repair parameters. The old
`ActionLearner` rejects student models so an accidental mode mix cannot freeze
the student silently.

Let `Lrgb` be the existing mean left/right RGB Charbonnier loss on fixed
deployment masks. Let `Lkd` be masked mean SmoothL1 between student disparity
and detached aligned teacher disparity, in input-grid pixels. Let `Lrepair`
be the mean of reconstruction losses after stereo and Gaussian repairs.

```text
Student stage: L = reconstruction_weight * Lrgb + kd_weight * Lkd
Joint stage:   L = reconstruction_weight * Lrgb + kd_weight * Lkd
                 + repair_weight * Lrepair
Selector:      MSE(predicted STEREO/GAUSSIAN utilities, detached measured utilities)
```

Defaults are weight 1 for each reconstruction/KD/repair term and learning
rate 0.001 for each Adam optimizer. Gradients are checked for finiteness and
clipped to norm 1 for each owned module group before updates. Native student
execution explicitly uses FP32; it does not silently enable AMP or omit a
required scaler state. Routing settings reuse the existing routing config.
The runtime contract explicitly accepts FP32, including configuration and
checkpoint round trips. Frozen-reference defaults remain FP16 with FP32
geometry; its trainer disables autocast when an explicit FP32 profile is used.
Student config owns the global seed and epoch schedule; the legacy routing
config's `seed`, `action_epochs` and `selector_epochs` do not drive this mode.
Routing width, tile/halo, repair bound, action/selector learning rate, utility
cost weight and deployment budget still apply.

`configs/training/student.yaml` runs one student epoch and zero joint epochs.
`configs/training/student_joint.yaml` runs one student epoch then one joint
epoch. The student remains trainable throughout joint learning. Every joint
sample collects current utility labels before any optimizer changes, then
updates model, repairs and selector once. Both differentiable repair branches
start from the same full-frame student state; they are not chained. The
selector observes a rotating eligible core using the same real repair actions.
STOP utility is exactly zero. Utility remains the canonical RGB-loss decrease
minus measured action milliseconds times the configured cost weight.

Features and labels are detached; action choice is discrete. There is no
policy gradient, straight-through selector, or upstream gradient from utility
MSE. Student gradients come from reconstruction, KD and differentiable repair
losses. This is alternating supervised joint optimization, not RL.

Teacher supervision is a separate `teacher_target`/provider interface. GT is
rejected from source samples before forward or teacher-provider invocation.
Default CLI KD uses a frozen live teacher. With `--cache-root`, only existing
identity-matched cache entries are read; a missing item fails without cache
generation or live inference. Cache mode builds the teacher identity descriptor
and verifies its configured external artifacts, but never calls `freeze()` or
loads the teacher model. External source/config/weights still need to exist for
identity verification; this is not a portable cache identity migration.
`kd_weight: 0` permits reconstruction-only training with no teacher.

## Recovery, inference and frozen compatibility

The opt-in `configs/training/student_quality.yaml` is a user-approved ten-epoch
student-only, no-KD qualification schedule. Use the existing metric geometry
intervention config for separate TRAIN-GT depth supervision; no repair or
selector optimizer is stepped in this student-only schedule. It is not the
teacher-backed experiment or a reproduction of the paper's training protocol.

`model_lr_schedule: cosine_epochs` decreases model Adam LR from 0.001 in epoch 0
to 0.0001 in epoch 9, holding that floor thereafter. The canonical formula is
in `training/student_schedule.py`. Its explicit ten-epoch decay horizon remains
fixed if validation selects a shorter final run, so full-data training uses the
same LR prefix. Action and selector LRs retain their existing owners. Frozen
student mode requires constant model Adam settings.

LR is applied before each sample update. A checkpoint on an epoch boundary
contains the LR of its last committed update; strict loading validates that LR
from the saved cursor before restoring Adam. Constant-mode config serialization
omits neutral schedule fields and retains historical config hashes. Changing
an active experiment's epoch budget or source requires an explicit new artifact
with preserved ancestry, not in-place editing or a weakened resume check.

`student_joint_checkpoint.v1` stores owned weights, resolved construction,
student and routing configs/hashes, all Adam states, measured costs, RNG and
exact sample cursor. It records code hash, Git commit/dirty state, manifest
and role hashes, seed, teacher identity when used, renderer, PyTorch/CUDA and
GPU identity. Teacher and frozen reference weights are absent.

Resume requires the complete recorded identity/config and consistent cursor.
Schema, ownership, model tensor shapes/finiteness, Adam parameter order/config,
moments and step counters, costs and RNG are checked before loading live
weights. Failed training retains the last safe checkpoint. Existing outputs
and checkpoints are never overwritten. SIGINT/SIGTERM and available SIGHUP
stop at sample boundaries. Signal registration also supports Windows, where
SIGHUP is absent.

Deployment reconstructs from the student checkpoint, loads only its owned
weights/cost calibration, and does not restore Adam/RNG or instantiate/load
FoundationStereo. Code/data/roles/renderer and construction configs remain
pinned; different deployment hardware is allowed. `--selected` requires
measured cost calibration from joint training. Base-only inference works for
student-only checkpoints. Evaluation restores all nested train/eval flags
even on failure and never steps optimizers or changes weights/buffers.

Frozen mode remains `python -m reliable_endo_gs.cli.phase2` with
`configs/routing/frozen_a04.yaml` and its existing checkpoint schema. Its
reference-freeze checks remain enforced. The shared stereo action now respects
explicit FP32 precision, and feature extraction uses no-grad at entry.

Source fingerprints change with this implementation. Old frozen server
artifacts remain tied to their recorded original checkout; strict resume on
a different source still fails. Use that original checkout for the running
frozen experiment. No artifact relabeling, compatibility bypass or migration
policy is implemented here.

## Exact commands

An explicitly configured `student_epochs: 0` with positive `joint_epochs`
updates student/head, both repair heads, and selector from the first sample.
The historical warmup configuration remains available. Progress records include
elapsed time, gradient norms for each of the five components before clipping,
and the measured STOP/stereo/Gaussian utility labels. These diagnostics show
learning activity; they do not establish Gate I/II or held-out quality.
Runtime identity records the concrete renderer's identity rather than the
construction configuration's injection placeholder.

The native zero-repair smoke comparison bounds absolute RGB error by 2e-4
and mean absolute RGB error by 1e-7 on 0..1 images, while retaining the original
Gaussian field tolerances. Independent GPU 0 diagnosis found repeat rendering
and zero stereo repair bit-identical, but zero Gaussian repair's quaternion
matrix roundtrip changed components by at most one FP32 ULP (1.1921e-7).
On the inspected sample this caused maximum right RGB error 1.1468e-4 and mean
error 1.6285e-8. The smoke report retains per-action/per-view errors. These
tolerances affect only the smoke assertion, not composition or training.

Training may explicitly request `--cache-root "$CACHE_ROOT" --populate-cache`.
This generates a missing frozen-teacher target once through canonical A01
collection, restores the training RNG around teacher loading/inference, and
uses checksum-verified cache hits thereafter. Target mode is pinned in resume
identity. Without this flag, cache misses still fail without inference. The
flag requires training with positive KD and a cache root; smoke and inference
cannot populate caches. No GT is loaded or written.

These commands assume an existing authorized runtime. No dependency install
was performed. Native commands below are **not run** in this implementation.
Configure external assets first; choose new output paths for every invocation:

```bash
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${ALLOCATED_GPU:?set the explicitly allocated physical GPU}"
# Set DATA_ROOT, MANIFEST, OUT_ROOT, TRAIN_SAMPLE, and STUDENT_CHECKPOINT.
# Raw data/weights/output paths stay external.
```

Positive-KD training, including cache mode, requires the existing pinned
`RELIABLE_ENDO_TEACHER_CONFIG` YAML. For live KD additionally set `TEACHER_ROOT`
to that config's `source_root` and put it on the Python import path before
starting Python (as in `docs/a01_external_teachers.md`):

```bash
export PYTHONPATH="$PWD/src:${TEACHER_ROOT:?set the pinned teacher source root for live KD}"
```

Cache mode does not import the teacher model; the initial `PYTHONPATH` containing
only this checkout's `src` suffices. Student-only inference likewise uses only
that initial setup and does not require any teacher configuration or source.

Bounded native smoke: exactly one declared final_train sample, one student
update and one joint update, including zero-repair identity checks:

```bash
python -m reliable_endo_gs.cli.student smoke \
  --config configs/training/student_joint.yaml \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" \
  --sample-id "$TRAIN_SAMPLE" --output-dir "$OUT_ROOT/student-smoke"
```

The smoke uses a one-sample sequence and marks its checkpoints
`bounded_smoke`; those cannot be deployed as full-run artifacts. It is not a
dataset epoch or an accepted scientific experiment. Optional `--cache-root`
reads an existing KD cache instead of running live teacher inference.
The smoke verifies that student, head, stereo repair, Gaussian repair and
selector parameters all changed. `result.json` records these checks, elapsed
training seconds and peak allocated/reserved CUDA bytes; `run.json` carries
the hardware/config/artifact identities. Run this only on an explicitly
allocated GPU using a separate checkout and fresh external output directory.

Bounded full-manifest student training and strict resume, each consuming at
most two samples; these normally remain inside the student warmup stage:

```bash
python -m reliable_endo_gs.cli.student train \
  --config configs/training/student_joint.yaml \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" --max-steps 2 \
  --output-dir "$OUT_ROOT/student-bounded"
python -m reliable_endo_gs.cli.student train \
  --config configs/training/student_joint.yaml \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" --max-steps 2 \
  --checkpoint "$OUT_ROOT/student-bounded/stopped.pt" \
  --output-dir "$OUT_ROOT/student-resumed-bounded"
```

Full student-only learning (one full final_train epoch), **not run**:

```bash
python -m reliable_endo_gs.cli.student train \
  --config configs/training/student.yaml \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" \
  --output-dir "$OUT_ROOT/student-full"
```

Full joint schedule (one student epoch plus one joint epoch), **not run**:

```bash
python -m reliable_endo_gs.cli.student train \
  --config configs/training/student_joint.yaml \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" \
  --output-dir "$OUT_ROOT/student-joint-full"
```

The full joint command starts a separate experiment with its own warmup.
Changing a completed student-only checkpoint's epoch schedule is deliberately
not accepted as strict resume. Artifact transfer/initialization across
different schedules needs a separately specified migration policy.

One-sample student-only inference, **not run**, with no teacher configuration,
source path, cache or weights required:

```bash
python -m reliable_endo_gs.cli.student infer \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" \
  --checkpoint "$STUDENT_CHECKPOINT" --sample-id "$TRAIN_SAMPLE" \
  --output-dir "$OUT_ROOT/student-infer"
```

Add `--selected` for the learned selector plus one repair. Inference writes
`prediction.pt` (disparity, renders, centers and source indices) and
`result.json` with provenance. `--data-role final_test` accepts a declared test
sample only for the approved final evaluation; it is not a tuning command.
No full student benchmark/model-selection protocol is introduced here.

Frozen-reference bounded/full commands, **not run**; use the matching original
source checkout and immutable `A04_REFERENCE_CHECKPOINT`:

```bash
python -m reliable_endo_gs.cli.phase2 train \
  --config configs/routing/frozen_a04.yaml \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" \
  --reference-checkpoint "$A04_REFERENCE_CHECKPOINT" --max-steps 2 \
  --output-dir "$OUT_ROOT/frozen-bounded"
python -m reliable_endo_gs.cli.phase2 train \
  --config configs/routing/frozen_a04.yaml \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" \
  --reference-checkpoint "$A04_REFERENCE_CHECKPOINT" \
  --output-dir "$OUT_ROOT/frozen-full"
```

## Files changed

| File | Change |
| --- | --- |
| `src/reliable_endo_gs/reconstruction/student.py` | Stereo model, differentiable target acquisition, output/provenance contract |
| `src/reliable_endo_gs/training/student_config.py` | Strict experiment settings and hashing |
| `src/reliable_endo_gs/training/student_learning.py` | Disjoint optimizers, objectives, intervention supervision and immutable evaluation |
| `src/reliable_endo_gs/training/student_checkpoint.py` | Separate checkpoint schema, validation, resume and deployment loading |
| `src/reliable_endo_gs/training/student_run.py` | Ordered epochs, bounded samples, signals, logs and safe checkpoints |
| `src/reliable_endo_gs/training/student_runtime.py` | External assets, optional frozen KD and bounded native smoke |
| `src/reliable_endo_gs/cli/student.py` | Thin smoke/train/infer CLI |
| `src/reliable_endo_gs/actions/repair.py` | Explicit student gradient support and correct precision scope |
| `src/reliable_endo_gs/routing/selector.py` | No-grad deployment feature extraction |
| `src/reliable_endo_gs/training/action_learning.py` | Prevent accidental student freezing in the old mode |
| `src/reliable_endo_gs/training/harness.py` | Available-signal registration for Windows/Linux |
| `src/reliable_endo_gs/config/deployment.py` | Explicit FP32 runtime support; preserved FP16 defaults and FP32 geometry requirement |
| `src/reliable_endo_gs/training/a04_reference.py` | Disable AMP for an explicit FP32 profile |
| `configs/training/student.yaml`, `student_joint.yaml` | Student-only and joint schedules |
| `tests/training/test_student_joint.py` | Gradients/updates/ownership, identities, GT, deployment and recovery |
| `tests/training/test_student_runtime.py` | Real-config CLI startup, cached warmup/joint, no teacher load on hit/miss, FP32 checkpoint deployment and smoke reporting with CPU substitutes |
| `README.md`, `PLAN.md`, `docs/local_student_joint_handoff.md` | Current entrypoint/status links while retaining historical evidence |
| `docs/student_joint_learning.md` | Architecture, schedule, limits, validation and commands |

## Implementation validation and evidence still needed

Implemented: all four requested modes; reusable real repairs, same-state
detached supervision, teacher/GT exclusion, safe recovery, and bounded commands.
Focused CPU synthetic tests exercise the canonical geometry/composition and a
differentiable analytic renderer. They establish gradient/ownership behavior,
not native rasterizer parity, real-data quality or runtime savings.

Original implementation results: **107 focused tests passed**; repository lint and format
checks passed (339 files); mypy passed over all 166 source files; student and
frozen CLI help, checked-in configuration parsing, infrastructure smoke,
package compilation and Git whitespace checks passed. The focused set covers
student reconstruction gradients, actual parameter updates, teacher/reference
immutability, exact optimizer ownership, same-state/detached labels, GT
exclusion, STOP/zero-repair identity, primitive cores, student and joint resume,
compatibility failures, teacher-free deployment and evaluation state/mode
restoration, including failure paths. The full repository test suite was not
run in this implementation.

A subsequent startup audit found that the student's FP32 override conflicted
with the old runtime profile's FP16/BF16 validator. Learner-only tests missed
this failure. That validator is fixed and CLI orchestration tests now cover
both train and smoke startup using checked-in configs. They also execute real
warmup/joint updates and FP32 checkpoint reconstruction through deployment,
substituting the native device/renderer and dataset with explicit CPU test
boundaries. They are not native/server validation.

After the startup/cache fixes, the expanded focused command below passed
**137 tests**. Ruff lint and format passed (340 files), and mypy passed over
166 source files. Native smoke/server readiness remains unverified. The user
subsequently authorized commit/push before native server validation; publishing
this implementation does not establish CUDA or server readiness. The server
host/checkout/environment have not yet been supplied. The earlier restriction
against automatically launching GPU experiments still applies to the prepared
native smoke.

Reproduce the focused checks with an existing environment and a fresh external
`VALIDATION_TMP` (the local run used `-p no:cacheprovider` to avoid inaccessible
existing pytest caches):

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  -p no:cacheprovider --basetemp "$VALIDATION_TMP" --tb=short \
  tests/training/test_student_joint.py tests/training/test_student_runtime.py \
  tests/routing/test_frozen_a04.py \
  tests/architecture/test_a04_reference_reconstruction.py \
  tests/integration/test_a04_training_recovery.py \
  tests/rendering/test_metric_native_boundary.py \
  tests/architecture/test_package_boundaries.py tests/config/test_ci_environment.py \
  tests/architecture/test_deployment_foundation.py tests/integration/test_a04_benchmark.py
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
python -m mypy src
PYTHONPATH=src python -m reliable_endo_gs.cli.student --help
PYTHONPATH=src python -m reliable_endo_gs.cli.phase2 --help
PYTHONPATH=src python -m reliable_endo_gs.cli.main smoke \
  --config configs/experiment/infrastructure_smoke.yaml
python -m compileall -q src
git diff --check
```

Local validation uses the existing Windows Python 3.14.3 environment:
PyTorch 2.9.0+cu128, NumPy 2.4.2, Ruff 0.16.3, mypy 2.3.1, pytest 8.4.2,
SciPy 1.17.1 and tifffile 2026.9.15. Tests run on CPU tensors. This is different
from Python 3.10 / CPU PyTorch 2.7.1 and the other versions pinned in
`environments/ci-constraints.txt`. The pinned CI environment was inspected but
not installed. `pip check` reports existing boxmot, crawl4ai and transformers
dependency conflicts; OmegaConf is also absent. Exact-constraints CI validation
is blocked by this environment mismatch and the instruction not to install
dependencies. RTK is unavailable in PATH; shell validation used existing tools
directly. Pytest used fresh external basetemp directories because the default
temporary/cache directories were inaccessible.

Not tested: real SCARED, native GPU forward/backward, FoundationStereo weights,
full training, full cache generation, full benchmarks, CUDA resume, or server
artifact compatibility. None were launched. No merge or server job changes
were performed.

Still requiring evidence: student capacity and search range, KD and repair
weights, whether two-view photometric learning improves metric geometry,
utility heterogeneity and selector generalization as the student changes,
whether full-frame repairs justify their measured cost, end-to-end memory and
latency, and a validation/model-selection protocol that never tunes on
dataset_8–9. No performance or gate claim follows from these tests.
