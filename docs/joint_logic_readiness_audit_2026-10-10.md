# Joint learning logic and readiness audit — 2026-10-10

## Verdict and scope

**Follow-up status: launcher corrected; runtime retry blocked by WSL GPU access.**
The user's first execution failed before any optimizer update because the
prepared launcher omitted the external FoundationStereo source from
`PYTHONPATH`. This was a launcher/preflight omission in this audit. See the
follow-up below; the initial validation evidence remains recorded separately.

**GO for a bounded native joint check with a fresh cache; full training and
selector quality are not certified by this audit.** The command below has been
prepared and parsed, but has not been executed. No real-sample training,
teacher inference, cache generation, final TEST evaluation, dependency install,
or GPU reset was performed.

This is the separately approved student/action/selector experiment in PLAN.md.
The three trainable owners are student plus Gaussian head, both repair heads,
and selector gain/risk heads. FoundationStereo remains a frozen supervision
provider. This does not establish the complete probabilistic Phase I model or
pass the historical Gate I/Gate II.

Checkout HEAD: `27b029a980a00d887939fb4ea10b28b882e7be86`, with existing working-tree
changes preserved. Reviewed first-party Python source SHA-256:
`f20bcc79b42f1c4c697688fad975b359253eae93d707a478dad3460d0634dc8f`.

External evidence directory:
`/home/bkai/.local/share/reliableendo-gs/runs/joint-logic-audit-20261010-txRFIgQq`.

## Executable logic checked

- Deployment forward receives RGB, rectified cameras and sensor support.
  Teacher targets and offline GT are separate supervision arguments. GT does
  not generate proposals, input ROI masks, pre-action features or decisions.
- Every eligible region measures STOP, learned stereo and learned Gaussian
  repair from the same current student/action snapshot, before optimizer
  updates. The harness calls the actual action implementations. There are no
  BatchNorm/dropout modules in the audited student/repair/selector networks
  that would mutate a training snapshot during these counterfactuals.
- Counterfactuals and selector features are detached. Selector-only refinements
  do not update student/repair parameters. The current pilot updates both gain
  and risk four times per observed snapshot; model/actions update once.
- Unknown metric observations are excluded from selector loss and normalization,
  rather than labeled safe. Student/repairs can still learn from RGB proxies.
  Empty target proposals update only the student/head.
- Utility uses the canonical signed bounded per-component score with common
  scales. The pilot applies geometry priority 1.5 and stereo action multiplier
  1.0. Negative gains retain their sign. Physical harm labels are independent
  of utility normalization. Cost enters selection as a budget constraint.
- Global utility, local auxiliary and ranking losses use bounded gradient
  balancing. Explicit student stereo priority is applied after this balance.
  Risk has a separate network and BCE objective. No rule forces three action
  classes to appear.
- Stereo geometry corrects principal-point disparity once, uses a positive
  metric baseline and canonical depth/backprojection. Resize uses half-pixel
  camera coordinates; raw disparity scales with horizontal resize. Stereo
  repair preserves source-index correspondence and out-of-core values.
- The A03 boundary-neighbour implementation rejects wrapped image borders.
  Gaussian repair changes rotation/scale/opacity, preserves centers and the
  disparity map, and uses canonical composition. Its disparity-derived depth
  metric therefore cannot demonstrate improved Gaussian rendered depth.
- Resume checks source/config/manifest identities, normalization state and
  Adam counters. A failed update retains the last committed checkpoint. A new
  policy uses model-only initialization, not old selector/Adam/normalizer state.
- Physical selected deployment requires fresh dedicated inference-cost
  calibration bound to the final model/actions. Joint updates invalidate
  attached frozen selector/cost artifacts.

## Findings and remediation

| Severity | Finding | Status / consequence |
| --- | --- | --- |
| High | Historical KD cache no longer matches current preprocessing/calibration identities. | Confirmed misses on all seven sampled TRAIN folds. Use a new cache; do not edit old keys or metadata. |
| High | Feeding all 5,738 samples into joint selector updates would consume the existing internal VAL/CAL keyframes. | Prepared manifests from the already sealed FIT/VAL/CAL partition. These are internal experiment roles; dataset_1–7 TRAIN and dataset_8–9 TEST remain unchanged. |
| Medium | Real joint GT callbacks and device transfer dropped resampling provenance. | Fixed: canonical `OfflineMetricTruth.from_a00` verifies declared GT/validity resampling, all real callbacks use it, device transfer preserves it and diagnostics record it. |
| Medium | An unknown-GT snapshot after fitted normalization was logged as unfinished burn-in. | Fixed: fitted state remains accurate; unknown-outcome skips have an explicit reason and do not claim collected supervision. |
| Medium | Nonempty local GT is enough to produce a label, even with one valid pixel. | Evidence: 4/96 known corrected-audit regions have at most 32 local GT pixels, minimum 1, median 1,685.5. This is a statistical reliability limitation; no unvalidated cutoff was introduced. |
| Medium | Repair optimization and selector utility are different objectives. | Repairs learn RGB/warp/local/global/collateral proxies; metric GT supervises selector outcomes, not a differentiable physical GT loss for repairs. Falling repair loss does not guarantee positive metric utility. |
| Medium | Existing DEV frames are not independent keyframes. | 112 DEV IDs do not overlap TRAIN IDs, but share all 27 TRAIN keyframes. They support within-sequence diagnostics, not independent-keyframe generalization. |
| Medium | Warm-start representation already saw the parent TRAIN keyframes. | Recorded explicitly. Internal held-out groups exclude new joint optimizer updates; they are not representation-independent evaluation from scratch. |
| Medium | Risk cutoffs and common-utility pilot are not deployment-calibrated optima. | Stereo 0.51 / Gaussian 0.22 remain historical pilot cutoffs. Prior utility sensitivity measured a conditional safe oracle, not learned selector performance. Assess/calibrate on held-out TRAIN groups after actions stop moving. |
| Low | Training repeats full-image work despite sparse repair writes. | Two repair renders per view, repeated metrics/quantiles, validation readbacks and a second A00 RGB decode for offline GT remain. Previous context/head timing gains do not prove faster full training or production FPS. |

There is no verified optimizer-freezing, stale-label replay, wrong disparity sign,
or selector-gradient leakage defect in the checked path. This statement does not
assert that the selector learns an effective policy.

## Data and initialization checks

Original root: `D:\3DGS_datasets\SCARED`.
Processed root: `/mnt/d/3DGS_datasets/preprocessing/SCARED_A00`.
No data was copied, moved or written into the repository.

The existing 25% cohort contains 5,738 samples from dataset_1–7. Counts:
dataset_1 237; dataset_2 1,085; dataset_3 802; dataset_4 506; dataset_5 930;
dataset_6 1,164; dataset_7 1,014. All cohort mask paths and declared grids were
checked. Seven sampled masks were decoded and rehashed; this audit did not
rehash/decode all 5,738 medical samples.

One deployment/GT sample per fold was loaded independently at 512×640.
Deployment and supervision loads had identical calibration metadata; deployment
samples contained no GT. Resampling identity:
`nearest_exact_half_pixel_centres`.

The sampled dataset_7 frame `dataset_7/keyframe_4/frame_data001748` has only eight
valid GT pixels over the whole frame. This does not indicate missing RGB/camera
inputs; many proposed regions can have unknown GT metrics.

The old cache has 5,738 entries. All seven current sampled keys miss, with both
`image_transform_hash` and `calibration_transform_hash` differing. Teacher,
role and schema identities match. No cache migration or regeneration was done.

The existing partition is preserved exactly:

| Internal role | Samples | Keyframes | New manifest hash |
| --- | ---: | ---: | --- |
| FIT: joint optimizer updates | 2,788 | 13 | `309a0558e5bcee6054d5aea0b7b38ee3da02ba0cec7b09eed4f2036527794572` |
| VAL: read-only assessment | 1,572 | 7 | `a9ee8fd8ec712d375a526745501bc4061bdd731968be029c0915acce477758fb` |
| CAL: post-training risk calibration | 1,378 | 7 | `f0a15bdeae2ae9ec04a7db17fd0769e691be6a4daa495982187c98c9135c0aea` |

Partition identity:
`ddadbea0d6f75f87487db6740125c4e4142147c41e39902792231f363f135fb9`.
Verified derived manifests and binding live under external `assets-verified/`.
This allocates the previously chosen 25% cohort; it does not invent a new split
or silently treat calibration samples as optimizer training examples.

The canonical student constructor, production renderer configuration and
`initialize_student_model` accepted the pinned model-only checkpoint on CPU.
No native rendering/forward was invoked for that check. Checkpoint SHA-256:
`01ee7a0d065127bb50f996e1d9247610e7826c6da54c9bbdcddc3a02679c32c3`.
Its historical source is recorded separately. Repairs, selector, Adam states
and normalization start fresh; all three owners subsequently learn jointly.

This audit checked loader contracts and synthetic geometry. It did not repeat
the previous independent raw metric-GT/calibration or Q-vs-A00 experiment.

## Changed repository files in this pass

- `training/metric_quality.py`: canonical real-A00 offline GT construction.
- `training/balanced_learning.py`: preserve GT provenance and correct skip logs.
- `training/student_runtime.py`, `training/mask_validation.py`,
  `training/selector_refit.py`: consume canonical offline GT construction.
- `tests/routing/test_metric_joint.py`: regression coverage for provenance,
  forbidden roles and unknown-outcome diagnostics.
- This report.

Source paths above are relative to `src/reliable_endo_gs/` except tests/report.
Existing unrelated working-tree/submodule changes and historical artifacts were
preserved. No default utility/risk threshold, split policy or repair objective
was changed in this pass.

## Validation and execution evidence

- **82 passed**, 65.39 seconds: selector optimization, metric joint learning,
  student joint learning and student runtime suites. CUDA ownership/resume cases
  ran on physical GPU0. Renderers were synthetic; these are not native smoke
  results. Log: external `tmux-tests.log`, exit code 0.
- **27 passed**, 28.21 seconds: selector-refit and mask-validation consumer
  suites. Log: external `consumer-tests.log`, exit code 0.
- **39 passed**, 2.49 seconds: surface-neighbour, A03 prior, geometry-repair
  and A00 adapter suites, including boundary masks and sparse GT resize.
  Log: external `geometry-tests.log`, exit code 0. Total: **148 passed**.
- Ruff checks and format checks passed for six edited source/test files.
- Mypy passed five changed source files. `git diff --check` passed.
- Bounded CLI syntax, reviewed source/checkpoint/manifest bindings and shell
  syntax passed. Config/mask input bytes are pinned in `launch-inputs.json`.
- The launcher bootstrap verified physical GPU0 UUID
  `b9607686-60e0-2545-b328-ca3f0e79e24c` against the NVIDIA UUID. NVIDIA
  includes a `GPU-` prefix while PyTorch reports the bare UUID; comparison
  normalizes only that representation. Bootstrap and final binding checks
  exited 0 without invoking training.
- Native rasterizer import and forward/backward symbol checks passed on
  RTX 5070 Ti (CUDA capability 12.0), PyTorch 2.7.1+cu128. The loaded external
  binary SHA-256 is
  `fb18b0abb7f91eefa1751608ff2705fe5bfab105a3835c5bd6193e86c510e23e`.
  No native render/backward was executed; importability is not smoke evidence.
  The external rasterizer source HEAD matches the pinned
  `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d`; its existing header change only
  adds `<cstdint>`. That external change was preserved. The launcher checks
  the binary fingerprint and copies reviewed bindings into the new run.
- Scoped forbidden baseline/SCARED-C/lib imports were searched in the active
  student/action/routing path; none matched.
- Two earlier direct-tool validation attempts were incomplete: one was stopped
  after prolonged waiting, and the subsequent processes disappeared before a
  completion record. Tracebacks showed slow PyTorch/CUDA-library imports. Their
  logs are retained; they are not counted as passes. The successful checks ran
  in new private tmux sessions, without modifying the user's `rendo` session.
- External preparation helpers initially had an attribute/syntax error and an
  overstrict comparison against unresolved reference config. Failed artifacts
  were preserved. The successful verification used the canonical student and
  renderer constructors rather than weakening checkpoint compatibility.

## Bounded check command — corrected after first execution

```bash
bash /home/bkai/.local/share/reliableendo-gs/runs/joint-logic-audit-20261010-txRFIgQq/run_joint_check_gpu0.sh
```

This learns on **128 FIT samples** as a bounded native check, with the full
2,788-sample FIT manifest and unchanged three-joint-epoch schedule. The 128 cap
checks execution/gradients; it is not claimed sufficient for research quality.
The script pins physical GPU0 by NVIDIA UUID, warms up CUDA before the first-party CLI, checks reviewed
source/config/checkpoint/manifest identities, creates a fresh external run/cache,
and explicitly populates missing frozen-teacher targets. It starts no final
TEST evaluation and automatically launches no full epoch.

Before a longer run: inspect successful native backward/checkpoints, all-owner
update norms, number of observed metric regions, normalization completion,
harm per action, utility regression/ranking and paired metric gains on VAL.
Keep CAL out of optimizer updates. Recompute final frozen-action outcomes and
inference costs after joint training; do not reuse the old frozen table/costs.
Three selected action classes alone are not an acceptance criterion.

## Follow-up: FoundationStereo import failure

The first execution created external run `joint-common-gpu0-bQeeoav4` and failed
with `put only this teacher source root on PYTHONPATH`. Its console log and
`training/initial.pt` are preserved. It completed no optimizer update.

The external launcher previously exported only the repository `src` path.
FoundationStereo is loaded lazily on a cache miss, and its isolated `core`
namespace requires the pinned external teacher source root. Artifact-only
checks and bounded CLI parsing did not establish actual module importability.

Changes in this follow-up are the external launcher and two external helpers,
plus this report. First-party scientific Python source is unchanged:

- `run_joint_check_gpu0.sh` resolves the source root from the pinned teacher
  runtime configuration and exports repository `src` plus that one teacher
  source root. It runs teacher import preflight before creating a new run/cache.
- `check_teacher_import.py` verifies artifact hashes, isolated namespace,
  dependencies, the actual `core.foundation_stereo` import origin and constructor
  symbol. It does not deserialize a checkpoint, instantiate a model or infer.
- `run_teacher_import_check.sh` provides a standalone check with GPUs hidden.
- GPU inspection and preparation commands have explicit timeouts. A timeout
  prevents training launch, but cannot guarantee termination of a process stuck
  in an uninterruptible kernel wait.

Validation completed: Bash syntax, Python helper syntax and unchanged reviewed
first-party source hash. Independent standard-library namespace resolution
located `core.foundation_stereo` inside the pinned external source. Evidence:
external `teacher-namespace-static-check.json`.

Actual module-import verification did **not** complete: both the audit's
`nvidia-smi` process and GPU-hidden PyTorch import were observed in Linux `D`
state at `dxgglobal_acquire_process_adapter_lock`. This is a separate observed
WSL GPU-access blocker, not proof of the cause of the earlier namespace error.
Termination was requested for only these two audit processes; an uninterruptible
wait can retain them until the kernel call returns. Other users' processes and
tmux sessions were left intact. No GPU/WSL reset, dependency installation,
teacher inference or new training was performed in this follow-up.

The corrected command above is ready to retry **after WSL GPU inspection
responds**. Runtime import, native backward and joint learning remain unverified
for the corrected launcher; do not interpret the static checks as a train pass.
