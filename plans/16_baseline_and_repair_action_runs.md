# Plan 16: matched baseline completion and repair-action training

2026-10-28 amendment: the user authorized the corrected development implementation
in [Plan 18](18_frozen_repair_selector_corrections.md), including opt-in automatic
regions for Stage A and keyframe-based Gate A uncertainty. The baseline recipe is
unchanged. This amendment does not record completion of a new Stage A experiment.

Historical draft status (2026-10-27): **PROPOSED; nothing had launched at drafting**.
Current implementation authority is the amendment above. Development-only:
Gate I is not certified and Gate II is not passed.

## Scope and authority

The user asked for a plan covering two runs that Claude executes:

- **Track B**: finish the matched Endo-E2E-GS baseline and prepare its evaluation.
- **Track A**: train the Phase II repair actions with the KD student frozen ("Stage A").

Everything else belongs to the user and is out of scope: selector training (Stage B), opening
final_test, additional baselines, extra seeds, paper-protocol reproduction and the paper itself.

Authority and precedence (AGENTS.md order): the user's explicit instructions override the historical
gate order, as already recorded in `docs/local_student_joint_handoff.md`. The sequence follows the
user correction of 2026-10-20 in `docs/student_joint_learning.md`: KD student, output-quality
evaluation, explicit student/head freeze, then repair-action learning. **Stages are never chained
automatically and no training starts without an explicit user instruction to run it.**

Existing pre-registration that this plan must not alter: `COMPARISON_PLAN.md` in the baseline run
directory (sha256 prefix `c0378cf0`): final-vs-final checkpoints only, no per-epoch best selection,
dataset 7 for monitoring only, final_test (datasets 8 and 9) evaluated once per model after both
finish, and a metric-wise claim rule (both datasets, paired interval excludes zero). Changes may only
be dated addenda written before any matched-baseline result exists.

## Contribution under test (user statement, 2026-10-27)

The contribution is an end-to-end inference system with high FPS whose quality does not drop: a
distilled feed-forward student replaces the slow teacher-based pipeline, and repair actions (applied
only where useful) recover the quality lost by distillation. Consequences for this plan:

- The claim is a **speed/quality trade-off, not "better on every metric"**. The quality part is a
  non-inferiority claim against a margin delta per headline metric (not yet chosen; the user sets it
  before final_test is opened), together with a measured speed-up. The superiority rule in
  `COMPARISON_PLAN.md` stays in force for any "better than" statement.
- Systems to compare, on the same frames, resolution and idle GPU: (1) teacher-backed A04 pipeline
  (quality reference, slow end); (2) student alone; (3) student + repair (always-stereo,
  always-Gaussian, later the selector); (4) the matched Endo-E2E-GS baseline, the main competitor in
  both FPS and quality.
- FPS evidence follows AGENTS.md: synchronized, warmed-up, idle GPU, hardware recorded, median and p90
  over many frames, whole pipeline (student, geometry and Gaussian construction, region and selector
  work, repairs, both-view rendering) and memory. Parameter counts, FLOPs and selected pixels are not
  evidence.
- Numbers that exist today (not a benchmark of this system): the A04 pipeline with the live teacher
  measured 498 ms per frame (2.0 FPS; one frame, 5 repetitions, GPU not isolated, 3.0 GB peak). Repair
  actions have an estimated mean cost of about 98 to 105 ms per frame in
  `docs/selector_controlled_comparison.md`, and currently render both complete frames. The student's
  own inference latency has never been measured. Whether student + repair is faster than the baseline,
  and whether repair erases the speed gain, is therefore **unknown**; it is the cheapest decisive
  measurement and is scheduled first (A0.9).
- If repair cost removes the speed advantage, the AGENTS.md rule applies: reconsider the claim and
  follow an approved simplification (regional repair, repair only where it pays, or the student with a
  fixed repair) instead of forcing the original hypothesis.

## Defaults requiring confirmation before launch

| # | Decision | Default in this plan | Why |
| --- | --- | --- | --- |
| 1 | Student frozen for Stage A | `final.pt` of `kd-train-1236-20261024T192708Z/attempt-15` (sha256 prefix `f77547ef`) | Matches the pre-registered final-vs-final rule. Epoch 4 had higher left PSNR on val_7 (+0.44 dB, 95% bootstrap CI +0.30 to +0.56) but choosing it would need the same dev-based selection applied to the baseline. |
| 2 | Baseline stage-1 length | 100,100 updates, then stage 2 for 10,000 updates (horizon 10,100) | Stopping at 60,000 leaves the OneCycle LR at about 40% of peak (hard-coded 100,100 horizon), a handicapped baseline. The decision is the user's and, per the pre-registration, is made before update 60,000 and not revisited after results. |
| 3 | GPUs | Baseline on GPU 1, Stage A on GPU 0 if A0 measures enough free VRAM, else queue Stage A behind the baseline | GPU 0 carries other jobs (`e2z_lane.sh`, `--gpu 0`). Both trainers pin a physical GPU by UUID. |

## Track B: matched baseline

State on 2026-10-27 (read from the run directory): `endo-baseline-a00-train1236-20261026T171147Z`,
2,141 of the stage-1 updates done, stopped gracefully at 01:13 local, newest checkpoint
`train/ckpt-s1-002141.pt`, no process running. `train/config.json` has `stage1_updates=60000`,
`stage1_updates_confirmed=false`. The trainer reads `config.json` every loop iteration; when stage 1
reaches `stage1_updates` it pauses with `awaiting_stage1_decision` unless `stage1_updates_confirmed`
is true, and `stage1_updates` cannot exceed the frozen horizon 100,100.

- **B0 decision record (no GPU).** Back up `train/config.json`, set `stage1_updates=100100` and
  `stage1_updates_confirmed=true`, and append the decision with its time to the run record. The
  frozen recipe keys (`seed`, `lr`, `weight_decay`, `eps`, `clip`, horizons, `output_hw`,
  `corr_implementation`) are not touched. If GPU 1 is used, change the two hard-coded GPU 0 constants
  in operational code outside `src/` (`envrun.sh`, `GPU_UUID` in `supervise_baseline.py`) and record
  both file hashes in `launch-hashes.sha256`. Resume does not validate the GPU UUID (it is only
  recorded in `run.json`), but a GPU change means the continuation is not bit-identical.
- **B1 train.** Relaunch `supervise_baseline.py` in the baseline's own tmux socket (never touch the KD
  socket). It resumes from the newest checkpoint. Remaining work is about 97,959 stage-1 plus 10,000
  stage-2 updates. The logged 0.19 to 0.25 s per update did not hold on 2026-10-27: the run measured
  0.64 s with 6 loader workers (loader threads waiting on 9p reads from `/mnt/d`; model step only
  0.037 s) and about 0.32 s with 16 workers and prefetch 32 (steady state, 4,015 updates), so expect
  **about 9.5 hours, with a plausible range of 8 to 13**, and re-measure on a longer sample. A longer
  sample on 2026-10-28 (3,239 to 22,560 updates in 1 h 36 min, GPU 1 alone with the other jobs on GPU 0)
  gave about 0.30 s per update, i.e. about 7.2 hours remaining from update 22,560. Checkpoints
  every 500 updates. Dataset 7 is monitoring only. The
  final checkpoint is the end of stage 2 (`result.json` state `complete`).
- **B2 evaluation adapter (code task, needs user approval before writing).** The frozen
  `benchmark a04` command takes an A04 reference checkpoint, so it cannot evaluate this baseline
  directly. The adapter must run the trained baseline through the same metric functions and
  aggregation as the harness: left/right PSNR, windowed SSIM, LPIPS (AlexNet v0.1 local assets),
  disparity EPE, depth MAE/RMSE and the geometry metrics, on the same sample IDs, with the same
  first-party renderer (`ProductionFieldRenderer`), per-frame JSONL fields and provenance (commit,
  config hash, checkpoint sha256). Existing pieces: `baseline/evaluation/a00_small.py`
  (`evaluate_a00_small`) already computes PSNR, SSIM and geometry with repository functions. Not yet
  verified: LPIPS, harness-identical aggregation, and that it accepts the upstream-unmodified model
  checkpoints written by `train_baseline.py`. Validation: synthetic tensors give identical metric
  values through the adapter and through the harness functions. Keep it thin under `scripts/` or in
  `baseline/evaluation/`, with no duplicated formulas.
- **B3 dev table (val_7, 56 fixed samples).** Run both models through the same evaluator. This is
  monitoring, never used to choose a checkpoint. The baseline's latency uses the A0.9 protocol (same
  GPU, idle, same resolution) so FPS and quality are compared on the same footing.
- **B4 final_test is not part of this plan.** The user opens it, once per model, after both runs are
  complete, per the pre-registration. This plan only prepares and verifies on val_7.

## Track A: repair actions with the student frozen (Stage A)

- **A0 preparation and checks (small GPU use, scratch output only).**
  1. Source snapshot of the working tree including the files that exist only untracked
     (`reconstruction/stereo_features.py`, `training/student_schedule.py`,
     `configs/training/student_kd_multiscale.yaml`), with its sha256. Nothing is committed.
  2. New config `configs/training/student_frozen_repair.yaml`, copied from
     `student_kd_multiscale.yaml` so the architecture matches the checkpoint (`stereo_architecture:
     multiscale`, `width: 64`, `bins: 64`, `minimum_effective_px: 0.25`), with `student_epochs: 0`,
     `joint_epochs: 1`, `kd_weight: 0.0`, `reconstruction_weight: 1.0`, `repair_weight: 1.0`, a fixed
     seed. KD weight 0 needs no teacher.
  3. Intervention config: existing `configs/routing/intervention_metric_geometry_frozen_student.yaml`
     (`student_update_mode: frozen`, schema `intervention_selection.v2`), unchanged.
  4. Load check: `--initialize-model-from final.pt` is accepted (positive model Adam steps, matching
     ownership) and records `checkpoint_model_only` provenance.
  5. Equivalence: base outputs from the snapshot reproduce the recorded `final.pt` val_7 metrics on
     the 8 samples within tolerance.
  6. Bounded frozen smoke on at most 3 samples: student/head parameters, buffers and model Adam bitwise
     unchanged; both repair heads changed; finite losses.
  7. Measure seconds per sample and peak VRAM, replacing the old 1.3 s/sample estimate (taken from the
     2026-10-13 comparison with an older student).
  8. Generalization gap of `final.pt`: base-only evaluation on a train_1236 subset versus val_7 with the
     same script. Informs Stage B data only. Proposed thresholds (need approval): the gap is "small" if
     left PSNR differs by at most 0.5 dB and depth MAE by at most 1.25 times.
  9. **Inference latency feasibility profile.** *Deferred by the user on 2026-10-28 ("no FPS run yet");
     A0 proceeds without it and it stays a required step before any efficiency claim.* On an idle GPU, with warm-up and
     synchronization, median and p90 over at least 200 val_7 frames at batch size 1: student forward,
     geometry and Gaussian construction, left/right render, stereo repair, Gaussian repair (including
     any re-render) and selector inference, per component, with peak memory and hardware. Repeat the
     same protocol for the matched baseline on any checkpoint of its architecture (inference time is
     not expected to depend on trained values; confirm by re-measuring on the final checkpoint).
     Purpose: decide before the multi-hour runs whether high FPS is attainable and which component
     dominates. It changes no weights and does not open final_test.
- **A1 train.** `student train` in frozen mode on `train_1236.json` (13,150 samples), 1 epoch. Estimate
  about 4.7 hours at 1.3 s per sample, to be replaced by the A0 measurement. The existing frozen mode
  requires `selector_updates_per_snapshot` between 1 and 16, so the selector is trained alongside the
  repairs. **That selector is discarded:** not evaluated, not promoted, and refit in Stage B on
  stationary outcomes. Run under a resume supervisor modelled on the KD one (disk reserve, bounded
  restarts, prune). Launch template (flags are verified against `--help` in A0, not run now):

  ```bash
  python -m reliable_endo_gs.cli.student train \
    --config configs/training/student_frozen_repair.yaml \
    --construction-config configs/reconstruction/a04_reference.yaml \
    --routing-config configs/routing/frozen_a04.yaml \
    --intervention-config configs/routing/intervention_metric_geometry_frozen_student.yaml \
    --roles configs/teachers/a01_scared.yaml \
    --manifest "$P/train_1236.json" --data-root "$DATA_ROOT" --gpu-input \
    --initialize-model-from "$KD_RUN/attempt-15/training/final.pt" \
    --output-dir "$OUT/stageA"
  ```
- **A2 verification.** Student/head tensors and model Adam equal the initialization (hash); repair
  heads changed; finite; provenance records the initialization path and sha256.
- **A3 repair evaluation on val_7 (Gate A).** For each of STOP, always-stereo and always-Gaussian
  (fixed policies, which are also the AGENTS.md "fixed refinement" and single-source baselines): mean
  change versus STOP in left PSNR, SSIM and depth MAE, guard-violation fraction (guards from the
  config's `metric_quality`), reported separately per action. No hyperparameter is tuned on val_7.
  **Gate A passes** if at least one non-STOP action improves mean left PSNR over STOP with a
  frame-level 95% bootstrap interval excluding zero and the gain has the same sign in at least 3 of the
  4 val_7 keyframes. Its violation fraction is reported with a proposed ceiling of 25% (thresholds are
  mine and need approval). If Gate A fails, stop and report the negative result; do not retune.
- **A4 handoff.** Record student, repair-head and config hashes, the split hashes and the result
  table. Stage B (selector) and any final_test use are the user's.

## Pre-registered rules for this plan (to be hashed at launch)

1. Frozen student = decision 1; no epoch selection.
2. No tuning on val_7 or final_test; Stage A is a single run with fixed hyperparameters.
3. The Stage A selector is discarded; nothing about routing is claimed from Track A.
4. Gate A thresholds as in A3 (once approved). A failed gate is reported, not retuned.
5. Results are labelled development-only; repair claims are limited to what A3 measures.
6. Output directories are new and external; successful outputs are never overwritten.
7. The speed/quality claim is non-inferiority with margins fixed by the user before final_test is
   opened. FPS comes only from the idle-GPU measurement of A0.9, repeated on the final checkpoints,
   never from training logs.

## Scheduling and operations

| Item | GPU | Estimate | Notes |
| --- | --- | --- | --- |
| B0 + B2 | none | hours of work | B2 needs approval |
| B1 baseline train | 1 | 6 to 7.5 h | unmeasured under contention |
| A0 checks | 0 or 1 | about 1 h | measures real cost and VRAM |
| A1 train | 0 | about 4.7 h (to be measured) | resume supervisor |
| A3 evaluation | any | minutes to an hour | |

On two GPUs the tracks run in parallel (about 7 h wall clock); on one GPU, sequentially (about 12 to
13 h). During the previous run the WSL distro was powered off from outside about every few hours, so
every trainer runs under a resume supervisor (relaunch from the newest checkpoint) and the user should
keep a Windows terminal open (for example `wsl.exe -d Ubuntu --exec sleep infinity`). Close the
`watch -n 0.5 nvidia-smi` panes: a stalled GPU call blocked the last trainer start for about 13
minutes. Do not call `nvidia-smi` while a trainer starts.

## Risks and limits

- The student trained on all of `train_1236`, so Stage A labels come from in-sample frames; A0.8
  measures the gap but cannot remove it.
- Selection-free `final.pt` is weaker on the left view than epoch 4 (-0.44 dB) and stronger on the
  right view; this is disclosed, not tuned away.
- One seed per model; intervals are over frames and keyframes, not seeds.
- The baseline is trained with GT-disparity supervision and a different recipe; compute is not matched.
- The upstream protocol (within-sequence every-8th-frame test split, PSNR averaged over all pixels
  after masking) is different from this repository's, so published numbers are not comparable.
- The working tree is dirty; Stage A must run from the A0 source snapshot, and its identity records the
  snapshot hash.
- The efficiency claim is central to the contribution and is **not yet supported**: no student latency
  exists, and repair is estimated at about 100 ms per frame. A0.9 is the first test. Final FPS numbers
  still need the idle-GPU measurement on the final checkpoints; this repository's GPUs are shared with
  other jobs, so an idle window must be arranged.

## Selector version for Stage B (review requested by the user; Stage B itself is not run here)

Read on 2026-10-27 from the three frozen-student configs and the 2026-10-13 audits.

- **Variants.** `intervention_metric_geometry_frozen_student.yaml` (61-dim features, one union-risk
  head per action), `..._mask_tokens_frozen_student.yaml` (mask tokens plus `selector_risk_mode:
  local_global`) and `..._signed_gain_frozen_student.yaml` (mask tokens, local/global risk and
  `selector_gain_mode: signed_expectation`). All three have identical physical utility, guards,
  cutoffs and repair-loss weights, so **Stage A repair training is the same under all three**; the
  Stage A selector is discarded. Stage A therefore uses the geometry config (cheapest features) and
  does not commit the Stage B choice.
- **Why harm and utility distributions differ (documented, none solved).**
  1. Harm is action-specific. Joint stereo harm was 39/39 local-only (depth MAE/RMSE, local RGB);
     Gaussian harm mixes 22 local-RGB with 9 global/collateral. The active selector learns one
     union event per action, so selected Gaussian predicted 7 to 10% harm versus 18 to 20% actual, and
     harmful versus safe selected stereo scored 30.8% versus 34.6% (no discrimination). Candidate
     union-harm AUROC was 0.44 to 0.72.
  2. Utility adds unlike units (depth mm, PSNR dB, SSIM, RGB) with `metric_scales` from 1.0 to
     0.0001, geometry-first (`stereo_weight: 3.0`, `geometry_priority: 1.5`, collateral and local
     render weights 0). Gaussian repair cannot change depth, so the two actions are scored mostly on
     different terms. This is also geometry-first while the stated objective (2026-10-17) and this
     paper's claim are render quality plus FPS.
  3. The decision policy is hand-set and asymmetric: harm cutoff 0.51 for stereo versus 0.22 with a
     0.5 guard multiplier for Gaussian.
  4. Cohorts were tiny and keyframe-clustered: the refit cohort was 56 frames (FIT/VAL/CAL 32/8/16
     from 8/2/4 keyframes), both VAL keyframes in dataset_6, stereo harm 8/14 in CAL versus 5/30 in
     FIT. Calibration improved the CAL loss but worsened development (stereo Brier 0.222 to 0.326).
  5. Labels were non-stationary while actors trained concurrently; Stage B uses stationary labels.
- **No variant has passed.** Joint selector, separate heads with stationary refit (140 frames:
  utility 0.129 versus 0.132, 16 violations), signed gain (0.0997 versus 0.1022; Gaussian-only 0.1037
  beat it), the 28-frame refit (NO-GO: fixed Gaussian 0.178 beat 0.169). Mask tokens have never run
  natively. All of this was on older students and small cohorts, not the KD student.
- **Recommendation.** Compare geometry/union (control), mask_tokens + local_global and signed_gain +
  local_global on the same stationary outcomes with leave-keyframe-out folds over the 19 keyframe
  groups of train_1236 and val_7, against STOP, fixed stereo, fixed Gaussian and the oracle, reporting
  per-action per-event Brier, calibration, harm type, unknowns, utility and both-view render quality.
  Keep utility weights and thresholds unchanged in the first comparison (as the audits advise), and
  decide separately whether to add a render-first utility arm. Calibrate cutoffs on CAL, not by hand.
- **Not available for the new student yet (unverified or missing).** New cohort manifests; the target
  mask/ROI index and the TRAIN cost artifact that the refit CLI requires (the existing path includes an
  RGB-review step, `selector_prepare review/finalize`); and whether one outcome store holds the
  features needed to refit all three variants (mask tokens need token features). These must be
  checked before Stage B and are not done here.

## Out of scope

Stage B selector training and variant comparison, any final_test evaluation, additional baselines,
extra seeds, reproduction under the paper's own protocol, and paper writing.
