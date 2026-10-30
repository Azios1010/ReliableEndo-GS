# Plan 17: selector training (Stage B) on the frozen student and repairs

2026-10-28 amendment: the user authorized implementation of the corrected opt-in
development protocol in [Plan 18](18_frozen_repair_selector_corrections.md).
Its split/seed, automatic-region, utility and gate rules supersede the draft
details below for new runs. No new Stage B scientific run has been completed.

Historical draft status (2026-10-27): **PROPOSED; no Stage B run had launched**.
Current implementation authority is the amendment above; execution still requires its gates.
Development-only: Gate I is not certified and Gate II is not passed.

This plan continues `plans/16_baseline_and_repair_action_runs.md`. Plan 16 trains the repair actions
(Stage A) with the KD student frozen. This plan covers the learned selector that chooses STOP, stereo
repair or Gaussian repair per region. The user described this part as theirs; Claude prepares and
verifies, and no stage runs without an explicit instruction to run it.

## Position in the program and gates

```text
Plan 16: A0 checks -> A1 repair training -> A2 verify -> A3 / Gate A
Plan 17: B0 prerequisites -> B1 stationary outcome collection -> B2 distribution audit and Gate B
         -> B3 controlled variant comparison -> B4 decision and operating points -> B5 handoff
```

- Stage B may start only after Gate A passes and the repair artifact (student, head and both repair
  heads) is frozen and hashed. **Any later change to the student or repairs invalidates every Stage B
  label, fit and calibration**; the existing audits state that a calibrated head is valid only for the
  exact artifact it was fitted on.
- Gate B (below) is the repository's "oracle shows heterogeneous utility" check (AGENTS.md Gate II).
  If it fails, the selector is not trained and the system keeps a fixed action; the negative result is
  recorded and the claim changes, not the evidence.
- final_test (datasets 8 and 9) is outside this plan.

## Why the selector matters for the contribution

The contribution is high FPS without a quality drop. Repair costs about 100 ms per frame (estimated, see
plan 16), so the selector's job is to pay for repair only where it helps. It therefore has to beat three
simple alternatives on the quality-versus-latency curve: never repair (student only), always repair, and
always apply the single best fixed action. If it cannot, AGENTS.md applies: reconsider the claim and take
an approved simplification (fixed repair, regional repair) rather than forcing the selector.

## What already exists (read on 2026-10-27, not yet exercised on the KD student)

- `python -m reliable_endo_gs.cli.selector_refit` takes the frozen checkpoint with its SHA-256, data root,
  training and development manifests, roles, worklist, target-mask index, refit config and a new output
  directory. Modes: `--preflight-only`, `--prepare-costs` (GT-free native cost calibration),
  `--collect-only` (collect frozen outcomes and stop before fitting; this is what makes "collect once,
  fit many" possible) and `--resume` (reuse verified immutable shards).
- Partition `stratified_keyframe_hash` (`configs/routing/selector_refit_quarter.yaml`): within each
  training dataset, SHA-256 ordering assigns one whole keyframe to validation, one to calibration and the
  rest to FIT, sealed before any GT or outcome is decoded. For `train_1236.json` (datasets 1, 2, 3 and
  6; 15 keyframes) this should give about 7 FIT, 4 validation and 4 calibration keyframes (to be
  confirmed at preflight).
- Fitting uses FIT only for normalization, validation to stop, calibration for monotone gain and Platt
  risk corrections. Calibration needs at least 8 known regions with safe and harmful labels each
  present in at least 2 keyframes; otherwise the action abstains (never certified safe). Unknown
  labels are excluded, never counted safe. Utility weights, physical guards and cutoffs stay fixed.
- Selector variants are config switches: `intervention_metric_geometry_frozen_student.yaml` (61-dim
  features, one union-risk head per action), `..._mask_tokens_frozen_student.yaml` (mask tokens,
  `selector_risk_mode: local_global`), `..._signed_gain_frozen_student.yaml` (adds
  `selector_gain_mode: signed_expectation`).

## Data design (proposed)

| Role | Source | Use |
| --- | --- | --- |
| FIT / VALIDATION / CALIBRATION | `train_1236.json`, whole keyframes via `stratified_keyframe_hash`, sealed first | Fit, stop, calibrate the selector heads |
| Development evaluation | `val_7.json` (4 keyframes, 4,056 frames), never trained on by the student | One reported result per arm; never used to fit, stop, calibrate or pick a cutoff |
| final_test | datasets 8 and 9 | Untouched |
| Excluded | datasets 4 and 5 | 17 to 22 px vertical misalignment |

Risks of this design, stated up front:

- The student trained on every FIT, VALIDATION and CALIBRATION frame, so its errors there are smaller
  than on unseen frames and the harm/utility distribution may differ from deployment. Plan 16 step A0.8
  measures the gap. If the gap is larger than the pre-declared thresholds (left PSNR above 0.5 dB or
  depth MAE above 1.25 times), the fallback is leave-one-keyframe-out over the 4 val_7 keyframes only,
  reported separately from the in-sample fit, accepting far less data.
- There are very few independent units: 15 train keyframes (7/4/4) and 4 development keyframes.
  Adjacent frames are near duplicates, so confidence intervals are clustered by keyframe and are wide.
- The earlier cohort (56 frames from 14 keyframes) showed stereo harm 8/14 in CAL versus 5/30 in FIT;
  the same imbalance can recur.
- The worklist is bounded, not the whole 12.9k frames. Its size is fixed from the A0 cost measurement
  (target: outcome collection of a few hours) before any outcome is seen, stratified by keyframe, with
  the sample IDs sealed and hashed.

## The ROI problem (decision for the user)

The refit CLI requires a target-mask index. In the existing path it comes from an operator protocol that
visually reviews 16 temporal RGB anchors per keyframe, and other frames inherit the nearest anchor;
these are coarse operator tissue ROIs, not a validated segmentation (`docs/selector_quarter_readiness.md`).
For the 15 train and 4 development keyframes that is 19 x 16 = 304 anchors to review. **Operator ROIs
are not an automatic producer**, so a system that uses them is not end-to-end in the sense of the
paper's claim. Options:

1. Use operator ROIs for fitting and development, and report end-to-end results with that dependency
   disclosed, plus an ablation with an automatic ROI proxy (for example predicted or sensor support)
   to bound the effect. Needs the review work (the user's, unless delegated).
2. Use the historical fixed-grid regions with no ROI. Fully automatic, but the grid-path selector
   evidence is the older one, and it needs a check that the refit CLI can run without a mask index.
3. Build an automatic ROI producer from deployment RGB. Not implemented; a separate task.

Recommendation: option 1 for the first controlled comparison, with the automatic-proxy ablation
pre-declared, and a clear statement of the dependency in any claim. The user chooses.

## Stages

- **B0 prerequisites (no GPU or small).** Gate A passed; repair artifact frozen and hashed; ROI decision
  made and index produced; worklist, partition and role hashes sealed; `--preflight-only` passes on the
  new student checkpoint (compatibility with a Stage A frozen-student artifact is unverified).
- **B1 stationary outcome collection (idle GPU preferred).** `--prepare-costs` for the GT-free measured
  action cost artifact, then `--collect-only`: execute STOP, stereo repair and Gaussian repair on every
  bounded candidate region with the frozen artifact, compute deployment features before loading GT,
  then separate GT physical metrics (all 12 components), persist full outcomes including unknowns.
  Check up front that the stored features suffice to refit all three variants (mask tokens need token
  features); if they do not, store them in this pass rather than recollecting later.
- **B2 distribution audit and Gate B (before any fitting).** Write the table the earlier selectors
  lacked: per role (FIT, VALIDATION, CALIBRATION, development), per action, per event (local, global,
  any) the known count, harm rate, unknown rate and utility quantiles; the same per keyframe. Pre-declared
  Gate B: each non-STOP action is the oracle-best on at least 10% of known regions, and the oracle mean
  utility exceeds the best fixed policy by at least 10%; calibration support is reported per action
  (unsupported means abstain). Thresholds are proposed and need approval. A role with fewer than 2
  keyframes of a class is reported, not repaired. If Gate B fails, stop.
- **B3 controlled variant comparison.** Fit all arms from the same outcome shards, partition and three
  predeclared seeds (0, 1, 2): C0 geometry/union risk (control), C1 mask tokens with local/global risk,
  C2 signed gain with local/global risk. Optionally C3, a render-first utility arm, only if the user
  approves it and the stored outcomes allow recomputing utility without recollecting (unverified).
  Mandatory references: STOP, always-stereo, always-Gaussian, oracle. Report per action and per event:
  Brier, ECE, AUROC, selected-action harm rate, harm type, unknowns, achieved utility, guard
  violations, and both-view render quality (left PSNR, SSIM, LPIPS first). Candidate and selected
  populations are reported separately.
- **B4 decision and operating points.** The variant is chosen on VALIDATION and CALIBRATION, not on the
  development result. A selector is promoted over the fixed policies only if, on development, its mean
  utility exceeds the best fixed policy with a keyframe-clustered 95% interval excluding zero, its guard
  violation fraction is no higher, and this holds in at least 2 of 3 seeds. Otherwise the result is
  NO-GO, the fixed action (or STOP) is kept and the negative evidence is preserved. Risk cutoffs come
  from calibrated probabilities on CALIBRATION with a pre-declared harm budget (no larger than the best
  fixed policy's harm rate), not from hand tuning, and 3 to 5 budgets give a quality-versus-latency
  curve fixed in advance.
- **B5 handoff.** Record the selector artifact bound to the model and repair hashes, config and
  partition, the audit tables, the comparison report and the decision. Latency and FPS of the selected
  pipeline come only from the idle-GPU measurement on the final artifact (plan 16 A0.9 protocol),
  including region extraction, selector inference, repair, re-render and memory.

## Pre-registered rules (to be hashed at launch)

1. Frozen artifact = the Stage A output; no later change without discarding Stage B.
2. Worklist, partition, seeds, arms, thresholds and operating-point budgets are fixed before outcomes are
   seen; no cutoff or utility weight is tuned on development or final_test.
3. Unknown outcomes are excluded and reported; unsupported calibration abstains.
4. Gate B and the B4 rule decide go or no-go; a no-go is not retuned.
5. Results are development-only; no superiority or FPS claim is made from them.

## Compute estimate (formulas, to be filled by plan 16 A0)

Outcome collection is about (frames) x (3 actions evaluated on all candidate regions) x (re-render of
both views, about 100 ms per action estimated) plus GT metrics; the worklist size is chosen so this takes
a few hours. Fitting the small heads (about 1.7k parameters each) takes minutes per arm and seed; the
three arms x three seeds are cheap. The operator ROI review is manual time for 304 anchors if option 1 is
chosen. Idle-GPU windows are needed for the cost artifact and the final latency measurement.

## Risks and limits

- The ROI index is operator-made (see above) and the system is not fully end-to-end with it.
- Student errors on FIT, VALIDATION and CALIBRATION are in-sample; the audit and A0.8 quantify but do not
  remove this.
- Few keyframes; wide intervals; a single student seed.
- The current utility is geometry-first (`stereo_weight: 3.0`, `geometry_priority: 1.5`, collateral and
  local-render weights 0) while the paper's claim is render quality plus FPS; the first comparison keeps
  it unchanged (as the audits advise) and reports render metrics next to it.
- The decision policy cutoffs are hand-set and asymmetric in the existing configs; this plan replaces
  them with calibrated, pre-declared budgets, which is a deviation to disclose.
- val_7 is also used in Gate A (plan 16); it is not touched for any fitting decision, but repeated looks
  at it must be counted when reporting.

## Out of scope

Training or changing the student or repairs, final_test evaluation, the paper-protocol reproduction,
extra student seeds, building an automatic ROI producer, and any claim about FPS or superiority.
