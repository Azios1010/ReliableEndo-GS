# Joint-first audit with a separate frozen-student comparison

## User direction

The user prioritizes joint training and explicitly requested a separate frozen
branch to compare effects. Joint remains the default experimental configuration.
This controlled diagnostic does not certify historical research gates or replace
the approved program with a permanently frozen student. Original SCARED TRAIN
1–7 / final test8–9 and immutable manifests remain unchanged.

## What the audit establishes

The preceding audit is preserved in
[joint geometry/selector audit](joint_geometry_selector_audit_2026-10-13.md).
It showed useful stereo and Gaussian opportunities and three-action selection.
It did not establish reliable harm prediction or that student joint updates are
the sole cause of selection failures.

A further action-specific audit changes the emphasis of the diagnosis:

| Selected action, raw geometry checkpoint203 | Stereo | Gaussian |
| --- | ---: | ---: |
| Selected with known metric outcome | 38 | 87 |
| Selected with unknown metric outcome | 0 | 4 |
| Actually harmful selections | 2 | 18 |
| Mean predicted risk among known selected actions | 6.11% | 6.98% |
| Actual harm fraction among known selected actions | 5.26% | 20.69% |

Of 20 selected guard violations, 18 are Gaussian. Fourteen of those violate local
RGB, with some additional whole-view PSNR/SSIM/outside RGB failures. The earlier
context8variant similarly has16Gaussian and4stereo violations;14Gaussian
violations are local RGB. Overall stereo candidate risk underestimation does
not establish that stereo dominates deployed harm. Candidate and selected-action
populations must be reported separately. Unknown outcomes are not counted safe.

### Loss, utility and risk logic

- Both views have reconstruction supervision. Local reconstruction uses the
  left region and its **pre-action projected right footprint**, not an assumed
  identical left/right pixel box. Fixed support prevents changing denominators.
- Student objective includes RGB, warp, KD, repair and the opt-in metric depth
  term. Repair gradients can enter student/head in joint mode. Changing only
  scalar gradient norms does not resolve conflicting directions.
- Six recorded geometry-run probes have RGB/geometry and repair/geometry
  negative cosine in 4/6 cases, KD/repair in 4/6. These are sparse observations,
  not proof of a universal gradient conflict or causal attribution.
- Physical decision utility and harm are distinct. Current physical utility
  contains global image/geometry terms, with local render/collateral utility
  weights zero. Gaussian does not alter stereo depth, so its gain mostly comes
  from global render changes. The raw harm guard still checks local RGB.
  An action may have positive utility yet fail local preservation; this is an
  intentional constrained-decision contract, not proof of a sign error.
- The local gain auxiliary is local geometry. It is identically zero for the
  current Gaussian action, which changes surface rotation/scale/opacity and
  leaves centers/stereo depth fixed. Thus Gaussian local-render safety relies
  on repair losses and the binary risk head, not its local-geometry gain output.
- Independent action gain/risk heads and independent clipping mean a large gain
  scalar does not automatically consume the risk head's gradient clipping budget.
  Risk BCE is trained; it is not an untrained hand-written risk formula.
- Candidates, features and outcomes change with actor weights. Fresh measurement
  prevents stale labels but does not make the learning target stationary.
  Frozen-student and joint comparisons are needed to measure that effect.

These facts justify examining Gaussian conditional risk, local render tradeoffs,
label coverage and temporal outcome drift. They do not justify changing utility
weights/cutoffs merely to force STOP/stereo/Gaussian counts.

### Evaluation/data limits

Current FIT/VAL/CAL partitions are disjoint, but the older backbone ancestry
contains all 140 diagnostic VAL and all 1,378 CAL images. This experiment is therefore
**development-only**. It does not use final test 8–9. Earlier data audit verified
units/grid/resampling on bounded samples; unusual GT extrema have no established
sentinel interpretation and are not silently filtered. No dataset is copied or
modified, and no teacher inference is needed for the comparison.

## Two explicit branches

`InterventionConfig.student_update_mode` has two values:

| Branch | Student stereo + Gaussian head | Repairs | Selector |
| --- | --- | --- | --- |
| `joint` (default) | learn | learn | learn |
| `frozen` (opt-in) | weights/Adam fixed, eval mode | learn | learn |

Configurations:

- Joint: `configs/routing/intervention_metric_geometry_supervision.yaml`.
- Frozen: `configs/routing/intervention_metric_geometry_frozen_student.yaml`.

Only `student_update_mode` differs. Frozen requires balanced v2 and excludes
student-only warmup. Neutral joint mode is omitted from historical config
serialization, preserving historical hashes and strict resume.

### Matched gradient weighting and checkpoint ownership

The frozen branch deliberately retains **virtual student derivatives** for the
same shared-gradient balancing probes and input Jacobians through the fixed
Gaussian head. It excludes student from optimizer updates, keeps it in eval
mode, and discards its gradients after each transaction. This avoids changing
the balancing reference from student parameters to repair parameters, which
would introduce an extra intervention into the comparison. This implementation
claims frozen learned state, not minimum autograd memory/compute.

The frozen model Adam is retained intact, including pre-freeze moments and step
counts. A separately recorded `frozen_student_start_step` identifies the model's
last update; cursor steps thereafter count committed samples rather than model
Adam steps. Strict checkpoint validation checks that model steps remain at the
anchor while action/selector counts follow their real updates. Historical joint
checkpoints cannot silently become frozen resumes and vice versa. A reviewed
explicit fork records the changed mode/source identity; no model-only reset is
used. Returning to joint requires a new explicit fork, not an implicit toggle.

Synthetic tests verify model weights and Adam unchanged, model gradients cleared,
live geometry/stereo/Gaussian repair gradients, normalization/selector updates,
resumption and malformed anchors. Earlier string-validation edit failed tests
and was corrected before native launch. 135 focused tests then passed; full mypy
passed 228 files and Ruff/format 350 files. Synthetic tensors run on CPU after a
small CUDA bootstrap on GPU0 to accommodate the existing environment's cold
CPU-autograd issue. Pure CPU portability remains uncertified.

## Predeclared native comparison

External run identity: `joint-versus-frozen-gpu0-11fuurbl`.

- Both start from the same geometry203checkpoint:
  `531bb9295fd335b710dd019f297630d938e9bdb0db0c3ff765f27e7a6d04d76d`.
- 256 additional identical seeded FIT entries per branch, cursor 203→459.
- All model/action/selector weights, three Adam states, normalizer and both
  gradient-balance states retained bitwise at the fork.
- Same geometry loss, teacher cache, utility, raw guard thresholds, risk cutoffs,
  candidate/mask policy and four selector updates per known snapshot.
- Context8 is not enabled, because its previous pilot did not improve utility.
- Native evaluation on the same sealed 140 diagnostic IDs, with fresh FIT-only
  cost calibration for each branch. Neither VAL nor finaltest is used for
  optimizer updates, threshold tuning or checkpoint selection in this run.
- GPU0 is used for both branches because GPU1 was occupied by another job.
  Jobs run sequentially in an isolated private tmux socket; no external job is
  terminated or reconfigured. Cache remains read-only; no teacher inference.
- Reviewed source:
  `c48ef5b9c4944651d41b3f84fb3e19cfdd673b1bfba379f6222833b5e60d40d3`.

External plans/launcher identities, branch worklists, fork-validation reports,
checkpoints and native reports preserve full provenance. Native numerical
histories and resulting regions may diverge; this is an experiment on the
student update mechanism, not a claim of identical terminal actor states.

## Required outcome report

Report base and selected depth MAE/RMSE, both PSNR/SSIM views, mean utility,
physical violations and unknown outcomes, action counts, per-action gain MAE/
risk Brier, missed opportunities, actual optimizer exposure, weight drift and
measured runtime/peak allocated memory. Separate frozen-model integrity from
repair/selector quality. Selecting all three actions or low train loss alone
is not acceptance. Results are recorded after both evaluations complete.


## Completed results

Both branches completed 256 additional samples (459 total cursor), exit 0. Both
native assessments completed the same 140 IDs, with weights unchanged during
assessment and no teacher inference. This is one bounded comparison, not a
multi-seed significance test or an independent paper benchmark.

| Selected policy | Joint | Frozen student |
| --- | ---: | ---: |
| STOP / stereo / Gaussian | 4 / 12 / 124 | 13 / 31 / 96 |
| Depth MAE (mm) | 10.43625 | 9.27215 |
| Depth RMSE (mm) | 15.64520 | 15.69684 |
| Left PSNR (dB) | 19.75424 | 19.77136 |
| Right PSNR (dB) | 19.24378 | 19.41124 |
| Mean physical utility | 0.13571 | 0.15678 |
| Guard violations | 26 | 20 |
| Unknown selected metric outcomes | 3 | 4 |
| Safe-gain ranking accuracy | 59.24% | 79.25% |
| Stereo gain prediction MAE | 0.15426 | 0.12411 |
| Stereo risk Brier | 0.19080 | 0.18660 |
| Gaussian risk Brier | 0.15615 | 0.18535 |

Frozen is better on several aggregate outcomes here, but not every metric:
joint has slightly lower depth RMSE and lower Gaussian Brier. Neither achieves
accepted safety:20/26 violations and unknown outcomes remain. The joint oracle
still chooses stereo54/Gaussian68/STOP18, showing useful stereo opportunities
exist. The actual joint selector chooses stereo12; among62stereo-best regions
its ranking is correct in only 8, versus 22/42 for frozen.

### Current failure decomposition

1. **Joint base reconstruction regression:** the frozen base preserves the parent
   state; joint changes its base geometry/right render. Reported selector depth
   errors reflect substantial base errors, not only the tiny repaired footprint.
   Freezing measures the combined effects of base updates, feature drift and
   evolving outcomes; it does not identify which individual loss caused them.
2. **Conditional stereo gain prediction is weak:** joint stereo prediction STD
   is 0.04061 versus actual 0.18186; predicted mean -0.00992 versus actual +0.06118.
   TRAIN stereo mean utility is -0.00998. The near matching marginal means and
   small output spread are consistent with weak conditional learning/transfer,
   not proof that features contain no information. Native oracle opportunities
   remain. Frozen also underrepresents utility spread, so this is not solved.
3. **Gaussian conditional risk/local preservation:** joint selected Gaussian has
  22 harms/121 known selections; mean predicted risk 9.64% versus actual 18.18%.
   Seventeen failures violate local RGB. Frozen has 18/92 Gaussian harms with
   predicted 7.21% versus actual 19.57%, including 14 local RGB failures. Thus student
   freezing alone does not solve Gaussian risk.
4. **Stereo policy also permits substantial risk:** joint selected stereo has
  4harms/12selections; predicted mean risk 33.35% and actual 33.33%. Its configured
   cutoff 0.51 permits those risks. It is incorrect to describe every stereo
   failure as risk underestimation. This cutoff is a historical pilot candidate,
   not an accepted safety budget. No cutoff is retuned on these reported images.
5. **Supervision coverage/transfer:** only 200 joint and 199 frozen primary snapshots
   have complete metric labels; each has 228 known regions. Unknown outcomes are
   excluded, not negative/safe labels. Gaussian requires complete metric truth
   even though its stereo-depth increments are structurally zero, reducing
   supervision coverage. Any partial-label extension needs an explicit contract
   and tests, rather than filling absent GT with fabricated values.

TRAIN stereo harm fractions are 33.33% joint versus 10.09% frozen, with utilities
-0.00998versus+0.00785. Gaussian fractions are 17.11% versus 16.23%. These are
within-branch descriptive outcomes; generated regions/actors differ, so they are
not paired-identical-ROI causal estimates.

### Integrity, exposure and resources

The two actual 256-ID worklists are identical. Normalization state remains exactly
common. Final model/action/selector update counts:

- Joint:459/459/1240.
- Frozen:203/459/1236.

The four-update selector difference follows one more known snapshot in joint;
no quota, oversampling or fresh-selector reset is introduced. Frozen final model
weights **and model Adam** are bitwise equal to the parent. Both repairs and
selector change in each branch. Frozen's anchor remains 203.

Measured wall times including initialization/hashing/fork are 386.84 s joint and
340.90 s frozen. Peak allocated memory is 4,393,279,488 bytes joint and
4,471,474,688 bytes frozen. These are bounded experiment resources, not warmed
matched-FPS claims. Virtual gradients were retained for fairness, so this freeze
implementation is not a memory-saving inference optimization.

Checkpoints:

- Joint459: `ca8c637577f7952fa10fc64c75102b509198f28c29a4a8cf5a140d23a466ea52`.
- Frozen459: `067c5e54abbfc1628bd6d4d481e9d26209b4ec7b9b9de967729428aab87d9604`.

External `completion-report.json` binds the exact source, paired IDs, full native
reports, branch-integrity proof and prior failure audit. Both workers are done.

## Joint-first next steps (not implemented in this pass)

1. Isolate repair-loss feedback into the student from ordinary base supervision.
   An explicit detached repair-to-student gradient ablation can keep all three
   owners learning together while student retains RGB/warp/KD/base-GT losses and
   repairs retain their local/global/geometry objectives. This would test a
   narrower mechanism than permanently freezing the student. Do not assume its
   benefit before matched evidence; retain the current joint branch as control.
2. Investigate Gaussian local-render risk supervision and action-conditioned
   residual/footprint features. Distinguish continuous constraint margins from
   binary harm calibration; keep common utility and raw physical guards owned by
   the existing canonical implementation. Do not invent an action-count target.
3. Validate risk policy/cutoffs on a predeclared FIT/inner-stopping/CAL protocol
   with an explicit safety/utility criterion, then bind calibration to a fixed
   actor checkpoint. The 0.51 stereo cutoff already accepts significant predicted
   harm, so increasing gain or merely waiting longer is not a demonstrated fix.
4. Use checkpoint acceptance jointly on base depth and both render views, then
   conditional selection utility/violations. Independent final evaluation needs
   an ancestry-clean protocol; these repeatedly inspected 140 images are diagnostic.

Joint remains the prioritized/default path. Frozen stays a separate comparison
artifact. No claim that joint training is impossible or that frozen is accepted
for final deployment follows from this single pilot.

## Changes and validation boundary

This pass modifies only `PLAN.md`, this report,
`config/intervention.py`, `training/balanced_learning.py`,
`training/student_learning.py`, `training/student_checkpoint.py`,
`training/student_run.py`; it adds
`configs/routing/intervention_metric_geometry_frozen_student.yaml` and
`tests/training/test_student_freeze_ablation.py`. Earlier unrelated working-tree
changes are preserved. 135 focused tests, mypy 228 files and Ruff 350 files pass.
Full remote CI, long/full-data training, multiple seeds, teacher inference,
student gradient-isolation variant and final test 8–9 were not run. No dependencies
were installed, datasets/manifests edited, successful outputs overwritten,
source committed/pushed or other GPU users' jobs altered.
