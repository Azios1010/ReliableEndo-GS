# Intervention selection experiment

The opt-in v2 remediation is specified in
[Balanced local/global action learning](intervention_balanced_learning.md).
The v1 definitions and evidence below remain unchanged.

## Authority and evidence

The user approved implementing target screening, correspondence-aware stereo
repair, a gain/harm guiding harness, and joint student/head/repair/selector
learning. This is an opt-in experiment, not Gate I/II certification. Historical
frozen routing and the original baseline keep their existing semantics.

The preceding diagnostic run used 5,738 original SCARED frames, 25% sampled
within dataset_1–7, for three joint epochs. Its 112 development frames share
keyframes with training but have distinct IDs. Native evaluation selected
Gaussian on all 112 frames, including the oracle; learned tile placement was
useful, while fixed stereo worsened depth MAE on 68/111 GT-valid frames. This
negative evidence is retained; the new method has not yet demonstrated an
improved geometry or quality/cost frontier.

## Executable changes

- `--intervention-config` selects `intervention_selection.v1`; absent that flag,
  the historical pointwise head, RGB-minus-time utility and allocator remain.
- Matching stereo action uses shared spatial features, a search of ±16 current
  grid pixels at step 2, full-right-image warping and a spatial residual head.
  It starts at exact zero correction. Corrections remain bounded to the lesser
  of 16 px and 50% of positive effective disparity. Core writes and canonical
  A03 sampling/geometry are preserved. Right reads are not limited to a halo;
  both repairs still render full frames. No ROI-proportional cost claim.
- At exactly zero Gaussian residual, the opt-in action anchors field values to
  the actual base field, preserving canonical composition derivatives. This
  avoids FP32 quaternion roundtrip drift without skipping learning gradients
  or relaxing smoke RGB tolerances. Nonzero residual composition is unchanged.
- Cached KD is screened by finite/positive effective depth, right FOV/sensor
  support and the configured renderer-supported depth range. An empty KD mask
  gives zero KD and an explicit diagnostic; the sample continues through RGB,
  correspondence and repair learning. Cache/image files and dataset roles are
  unchanged. Read-only cache misses never invoke a producer.
- Accepted teacher targets are not guaranteed correct. A teacher can be wrong
  while passing FOV/depth checks. Photometric screening is optional and defaults
  to disabled; no threshold is fabricated from GT. No stereo consensus or
  calibrated teacher confidence is claimed.
- All three owners learn jointly. Model-only warm start explicitly reuses the
  previous stereo student/Gaussian head, with new repair/selector weights and
  fresh Adam states. This is a new experiment, not checkpoint resume.

## Objective and guiding harness

All alternatives execute the shared action implementation from the same
pre-action state and parameter snapshot. Targets/features are detached before
selector learning; every optimizer update occurs after intervention collection.

The versioned normalized gain is:

```text
G = (RGB_before - RGB_after) / rgb_scale
  + stereo_weight * (warp_before - warp_after) / stereo_scale
  - collateral_weight * max(0, outside_after - outside_before) / rgb_scale
```

RGB uses the canonical two-view Charbonnier objective. Stereo consistency is
left/right RGB warping, on fixed pre-action support; invalid after-action
correspondences incur a penalty, never disappear from the denominator. It is a
proxy, not GT depth error. Collateral error uses left image core and its fixed
pre-action projection into the right view. It does not use post-action masks.

Harm labels record degradation of a declared component beyond a pilot
tolerance. The small 26 → 32 → 32 → 4 MLP predicts two gains and two harm
scores; STOP gain is exactly zero. SmoothL1 gain regression, informative
margin-weighted pair ranking and harm BCE supervise the selector. No action
quota, synthetic oracle winner balancing, or post-action evidence enters input.

Student/head loss normalizes RGB, warp residual and screened KD; both repairs
receive reconstruction plus correspondence supervision. The training config
defaults to zero warmup and three joint epochs, preserving user-requested joint
updates. The smoke config deliberately exercises one student and one joint
update to check both execution paths.

The new experiment uses one fixed seeded permutation of the unchanged training
manifest, recorded by an ordering hash and used by exact resume cursors. A bounded
pilot therefore spans the pool rather than starting with consecutive frames of
one keyframe. It is still a short learning diagnostic, not a completed epoch or
evidence of generalization. Legacy training keeps its existing order.

The shipped scales, search bounds, weights, harm cutoff and budget are explicit
pilot settings. They are not fitted calibration or proven scientific choices.
Use training/pilot evidence to freeze them before an independent final test.
GT remains outside deployment inputs and is used for offline evaluation/QC.

## Deployment and budget

For one intervention, select the largest predicted positive gain among eligible
actions whose predicted harm passes the configured cutoff and estimated cost
fits the remaining total budget. Cost is a constraint, not both a utility
penalty and divisor. STOP applies if there is no feasible beneficial candidate.

Total budget accounts for the already executed base and routing work. Measured
action-cost maxima are estimates; actual action time and overruns are reported.
A budget smaller than base cost cannot be guaranteed by choosing STOP. Harm
scores are uncalibrated until validation proves their calibration; passing a
proxy check is not proof that metric geometry is protected.

## First-party commands

Run from the repository, using the existing runtime Python and external roots:

```bash
PYTHONPATH=src python -m reliable_endo_gs.cli.selection quality \
  --data-root "$RELIABLE_ENDO_DATA_ROOT" --manifest "$TRAIN_MANIFEST" \
  --cache-root "$KD_CACHE" --config configs/routing/intervention_selection.yaml \
  --output-dir "$NEW_OUTPUT/quality" --max-samples 64

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m reliable_endo_gs.cli.student train \
  --data-root "$RELIABLE_ENDO_DATA_ROOT" --manifest "$TRAIN_MANIFEST" \
  --cache-root "$KD_CACHE" --config configs/training/intervention_joint.yaml \
  --intervention-config configs/routing/intervention_selection.yaml \
  --initialize-model-from "$PREVIOUS_STUDENT_CHECKPOINT" \
  --max-steps 256 --output-dir "$NEW_OUTPUT/train"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m reliable_endo_gs.cli.selection evaluate \
  --data-root "$RELIABLE_ENDO_DATA_ROOT" --manifest "$DEVELOPMENT_MANIFEST" \
  --training-manifest "$TRAIN_MANIFEST" --checkpoint "$NEW_OUTPUT/train/stopped.pt" \
  --output-dir "$NEW_OUTPUT/evaluation" --max-samples 112
```

Use a strict `--checkpoint` resume in a new output directory for continuation;
do not combine it with `--initialize-model-from`. A stopped bounded full-manifest
run can be evaluated diagnostically; a smoke-only checkpoint cannot be deployed.
No run output or cache belongs in this repository. Final test remains dataset_8–9.

## Validation and acceptance

Synthetic tests check ROI warp direction/units, principal-point handling,
projection of right-view support, finite gradients, zero-action identity, KD
screening/empty-mask behavior, core correspondence, gain/cost/risk/STOP selection,
ranking target detachment, checkpoint protocol rejection and evaluation
immutability. Native preflight/smoke and training require a GPU-enabled shell.

Native evaluation compares base, fixed actions, fixed action with predicted
region, learned selector and a gain/harm/budget-consistent oracle. It reports
actual action distributions, gain/regret, proxy harm, budget violations, RGB and
independent GT geometry. GT never generates training selector features/labels.
Sampled cloud metrics are diagnostics; no LPIPS or paper-parity claim.

Proceed to claims about multi-action selection only if oracle evidence has
repeatable useful support for both repairs, and the selector beats strong
region/fixed-action baselines at matched measured compute. An oracle dominated
by one action remains a negative result. Increasing model capacity or changing
weights to manufacture a gate pass is not acceptance evidence.
