# Joint geometry and selector remediation audit

## Scope and verdict

This is the user-approved isolated joint-learning experiment: student/Gaussian
head, stereo/Gaussian repair and action-specific selector gain/risk heads remain
trainable. It does not replace the historical freeze/gate program or certify
Gate I/II. Baseline and historical artifacts are preserved. Original SCARED
TRAIN folds 1–7 and final-test folds 8–9 are unchanged.

The geometry objective gap is repaired and has promising bounded native evidence.
Raw joint selector quality is **not yet accepted**. Right-view render regression,
remaining physical guard violations and evaluation ancestry overlap prevent a
benchmark/generalization claim. Selecting all three actions alone is insufficient.

## Verified findings

1. **Missing training objective, high:** prior metric joint code used TRAIN GT
   only for detached counterfactual gain/risk labels. The student and stereo
   repair did not directly optimize metric depth. RGB/warp/teacher objectives
   could improve while depth and actual stereo intervention quality regressed.
2. **Evaluation ancestry overlap, high:** current FIT/VAL/CAL manifests are
   disjoint, but an older backbone checkpoint trained on all 140 diagnostic VAL
   images and all 1,378 CAL images. `train_overlap: 0` checked only the current
   training manifest. These results are development diagnostics, not independent
   held-out performance. Final-test folds 8–9 were not used here.
3. **Joint selector underfits current outcomes, medium:** raw joint stereo gain
   predictions have much less spread than actual utility and underestimate harm.
   Stationary fresh-outcome refitting improves ranking and stereo selection.
   This demonstrates available useful action opportunities; it does not isolate
   a single causal failure of joint optimization or prove deployment safety.
4. **Unresolved data semantics, medium:** a read-only 14-FIT-sample audit verified
   canonical calibration, image/GT resampling and shared validity indices. Some
   GT maxima reach metres while most valid points are much closer. Authoritative
   sentinel/outlier semantics are not established, so no samples/pixels were
   silently filtered or relabelled. Screened teacher support is not accuracy proof.

## Implemented changes

### Explicit metric geometry learning

Canonical `training/metric_objective.py::metric_depth_objective` calls
`geometry.rectified.reconstruct_rectified` for both prediction and detached GT.
The camera convention, principal-point correction and baseline units are
unchanged. Depth is converted from metres to millimetres once.

```text
base term = mean SmoothL1((Z_pred - Z_GT) * 1000 / geometry_loss_scale_mm)
local term = mean of observed stereo-repair terms on pre-action region masks
geometry = base term + stereo_repair_geometry_weight * local term
```

The denominator is fixed by GT validity, sensor support and optional pre-action
region support. Invalid predictions cannot remove difficult pixels; they fail on
fixed valid support. Empty GT support produces differentiable zero without
fabricating observed outcomes. GT is never placed in `DeploymentStereoBatch`,
features, selector inputs or inference.

`geometry_supervision_weight > 0` adds a fifth `geometry` objective to the model
shared-gradient balancer. Its newly introduced EMA is measured before the first
weighted update. Base and local stereo gradients reach their actual owners;
all three optimizers continue learning. Existing RGB/warp/KD/repair objectives
are retained. RGB reconstruction already supervises both left and right views;
the measured right-view regression is not explained by absent right RGB loss.
Student-only warmup also requests explicit offline TRAIN truth
when geometry learning is enabled.

Configuration:
`configs/routing/intervention_metric_geometry_supervision.yaml` opts into
weight 1 and scale 1 mm. Historical configurations retain their hashes and
four-term objective. A previous checkpoint requires an explicit reviewed fork;
its old optimizer/normalizer/balance prefix must not be silently reset.

### Fresh multi-frame joint selector supervision

`selector_context_frames` defaults to 1 (historical behavior). The experimental
`configs/routing/intervention_metric_geometry_context.yaml` sets it to 8.

For each current TRAIN sample, the runner resolves up to seven preceding entries
in the same fixed seeded FIT order, within the current epoch. It reloads their
separate offline truth and **recomputes** detached features and real STOP/stereo/
Gaussian interventions using the current model/action parameters, before any
optimizer update. It never reuses outcomes from older actor weights.

Only selector features/labels are batched across frames. The current sample
still supplies student/KD/repair/geometry gradients and one model/action update.
The existing selector update count, physical utility, risk guards, thresholds,
normalization identity and action heads are unchanged. Normalization burn-in
continues observing only the primary sample. No action quota is introduced.
Context collection temporarily disables action gradients and restores flags
on success or failure; this is detached measurement, not an actor freeze phase.
Resume reconstructs context from the checkpoint cursor and fixed sample order.
Context receives no teacher inference and creates no duplicate medical files.

Unknown metric regions remain excluded rather than labelled safe. Known context
regions may train the selector when the primary sample has unknown GT support.
Forbidden roles, GT-containing deployment samples, duplicate identities and
mismatched truth are rejected. Epoch starts naturally have fewer context frames.
This costs extra counterfactual computation; no CUDA/FPS improvement is claimed.

### Evaluation reporting

Evaluation now explicitly records:

```text
training_overlap_scope = current_training_manifest_only
initialization_overlap_audited = false
evaluation_scope = development_diagnostic
```

The evaluator does not pretend to audit checkpoint ancestry itself. The separate
ancestry audit links pinned parent hashes and exact sample identities. Original
metrics/logs are retained rather than edited.

## Matched native geometry experiment

Both branches start from the same 128-sample checkpoint, retain model/action/
selector weights, all three Adam states, cursor/RNG and normalizer/selector
balance, then consume the same next 128 FIT entries. The new branch appends one
geometry balance channel. Utility, raw guards and risk cutoffs are unchanged.
Both are evaluated on the same 140 diagnostic images (20 per fold) with native
renderer and freshly measured FIT-only action cost artifacts. No teacher
inference occurs; KD cache is read-only. Native nondeterminism is not claimed
away. Generated TRAIN regions differ, so their outcome means are descriptive.

| Selector result | Historical objective, 256 samples | Metric geometry, 256 samples |
| --- | ---: | ---: |
| Depth MAE (mm) | 14.34915 | 9.61688 |
| Depth RMSE (mm) | 44.44206 | 15.02915 |
| Left PSNR (dB) | 19.70393 | 19.73109 |
| Right PSNR (dB) | 19.53235 | 19.43417 |
| STOP / stereo / Gaussian | 2 / 0 / 138 | 4 / 0 / 136 |
| Mean physical utility | 0.15821 | 0.16985 |
| Physical guard violations | 22 | 22 |

Fixed stereo mean utility improves from -0.06327 to +0.02010; violations fall
from 90 to 49/140. The geometry model's oracle chooses STOP22/stereo35/Gaussian83.
Raw selector still chooses no stereo. Geometry improvement does **not** establish
balanced rendering quality: right PSNR falls about 0.098 dB, and 22 guard
violations remain. Do not promote this checkpoint on MAE alone.

One original geometry evaluation ended after 18/140 images despite shell exit 0;
no cause is established. Its partial logs are preserved. A fresh independent
output directory completed 140/140 with unchanged weights. Completeness requires
report/sample-count checks, not merely exit status.

## Fresh selector diagnosis (fixed geometry actor)

Worklists were sealed before new outcomes: 160 FIT, 42 inner-stopping FIT and
84 CAL images, with disjoint selected keyframes. VAL is used only for reporting.
The actor is fixed only for this diagnostic; it is not the requested joint path.
Head fitting selects one whole snapshot at 50 steps under action-objective,
risk-BCE and ranking validation guards. Utility/cutoffs remain unchanged.

| Native 140-image diagnostic | Raw joint | Risk CAL only | Fresh FIT refit + CAL |
| --- | ---: | ---: | ---: |
| STOP / stereo / Gaussian | 4 / 0 / 136 | 0 / 8 / 132 | 7 / 20 / 113 |
| Mean physical utility | 0.16985 | 0.17055 | 0.18950 |
| Guard violations | 22 | 25 | 19 |

Calibration alone is insufficient. Fresh refit raises safe-gain ranking accuracy
from 68.35% to 77.85%; correct stereo-best rankings rise from 1/45 to 22/45.
Stereo gain prediction STD rises from 0.01393 to 0.09405 (actual 0.18550), and
stereo risk Brier improves from 0.33030 to 0.21549. Right-view PSNR remains below
the historical-objective control. This is useful diagnosis, not final acceptance.
The fitted artifact is bound to this exact model/action checkpoint and must be
invalidated after any actor update. It is not silently attached to joint training.

## Evidence identities and location

All outputs remain outside the repo under run identities:

- `joint-larger-gpu1-4vgk9wi_`: completed 1,024-step continuation and old regression.
- `joint-geometry-remediation-qayuedy8`: matched objectives, data/ancestry audit,
  fixed-actor selector refit, original failures and independent retry evidence.
- `joint-fresh-selector-gpu1-juby_sgn`: predeclared 75-step fresh-context pilot
  and paired 203-sample native assessment; completion reported separately below.

Pinned SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| Initial 128-sample checkpoint | `22f71cb96c2b744524a223bcc5406099ceee60081c660a58c4edc31080f2a425` |
| Historical-objective 256 checkpoint | `6023b1209a623ec633e4635a9d170693183b801781734923a7bc074e92081b4b` |
| Geometry 256 checkpoint | `2f24f2406b74a9d48f78f8cd1765bf789ca5f7220d3786bc53c0ab737bcc3fc0` |
| Older trained backbone ancestor | `2280fc676f2d1e9441cd9709b1f3949bf871b12c6271cd55e0c4cc85f14bfcfa` |
| Metric objective source | `26b72db87f51399c65b81eaab18ed150946c9fbe4a3acf6791ef76fa65a57bfb` |
| Source with evaluation scope metadata | `e089da8a3332294f5c17b60b46949fdfbcf18c34ecf75fc83226e6e28dcbc72b` |
| Fresh-context pilot source | `b109f978fadb5b35bfd31b021d0751dec82c7d36b2dfa1276f89a82c18394a70` |

External plans bind full input/config/source/checkpoint hashes, exact FIT/VAL/CAL
IDs, hardware and fork validation. Check `fork-validation.json`, completion
reports, checkpoint cursor and full image counts together. Successful artifacts
are not overwritten. Generic strict CLI resume of these custom experiment-fork
identities has not been certified; the bounded launcher uses an explicit reviewed
fork with first-party checkpoint APIs, rather than silent model-only reload.

## Validation

After fresh-context implementation: 131 focused tests passed in 33.04 s;
`mypy src` passed for 228 files; Ruff checks and formatting passed for 349 files.
Tests cover metric units/principal-point offset, fixed denominators, gradients,
empty GT, separate owners, historical hashes, actual context collection, failure
flag restoration, forbidden inputs, known-context/unknown-primary behavior and
checkpoint-resume context identities. Earlier new-test failures were fixture
errors (invalid normalization mode and impossible pre-fitted burn-in counters);
they were corrected without relaxing runtime contracts.

Synthetic tests use CPU tensors after initializing the existing CUDA runtime
on GPU1 because cold CPU autograd previously aborts in this environment. Pure
CPU portability is not certified. Full repository pytest/CI, long training,
final-test 8–9 evaluation, paper benchmark, teacher training and performance
optimization were not run in this audit.


## Completed fresh-context joint pilot

Both branches retain the same 128-sample parent and consume the same next 75 FIT
entries. The geometry-only checkpoint at cursor 203 is preserved from the earlier
run. The context branch reaches cursor 203 in 525.52 s on GPU1, peak allocated
3,787,266,048 bytes (3.53 GiB). Both native assessments finish all 140 **identical
sample IDs** with unchanged evaluated weights, exit 0 and no teacher inference.
`samples_sha256` hashes each full result file, so it naturally differs; the
separate ID-list hash verifies pairing rather than confusing metrics bytes with
sample identity.

| Native diagnostic at 203 samples | Geometry only | Geometry + fresh context8 |
| --- | ---: | ---: |
| STOP / stereo / Gaussian | 11 / 38 / 91 | 13 / 36 / 91 |
| Mean physical utility | 0.14226 | 0.14069 |
| Guard violations | 20 | 20 |
| Selected unknown-metric interventions | 4 | 5 |
| Depth MAE (mm) | 9.27209 | 9.28332 |
| Depth RMSE (mm) | 15.69668 | 15.71156 |
| Left PSNR (dB) | 19.76945 | 19.75227 |
| Right PSNR (dB) | 19.40998 | 19.46358 |
| Safe-gain ranking accuracy | 76.73% | 79.25% |
| Stereo risk Brier | 0.18746 | 0.19808 |
| Gaussian risk Brier | 0.17342 | 0.16714 |

**Negative result:** context8 has no demonstrated utility/safety improvement.
Do not promote it to a default or a long-run fix on the basis of selecting all
three actions. Geometry-only itself selects stereo38 at cursor203 but stereo0
at cursor256; this is changing joint-learning behavior, not proof of an incapable
selector or absent opportunities. The current experiment does not identify one
causal mechanism for that temporal instability.

Normalization state is exactly unchanged. Both model/action optimizers have 203
updates; selector counts are 440 versus 512. Known extra context permits 72 updates
where primary GT labels were unavailable. Thus this trial changes batch exposure
and eligible update count, not only batch width. Model and action relative L2
differences are 0.9947% and1.6814%; evaluation is not an identical-actor selector
ablation. No claim of isolated causality or statistical significance is made.

Pinned checkpoint hashes:

- Geometry-only203: `531bb9295fd335b710dd019f297630d938e9bdb0db0c3ff765f27e7a6d04d76d`.
- Fresh-context203: `c59fc069b7ff65580ce6e085c5cb3c3b7be7d1c39f38d5e33a8a2dd67ce10d58`.

The external `completion-report.json` binds the paired ID-list, evidence hashes,
full native reports, optimizer exposure and measured resources. No worker remains
training after the bounded launcher completes.

## Remaining work before an accepted large joint run

- Establish reliable risk prediction/calibration with actor changes accounted
  for; raw stereo risk remains underestimated. Frozen-actor calibration alone
  did not solve safety in the 256-sample diagnosis.
- Guard checkpoint acceptance jointly on metric geometry and both render views,
  rather than terminal sample count, loss, MAE alone or three-action presence.
- Evaluate convergence and heterogeneous intervention utility on larger,
  explicitly sealed TRAIN/inner-stopping cohorts; do not tune on the 140 reported
  images. Repeated inspection has made this cohort diagnostic.
- Audit initialization ancestry for a genuinely disjoint evaluation protocol.
  Do not reuse these development scores as paper benchmark evidence.
- Resolve GT extreme-value/confidence semantics from authoritative provenance
  before considering any data filtering. Existing data and manifests remain intact.
- Certify generic strict CLI resume for custom fork provenance and pure CPU
  portability separately if those modes are required.

These are remaining research/validation tasks, not implemented improvements.

## Files changed by this audit/remediation

The checkout already contained unrelated changes. This pass adds or changes only:

- `PLAN.md`
- `docs/joint_geometry_selector_audit_2026-10-13.md`
- `configs/routing/intervention_metric_geometry_supervision.yaml`
- `configs/routing/intervention_metric_geometry_context.yaml`
- `src/reliable_endo_gs/config/intervention.py`
- `src/reliable_endo_gs/training/metric_objective.py`
- `src/reliable_endo_gs/training/balanced_learning.py`
- `src/reliable_endo_gs/training/student_learning.py`
- `src/reliable_endo_gs/training/student_run.py`
- `src/reliable_endo_gs/evaluation/intervention_selection.py`
- `tests/training/test_metric_objective.py`
- `tests/training/test_selector_context.py`
- `tests/evaluation/test_mask_validation.py`

No source commit/push, dependency installation, raw/processed medical data edit,
manifest/split change or final-test evaluation was performed. New geometry/context
settings are opt-in, not changes to historical experiment defaults. The inherited
config header's fresh-selector guidance applies to migration from the old shared
architecture; this audit's same-head-topology loss fork instead explicitly retains
weights/optimizers and binds the changed objective identity. Generic implicit
resume across objective identities remains rejected.
