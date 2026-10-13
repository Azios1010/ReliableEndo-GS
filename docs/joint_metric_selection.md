# Joint selection with physical metric outcomes

This is an explicitly requested experimental joint student/actions/selector
path. It does not certify Gate II or promote the final router. Historical proxy
configurations, checkpoints and negative results remain available.

## What changed

The original `configs/routing/intervention_metric.yaml` opts into physical outcome labels
instead of RGB/warp decision labels. Student and both repairs continue learning;
the selector starts after stationary TRAIN normalization burn-in. TRAIN GT is
loaded through a separate `OfflineMetricTruth` provider, with matching sample ID,
role and output-grid checks. Deployment batches, features, masks and decisions
remain GT-free. GT tensor identities are hashed into training logs.

The shared canonical bounded utility uses global/local depth MAE gains (mm) and
left/right PSNR (dB) and SSIM gains. Geometry priority is 3 relative to render
priority 1; the geometry group splits equally between global and local MAE.
Both actions use the same function and scales. Stereo receives no constant
bonus. Declared numerical scales are 1 mm, 0.1 dB and 0.001 SSIM. They are not
estimated from development outcomes or claimed as clinical significance limits.

RMSE, local RMSE, p95, coverage, both render views, outside RGB and local RGB
are physical harm guards. Increasing a reward never overrides a guard. Gaussian
uses half the stereo tolerances: PSNR drop 0.005 versus 0.01 dB; SSIM drop
0.00025 versus 0.0005; outside RGB increase 5e-6 versus 1e-5; local RGB increase
5e-5 versus 1e-4. Geometry/coverage regression tolerances remain zero. The
canonical numerical epsilon remains 1e-6 in each metric's raw units.

Proposed risk cutoffs are 0.20 for stereo and 0.05 for Gaussian. These are
uncalibrated joint-model predictions, not probabilities with a validated safety
guarantee. They require held-out verification. Unknown metric outcomes are
excluded from selector losses, normalization and risk calibration statistics;
they are never treated as observed safe labels. Student/repair updates may
continue on those samples using admissible RGB/warp/KD losses.

Stereo model training priority is 3 after gradient balancing; repair warp
weight is 3. Selector local auxiliary weight is 0.25. These are declared pilot
choices, not an empirically proven optimum. Logs retain the effective gradients
so the local/global influence can be rechecked.

## Costs and comparisons

After joint updates, the bounded pilot measures shared action+render calls on
its TRAIN worklist, with one warmup and three measurements per eligible mask.
The declared pooled tail is the 95th percentile. All timings are retained in
`deployment_action_costs.pt`, bound to model/action/config, training identity,
runtime, grid and target-mask index. Training maximum timings remain diagnostic.
The evaluator uses `estimated_action_costs(state)` and the same calibrated
costs for selector, oracle and fixed-action budget checks. Actual budget overruns
are still reported: a quantile estimate does not guarantee every call fits.

Physical-mode evaluation requires dedicated TRAIN calibration. Updating model
or actions invalidates it. Development timing cannot fit this artifact. A
separate external artifact may also be passed to selection evaluation using
`--cost-calibration`. Runtime launch settings are part of the binding.

Development GT is read after the predicted decision. Its oracle then uses the
same physical utility and per-action guards as joint supervision, with unknown
outcomes masked. This oracle is an offline diagnostic, not a deployment input.

## Run boundary and checkpoint compatibility

The mask-validation CLI defaults to the new physical configuration and
`metric_tradeoff_joint_balanced.yaml`, paired with
`intervention_metric_balanced.yaml`. The original geometry-priority configuration
described above remains an explicitly selectable reference. Host wrappers must use both
matching configs. CPU preflight checks assets; a native run performs initial
STOP evaluation, bounded joint learning, TRAIN cost collection and development
comparisons. The existing worklist and role manifests are not replaced.

Changing the target, scales or guards changes the config identity. An old proxy
selector cannot be resumed under the new labels. Start a new run using the
existing model-only initialization path; preserve prior run outputs. The
bounded pilot is 56 TRAIN frames and 28 development frames on the existing
worklist, not a new split or full-cohort experiment.

Readiness requires synthetic tests and then a new bounded native check. Success
means finite updates and consistent targets/cost bindings; contribution evidence
additionally requires a reproducible advantage over fixed Gaussian, acceptable
depth/render tradeoffs, risk calibration, heterogeneous oracle utility and
matched runtime comparisons. No action diversity is forced.

## Balanced joint policy (2026-10-09)

The user requested a balanced revision after the 168-update pilot selected STOP
on 27/28 development samples. Geometry/render now each have total utility weight
one, with equal global/local geometry shares and equal left/right render shares.
The existing numerical scales and canonical bounded utility are retained.
Stereo optimization priority and warp weight remain three: optimization pressure
does not add a reward or permit violating a physical guard.

The new config pair declares the following *proposed engineering tolerances*.
They are neither empirically estimated noise floors nor clinical acceptance
limits. They must be checked on a new development pilot before promotion.

| Maximum tolerated regression | Stereo | Gaussian |
| --- | ---: | ---: |
| Global depth MAE / RMSE | 0.05 mm | 0.025 mm |
| Local depth MAE / RMSE | 0.25 mm | 0.125 mm |
| Global depth p95 | 0.10 mm | 0.05 mm |
| Either-view PSNR | 0.05 dB | 0.025 dB |
| Either-view SSIM | 0.002 | 0.001 |
| Outside RGB mean error | 0.0001 | 0.00005 |
| Local RGB mean error | 0.001 | 0.0005 |
| Geometry coverage | No drop | No drop |

Both action risk cutoffs are 0.20. Gaussian's stricter physical tolerances still
produce different labels; this removes its additional 5% cutoff rather than
forcing Gaussian selection. These heads are uncalibrated predictions: a cutoff
of 0.20 does not establish a 20% empirical violation guarantee. Positive utility,
known risk evidence, deployment validity and the calibrated compute budget still
govern selection. Severe regressions cannot be offset by higher geometry utility.

Changing the physical guards changes harm/ranking labels; changing group weights
changes gain labels. The old risk/gain head cannot be interpreted using the new
policy, nor can its old cost artifact certify the new run. Start model-only from
the same parent initialization on the same 56 TRAIN / 28 development worklist,
run three joint epochs (168 updates), then collect TRAIN costs and evaluate.
Preserve the old checkpoint/results. Report raw metric deltas and violations
under both old and new guards; improvements in new feasibility alone are not
evidence that the actual repairs improved.

The original 28-sample development log may be replayed to diagnose cutoff
sensitivity without another GPU forward pass. This is post-hoc development
analysis; its old predictions are not calibrated to the revised harm event.
It cannot prove that the revised selector learns or generalizes.

## Geometry gain candidate (2026-10-09)

The user requested a stronger measured geometry gain after the balanced run.
`intervention_metric_geometry.yaml` and `metric_tradeoff_joint_geometry.yaml`
raise geometry group utility from one to 1.5 while retaining render weight one,
the same scales and the same physical harm guards. Positive and negative geometry
changes both receive this factor; zero outcome gain stays zero. Both action types
use the canonical formula. No action-name bonus or post-hoc multiplication of
the old head's predictions is used. Labels/normalization require a fresh head.

The candidate also uses stereo/Gaussian cutoffs 0.51/0.22. These came from a
99-pair *post-hoc* sweep on the preceding 28-frame development log, maximizing
measured utility under the same observed violation/unknown ceilings, rather than
requiring action diversity. With the preceding 1:1 head that candidate chose
5 STOP, 1 stereo and 22 Gaussian, with 2 physical violations and 1 unknown outcome.
It does not establish calibrated risk or generalization, nor predict counts for
the new 1.5:1 head. The raw guard event has not changed, but the head's scores can
still shift during joint learning. Report calibration and raw metrics again.

Use the same parent model-only initialization, worklist, seed, three epochs and
TRAIN cost collection as the balanced reference. Preserve both references.
This is a candidate combining a utility-weight change and inference-cutoff change;
attribute results to that combination, not solely to raising geometry gain.

## Stereo-only gain correction (2026-10-09)

The user clarified that the requested gain increase applies only to stereo.
The preceding geometry-priority pilot did not implement that action-specific
preference and remains archived evidence. The corrected candidate is
`intervention_metric_stereo_gain.yaml`, paired with
`metric_tradeoff_joint_balanced.yaml`. Its common geometry/render utility returns
to 1:1. The canonical `metric_targets` applies
`stereo_utility_multiplier: 1.5` to the signed stereo global and local labels;
Gaussian labels match the 1:1 balanced reference exactly. Negative stereo gains
are scaled too, zero gain remains zero, and harm guards/events are unchanged.

This is an explicit action preference authorized by the user, not evidence of
1.5x better physical reconstruction. Report raw depth/render deltas separately.
Training and the physical evaluation oracle both call the same target owner;
the deployed head learns the scaled labels. Do not multiply its output again.
The default factor one is omitted from historical serialization/hashes. A
nonneutral factor requires physical v2 supervision and a new config identity.
Initialize a fresh selector from the same parent model-only checkpoint.

Risk cutoffs remain 0.51/0.22 from the preceding candidate. Raising utility does
not bypass those filters; all stereo candidates can still be rejected when
their predicted harm exceeds 0.51. No three-action result is guaranteed.

## Recovery after completed training

Physical GT can be unavailable in a selected region. Such a joint update trains
model/actions but neither observes selector normalization nor updates the selector.
Checkpoint validation permits this difference for the physical protocol while
checking every Adam state's step against the recorded owner count. Historical
proxy burn-in still requires its exact sample-minus-normalization relation.

Read-only selection evaluation can explicitly import a historical artifact using
`--checkpoint-sha256`. This pins the exact checkpoint bytes and records both the
checkpoint's source identity and the current evaluation source. It is a new
evaluation, not source-compatible resume or deployment certification. Config,
renderer, manifests/roles, normalization, optimizer states and cost binding remain
validated. The ordinary source check remains strict without this explicit option.
