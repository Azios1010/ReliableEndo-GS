# Mask tokens and local/global risk in the guiding harness

## Scope

The user approved masked student tokens with global context, explicit local/
global risk supervision, joint training as the primary direction and a separate
frozen-student comparison. This opt-in experimental path does not certify the
historical research gates. It preserves original SCARED train1–7/final-test8–9,
existing physical utility, raw guard tolerances and intervention budgets.

## Architecture

```text
Existing student matching pass -> detached stride-4 C-channel features
    -> sample only actual mask write pixels -> local mean + per-channel max
    -> pool valid native sensor support     -> global mean

Existing 61 pre-action statistics + these 3*C channels
    -> independent stereo gain MLP   -> decision utility + local auxiliary
    -> independent stereo risk MLP   -> local/global/any physical risk
    -> independent Gaussian gain MLP -> decision utility + local auxiliary
    -> independent Gaussian risk MLP -> local/global/any physical risk

STOP -> utility=0, intervention risk=0; no trainable STOP parameters
```

The MLP input layers learn fusion of masked feature channels and global context.
The student features already learn from the student's own losses; selector
inputs and labels are detached to preserve parameter ownership. Joint still
updates student/Gaussian head, repairs and selector. Frozen changes only student
optimizer ownership, preserving the previously implemented virtual derivatives.
Repair feedback into the joint student is unchanged by this experiment.

At C=32 the layout is 157 dimensions. There is no new backbone pass. Token
sampling interpolates existing native features only at mask write pixels,
bounded by the proposal's existing max_pixels/max_regions. Mask holes are not
pooled as target pixels. Native centers of the two k3/p1/s2 convolutions are at
(4*y,4*x); pixel-center sampling and border extension are explicit in `tokens.py`.
The backbone receptive field can span surrounding context. Full-frame rendering
and action reconstruction costs are unchanged; no FPS or memory-saving claim is
made without measurement. Gaussian-only feature ablations zero student tokens.

## The harness is a learning guide

`BalancedHarness.collect` executes the registered real STOP, stereo and Gaussian
actions on a common pre-action state. It measures the same canonical physical
components used by evaluation. It is the source of supervised action outcomes,
not just logging or a benchmark wrapper. Actions are not duplicated in a private
oracle. Proposals/features are formed before offline truth is passed to it.

Local events cover local depth MAE, local depth RMSE and local RGB. Global events
cover the remaining depth metrics, both PSNR/SSIM views, outside RGB and coverage.
These groups are disjoint and exhaustive. `grouped_harm` calls the existing
per-component raw guard and `metric_risk_targets` applies the actual action's
tolerances, including Gaussian tightening. The third label is exactly
`harm_any = harm_local OR harm_global`.

GT is separate offline supervision, never a deployment feature or mask source.
Incomplete physical observations have null event labels and are excluded from
selector learning and risk scoring. STOP has exact false labels. A lack of
intervention harm does not establish that base reconstruction is good.

The risk objective is an equal mean of three dimensionless BCEs per action,
with the existing total harm coefficient. It does not multiply risk by an mm
scale or average local/global probabilities. Safe-gain ranking continues to use
the actual union harm label. The local gain auxiliary remains local geometry;
the new local RGB safety supervision belongs to risk.

For predicted local/global probabilities pL/pG, the network learns a third
dependence scalar q and forms:

```text
lower = max(pL,pG)
upper = min(1,pL+pG)
pAny  = lower + (upper-lower)*sigmoid(q)
```

These Frechet bounds avoid assuming local/global event independence. Stable
union BCE uses a logit with floating-point epsilon clipping; saturated
probabilities satisfy the bounds to that numerical precision. Deployment retains
the existing per-action cutoff on pAny, so a high local risk cannot be cancelled
by low global risk. These are learned estimates, not physical safety guarantees.

## Controlled configurations

| Configuration | Features | Risk | Student |
| --- | --- | --- | --- |
| `intervention_metric_geometry_supervision.yaml` | 61 statistics | historical union | joint |
| `intervention_metric_grouped_risk.yaml` | 61 statistics | local/global/any | joint |
| `intervention_metric_mask_tokens.yaml` | 61 + 3*C tokens | local/global/any | joint |
| `intervention_metric_mask_tokens_frozen_student.yaml` | same tokens | same events | frozen |

The first comparison isolates the risk design; the second adds feature
information. Joint/frozen configurations differ only in student update mode.
`configs/training/student_joint_routing.yaml` supplies common width32, seed,
zero student-only warmup and one joint epoch; bound diagnostic work with
`--max-steps`. There is no changed split or action quota.

Historical config serialization/hashes and default network layouts stay intact.
New feature/risk modes are explicit config identities; strict checkpoint loading
rejects topology mismatches before mutation. `--initialize-model-from` starts a
**new** experiment with model weights only and fresh repair/selector/Adam states.
It is not continuation of all historical optimizer states. Both comparison arms
must use the same explicit initialization/reset policy, FIT worklist and seed.
Preserve previous checkpoints and output directories.

## Harness and evaluation connections

- Online training and fresh context snapshots share the canonical grouped
  labels. All three event BCE/Brier/rates are logged per action; raw predictions
  use `[regions, STOP/ST/GS, local/global/any]` order.
- Frozen outcome collection, immutable shard tables and selector snapshot
  readers preserve the dynamic feature schema and canonical event labels.
  Historical tables without event columns remain readable; new contradictory
  labels and labels attached to unknown outcomes are rejected.
- Stationary head fitting can consume these tensors. Native intervention
  assessment reports per-event candidate/selected Brier, means and harm rates,
  with unknowns excluded. Existing depth/render/utility/guard/cost reports remain.
- The subsequent [joint/frozen remediation](joint_frozen_remediation_validation.md)
  implements event-aware CAL fitting and records support for local/global/any
  separately. Historical scalar calibration artifacts remain incompatible with
  the event model. The separate historical 61-feature OutcomePredictor cannot be
  attached to a token selector. Neither path silently substitutes an old head.

## Next bounded command template — not executed in this pass

Use externally resolved processed A00 data, an immutable FIT manifest, a complete
deployment ROI index and a matching read-only KD cache. Set the listed variables
to verified external assets; do not point `A00_ROOT` at the original raw archives.
Set `GPU_UUID` to one free physical GPU UUID. The wrapper below applies this
environment's previously diagnosed CUDA bootstrap workaround, exposing that GPU
as logical cuda:0; no source module performs import-time GPU initialization.

```bash
env CUDA_VISIBLE_DEVICES= RELIABLE_ENDO_GPU_UUID="$GPU_UUID" \
  PYTHONPATH="$REPO/src" RELIABLE_ENDO_TEACHER_CONFIG="$TEACHER_CONFIG" \
  "$PYTHON" -c 'import os,sys,torch; os.environ["CUDA_VISIBLE_DEVICES"]=os.environ["RELIABLE_ENDO_GPU_UUID"]; torch.cuda.init(); from reliable_endo_gs.cli.student import main; raise SystemExit(main(sys.argv[1:]))' train \
  --config "$REPO/configs/training/student_joint_routing.yaml" \
  --construction-config "$REPO/configs/reconstruction/a04_reference.yaml" \
  --routing-config "$REPO/configs/routing/frozen_a04.yaml" \
  --intervention-config "$REPO/configs/routing/intervention_metric_mask_tokens.yaml" \
  --mask-config "$REPO/configs/routing/target_masks.yaml" \
  --target-mask-index "$ROI_INDEX" --roles "$REPO/configs/teachers/a01_scared.yaml" \
  --data-root "$A00_ROOT" --manifest "$FIT_MANIFEST" \
  --initialize-model-from "$MODEL_INIT" --cache-root "$KD_CACHE" \
  --max-steps 256 --output-dir "$NEW_EXTERNAL_RUN_DIR"
```

For the frozen comparison, change only the intervention config and output
directory. A sealed diagnostic evaluation uses the existing native intervention
assessment and a freshly bound FIT cost artifact. Do not tune weights/cutoffs on
VAL or final test. All earlier diagnostic VAL/CAL frames have backbone ancestry
overlap, so comparisons using those artifacts remain development-only.

## Acceptance and current limits

Synthetic validation covers token coordinates/holes, shared matching computation,
detached inputs, all component labels, risk gradients/learning, STOP, joint/frozen
ownership, strict resume, unknown exclusion, frozen outcome identity and event
scoring. Test evidence and source/config hashes are recorded externally under
`mask-token-risk-implementation-9a7yi_bn`.

Real sample smoke, native token training/evaluation, event-aware calibration,
matched latency/memory measurement, multiple seeds and final test8–9 have not run
in this implementation pass. Better real utility, risk discrimination and
render/geometry preservation remain hypotheses to test. Selecting three action
types alone is not acceptance. Scientific guard thresholds and outcome weights
were not tuned here.

### Completed implementation checks

The final reviewed source `315c16290c3a1015d89d89b53f7866489dae655c11e0012bdd0e884ce67e6596`
passed 214 focused tests in 121.16 seconds, with its source fingerprint unchanged
throughout validation. This includes 11 new test cases. Mypy passed all 230
source files; Ruff and format checks passed. CLI help and all new configurations
parsed successfully. An earlier regression exposed historical outcome serialization
compatibility; scalar-mode serialization was preserved and the full suite reran
successfully. No dependency installation or real training/evaluation ran.

Detailed file identities and validation evidence are in the external
`validation-report.json`, `implementation.diff` and `reviewed-source-tests.log`.
