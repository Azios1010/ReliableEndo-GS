# Metric outcome action-selection pilot

## Scope and authority

The user approved implementation after the 2026-10-06 audit of geometry/render
tradeoffs. This is an **opt-in diagnostic**. It does not certify Gate I/II,
replace the historical selector, or launch a long joint run. A bounded predictor
pilot tests whether pre-action evidence can explain heterogeneous downstream
outcomes while upstream model/actions remain fixed for comparable labels.
The existing joint experiment remains a separate configuration/artifact.

The prior native diagnostic had 14 development frames: correspondence improved
local stereo-depth MAE by >=0.1 mm in 15/53 GT-covered regions, but all 15 were
rejected by proxy RGB harm. This does not establish metric/render improvement.
On the prior 112-frame development table, stereo-vs-STOP ranking had zero
informative rows at the raw margin 0.02; harm prediction did not beat constant
prevalence Brier. Historical evidence and output files are preserved.

## What is measured and learned

The outcome schema explicitly records twelve changes, positive meaning better:

| Components | Units | Interpretation |
| --- | --- | --- |
| Global/local depth MAE gain | mm | Fixed GT+base-valid denominator |
| Global/local depth RMSE gain | mm | Fixed denominator, sensitivity to larger errors |
| Global depth p95 absolute-error gain | mm | Linear-interpolated quantile on the same support |
| Left/right PSNR gain | dB | Each native rendered full image, data range 1 |
| Left/right SSIM gain | dimensionless | Canonical windowed SSIM, STOP and each outcome |
| Outside RGB gain | raw RGB MAE | Fixed left core/right projected footprint |
| Local RGB gain | raw Charbonnier RGB | Fixed two-view core support |
| Geometry coverage change | fraction | Coverage relative to valid GT |

Metric depth means **triangulated stereo depth**, not native rendered depth.
The production renderer remains RGB-only; no metric rendered-depth improvement
is claimed. PSNR, SSIM and Charbonnier are distinct objectives. Local/outside
RGB protect spatial effects not visible in full-image means. Missing local GT,
invalid geometry, or missing render metrics produces an unavailable outcome,
not zero improvement or a safe label. Lost support may not shrink a denominator.

`evaluation/tradeoff_outcomes.py` uses canonical triangulation, depth metrics
and renderer metrics. Quantile interpolation is pinned beside the calculation.
The existing action implementations execute unchanged: correspondence candidate,
learned stereo, and Gaussian repair. No private repair is created for the oracle.

## Constraints before utility

`routing/tradeoff.py` owns the canonical feasibility, utility coefficients,
STOP handling and allocation. `metric_tradeoff.yaml` is a strict non-regression
reference, with all allowed degradations zero except numeric epsilon 1e-6.
This is a declared conservative experiment, **not a calibrated optimum**.
The 0.05/0.10 dB sensitivity values from the audit are not selected defaults.
Changing thresholds requires a new contract hash and development evidence;
never tune on final-test datasets 8/9 or force balanced action frequencies.

Each geometry, render, local/outside and coverage constraint is checked in its
raw declared unit. Statistical scales cannot silently change safety limits.
The allocator then considers positive normalized utility and a measured-cost
budget. Utility contains global/local geometry, both-view PSNR/SSIM, local RGB
and outside RGB terms. RMSE/p95/coverage constrain it rather than being counted
again in the scalar score. Utility weights are explicit pilot preferences.

A predictive decision must satisfy all predicted component constraints and
predicted violation probability <=0.1, else abstain. No feasible positive
candidate means STOP. Predictive constraints **do not guarantee actual safety**:
report false-safe decisions, per-action Brier/ECE, observed degradation and
unknown metric coverage. This cutoff is an uncalibrated pilot hyperparameter.

## Small outcome predictor and guiding harness

`routing/outcome_predictor.py` has independent change/risk encoders:

- Changes: 61 -> 32 -> 32 -> 36, reshaped `[regions,3,12]`.
- Risk: 61 -> 32 -> 32 -> 3 logits.
- Pre-action features are the existing 61 deployment diagnostics. No GT,
  teacher disparity, post-action observations or development timings enter them.
- The Gaussian action cannot change stereo disparity. Its five depth changes
  and coverage change are exactly zero by structural action contract. This
  does not certify other geometry properties such as rendered depth.
- Feature median/IQR and per-component Q75(abs(change)) scales fit **only** the
  collected training table, then freeze into the predictor artifact. Scales are
  common across actions; constants retain the ability to vary at deployment.
  Declared floors in metric units prevent zero scales; no action-specific
  multiplication inflates stereo utility to produce diversity.
- Normalized SmoothL1 supervises the separate component outputs. Violation BCE
  has its own encoder, with bounded training-only positive-class weighting.
  Prediction removes the known class-weight log-odds shift; this is not held-out
  calibration. Change and risk encoders are clipped independently so risk
  gradients cannot suppress change updates through a shared clipping multiplier.
- Ranking uses observed normalized margins among known-safe actions and STOP,
  with only a numerical tie threshold. It does not suppress all small raw
  stereo gains using the old margin 0.02.

The experiment compares a training-mean/prevalence constant, linear predictor
and MLP, all using the same training table and contract. No developer-set
multiplier is inferred from a raw scalar loss ratio. It logs the component,
risk and ranking objectives over the bounded fit. There are 7,367 trainable
parameters in the default MLP (verify this count through inspection, not FLOPs).

`load_outcome_predictor` checks schema, tensor shape/dtype/finite values,
normalization, structural invariance, config and frozen action/candidate
identity. It imports no upstream model or optimizer state.

## Immutable table collection and partitions

`selection collect-outcomes` extends the existing native intervention collector:

1. Verify pinned original-SCARED roles, manifests, candidate and checkpoint.
2. Choose deterministic, hashed, per-fold subsets with keyframe round-robin.
   Frame identities and subset hash are saved; no old manifest is edited.
3. Construct deployment state and 61 features without GT; execute all eligible
   region/action interventions from this same frozen pre-action state.
4. Only then resolve GT for offline outcome supervision and verification.
5. Save immutable raw metrics, fixed-support depth snapshots, component vectors,
   coverage/availability, feature inputs, native timings and scientific identity.

The script takes 8 training frames/fold (56 frames; normally 224 regions and
672 repair interventions) and 4 development frames/fold (28 frames; normally
112 regions and 336 repair interventions). Original datasets 1–7 only; 8/9
are final test and never used. Train/development frames must be disjoint.
Both inherit the current manifests. Shared keyframes are counted and disclosed;
this is not independent sequence validation. Scientific claims need a larger,
approved grouped evaluation with multiple seeds and failure-mode coverage.

A table is accepted only when its checksum, schemas, sample IDs, role, action
completeness, candidate/checkpoint/source/device identities and timing records
validate. Missing observations are masked in supervision, explicitly counted,
and cannot certify an oracle action. Failed/incomplete collections have no
complete `result.json`; they cannot be fitted as success.

## Costs and comparisons

Native base-with-matching, base-without-matching and feature costs are measured
separately. The latter base is checked to preserve disparity and Gaussian field
values; fixed/constant policies can omit matching evidence. Each action is
warmed and repeated three times with device synchronization, recording median,
maximum and raw wall-ms samples. Base/features are single-pass; this small
pilot is not a publication-ready latency benchmark.

Predictive action cost estimates are the maximum observed training costs.
Development intervention timings are consulted only after predictive selection
for reporting. A measured oracle may see costs and metrics as an unattainable
ceiling. Tiny predictor forward/materialization is measured five times/frame
on the same GPU; allocation, state and feature costs are included. I/O and
GT/oracle metric computation are excluded explicitly. CPU functional fitting
may replay GPU tables, but **cannot establish a matched native runtime claim**.
No repair action is rerun by table replay; a complete deployment benchmark is
still required to certify end-to-end latency.

The pilot reports budgets 250/500/1000 ms for constant, linear, MLP, oracle,
cyclic fixed actions and MLP restricted to each repair source. Report raw
component changes, constraint violations, unknown metrics, actual cost overruns,
calibration, and keyframe-cluster bootstrap intervals for safe utility versus
fixed Gaussian, Gaussian-only MLP and oracle. Unsafe selections have zero
safe utility, while their raw utility and violations remain visible. An
apparent gain must persist against strong single-source policies at matched
cost; diversity alone is insufficient. No automated Gate II approval exists.

## Entry points

```bash
python -m reliable_endo_gs.cli.selection collect-outcomes --help
python -m reliable_endo_gs.cli.selection outcome-pilot --help
```

`scripts/run_outcome_pilot.sh` is a thin bounded entry point. Supply existing
runtime Python, external processed root, pinned checkpoint/candidate probe,
immutable train/development manifests and external run parent through:

```text
RELIABLE_ENDO_PYTHON
RELIABLE_ENDO_DATA_ROOT
RELIABLE_ENDO_CHECKPOINT
RELIABLE_ENDO_PROBE_REPORT
RELIABLE_ENDO_TRAIN_MANIFEST
RELIABLE_ENDO_DEV_MANIFEST
RELIABLE_ENDO_RUN_ROOT
RELIABLE_ENDO_GPU=0
```

```bash
bash scripts/run_outcome_pilot.sh
```

The script stops if a stage fails, uses a unique external output directory,
collects both bounded tables, fits only the predictors for 100 full-batch
iterations, and writes `selector/result.json`, `decisions.jsonl`, training
losses and predictor artifacts. It performs no teacher/cache generation,
upstream/joint training or final-test access. No native CPU fallback exists.
No real dataset, copied image, disparity or artifact is committed.

## Remaining gates

Inspect oracle feasibility/diversity, learned component accuracy/calibration,
measured quality/geometry changes and utility/cost regret first. Strict
constraints may legitimately produce STOP or Gaussian dominance. Preserve
that negative result. If outcomes are useful but prediction is weak, perform
feature/loss/normalization ablations on the frozen table. If the action harms
render, test head/action adaptation as a separately identified pilot before
changing action labels. A final joint integration and deployable router require
approved evidence and artifact identities; old scalar-selector checkpoints
cannot resume into this outcome predictor.

## Opt-in balanced rules and train-group risk remediation

`configs/routing/metric_tradeoff_balanced.yaml` selects
`metric_action_tradeoff.v2`. The historical strict v1 config, checkpoint tensor
layout and configuration hashes remain supported. Neither experiment is the
default joint/deployment path or evidence of Gate II approval.

### Rules and utility

The proposed bounded operating point allows PSNR degradation of 0.01 dB per
view, SSIM degradation of 0.0005 per view, outside RGB increase 1e-5 and local
RGB increase 1e-4. MAE, RMSE, p95 and coverage limits stay at zero. These are
pilot preferences motivated by the threshold audit, not measured noise floors,
clinical limits or a statistically validated optimum. Guards retain raw units.

The utility owner remains `routing/tradeoff.py`, shared by oracle, training
and predictive allocation. With `s_c` fitted only on predictor-fit training
frames, v2 uses the signed smooth bounded term:

```text
x_c = gain_c / s_c
h_c = x_c / (1 + abs(x_c) / utility_gain_cap)
utility = sum(weight_c * h_c)
```

Geometry weights (global/local MAE) sum to one. Render weights (left/right
PSNR/SSIM) also sum to one. Their total capacities are equal; this does not
prove that their practical clinical/research value is equal. RMSE/p95/coverage
and local/outside RGB remain constraints. RGB does not receive additional
utility reward duplicating render quality. Smooth saturation prevents extreme
normalized magnitudes from dominating without hard-clamp ranking ties.
Infeasible actions cannot purchase eligibility with a high scalar score.
Raw metric changes, violations, unknown support and fixed-Gaussian comparisons
must remain visible because a scalar score is not a complete quality verdict.

### Risk and predicted guard crossings

The v2 risk encoder predicts a joint violation logit and twelve auxiliary
per-metric logits for each repair. The joint event is learned directly:
correlated metric risks are not multiplied or added as independent events.
Both joint and component BCE are unweighted proper scoring losses. Auxiliary
labels call the same canonical raw-unit feasibility owner. Structurally invariant
Gaussian depth/coverage outputs are zero and masked out of auxiliary learning.

The change encoder additionally receives a BCE surrogate on normalized signed
distance to each guard. This addresses the earlier mean-regression prediction
that nearly all Gaussian outcomes are feasible despite measured render harm.
It does not replace raw regression, change the physical constraints or allow
GT into inference features. Separate encoders and separate gradient clipping
remain intact. Component predictions and risk still require empirical checks.

One seed-hashed training keyframe per fold is reserved for calibration. Remaining
training frames fit the encoders, feature/metric statistics, ranking and costs.
Frame IDs, keyframe IDs, parent table SHA-256 and a partition hash are recorded.
Existing dataset roles, train/development manifests and final-test data are
unchanged. Development is evaluation only. Calibration uses a fixed small
temperature/intercept grid minimizing joint log loss. The risk cutoff stays
0.1; it is not tuned using development. Both safe and unsafe calibration classes
need at least two distinct keyframes per action. Insufficient support yields an
explicit unavailable action and abstention; its sentinel one is not reported
as a calibrated probability or a successful Brier/ECE result. Auxiliary component
probabilities are diagnostics, not independently calibrated deployment gates.

Artifact v2 loading checks calibration tensors against their status/partition,
config, action and candidate identities. Historical v1 predictors are not
converted silently. A small training-only calibration does not guarantee safety
or independent generalization; shared-keyframe development remains disclosed.

### Explicit policy replay without recollecting native actions

The same frozen action outcomes can supervise different proposed policy rules
when action and metric semantics are unchanged. `--relabel-policy` explicitly
authorizes v2 policy replay. Immutable collection configuration/checksums are
verified and kept; new policy, partition and artifact identities are saved
separately. No table header or native measurement is rewritten. Omitting the
flag still rejects mismatched collection/policy contracts.

```bash
python -m reliable_endo_gs.cli.selection outcome-pilot \
  --training-table "$OUTCOME_RUN/train" \
  --development-table "$OUTCOME_RUN/development" \
  --config configs/routing/metric_tradeoff_balanced.yaml \
  --relabel-policy --device cpu --budgets-ms 1000 \
  --output-dir "$NEW_EXTERNAL_OUTPUT"
```

The output directory must not exist and must be outside the repository.
CPU mode fits only the small predictors on saved feature/outcome tables;
it runs no teacher, repair, renderer or upstream model. It is not a native
latency benchmark. `render_changes_independent_of_gt` reports each render metric
on all available render observations even when local GT makes the full geometry
vector unavailable. Unknown geometry is never certified safe or imputed zero.

The audited small pilot still needs to demonstrate geometry benefit, risk
calibration and superiority over fixed/GS-only baselines. Increasing repair
frequency, lowering the scalar training loss or merely changing the rules
does not establish the selector contribution.

## Opt-in grouped validation and regularized predictor fitting

Pass `--training-policy configs/training/outcome_regularized.yaml` to enable
`outcome_training.v1` with the balanced v2 metric policy. These controls have
their own configuration hash; they do not change utility weights, physical
guards, risk cutoff, dataset roles or the default joint path. Omitting the
option preserves the historical fitting procedure.

The immutable TRAIN table is divided into three disjoint keyframe views:

- Fit: encoder updates, feature/metric normalization and action-cost estimates.
- Validation: one additional seed-hashed keyframe per dataset, used only to
  select a checkpoint. At least three training keyframes per dataset are needed;
  there is no fallback to a frame split.
- Calibration: the original one seed-hashed keyframe per dataset remains held
  out for temperature/intercept fitting after the best checkpoint is restored.

No development outcome chooses a checkpoint, stopping time, normalization or
calibration parameter. The parent train/development manifests are not changed.
Existing development frames share keyframes with the parent training manifest;
the internal three-way split does not turn that development set into an
independent sequence test.

The declared policy uses AdamW with weight decay 0.01 and hidden dropout 0.1,
up to 1,000 optimizer steps. From step 100 onward, validation is evaluated every
25 steps; six evaluations without an improvement greater than 0.001 stop the
fit. Selection minimizes the equal-keyframe mean of normalized metric SmoothL1
and proper joint-risk log loss. This is a predictive checkpoint objective,
not a deployment safety guarantee or a development utility optimization.
Validation uses fit-only scales and inference mode. The exact best model state
is restored before calibration. Dropout is disabled at inference, and historical
parameter names/layouts remain compatible. Artifacts retain all three frame
lists, partition hash, training policy/hash, best/completed steps and selection
score; the loader rejects inconsistent selection provenance.

```bash
python -m reliable_endo_gs.cli.selection outcome-pilot \
  --training-table "$OUTCOME_RUN/train" \
  --development-table "$OUTCOME_RUN/development" \
  --config configs/routing/metric_tradeoff_balanced.yaml \
  --training-policy configs/training/outcome_regularized.yaml \
  --relabel-policy --device cpu --budgets-ms 1000 \
  --output-dir "$NEW_EXTERNAL_OUTPUT"
```

For a new native collection, `scripts/run_outcome_pilot.sh` accepts
`RELIABLE_ENDO_TRAIN_PER_DATASET=16`, `RELIABLE_ENDO_DEV_PER_DATASET=4`,
`RELIABLE_ENDO_TRADEOFF_CONFIG` and `RELIABLE_ENDO_TRAINING_POLICY`, in addition
to its existing external-path variables. This requests 112 train and 28
development frames from datasets 1–7 using the existing deterministic sampler.
Additional frames are not automatically additional useful stereo examples:
that requires inspecting the newly measured interventions. The script requires
one visible GPU before allocating an output directory and never substitutes
CPU rendering. It trains only the outcome predictors on frozen actions.

### Bounded replay evidence (2026-10-07)

On the existing immutable 56-frame TRAIN / 28-frame development tables, the
three TRAIN views contain 27 fit, 14 validation and 15 calibration frames.
For seed zero, the MLP chose step 150 and stopped at 300. On development it
selected seven Gaussian repairs and 21 STOPs, no stereo repair. There were zero
measured violations among 26 decisions with complete metric support and two
unknown decisions. Mean left/right PSNR gains over all 28 render observations
were 0.248/0.139 dB. It still did not beat fixed Gaussian repair: the paired
safe-utility difference was -0.243, keyframe bootstrap interval
[-0.392, -0.100]. This is a negative contribution result on reused development,
not evidence of Gate II passing.

Matched-cohort unregularized controls used the same 27/14/15 split. At 100 and
1,000 steps, raw stereo-risk validation Brier scores were 0.100 and 0.181;
the 1,000-step model selected two stereo repairs and incurred three measured
violations overall. These controls demonstrate overfitting on this small table;
they do not isolate the causal effect of dropout versus weight decay versus
checkpoint selection. They also cannot be directly compared to the old
41-frame fit as though training support were equal. CPU replay is not a native
runtime benchmark. The expanded native collection has not run: the current
Codex runner reports zero CUDA devices.

## Opt-in risk discrimination and conditional context

The 2026-10-07 remediation adds `outcome_training.v2`; historical v1 training
hashes and artifacts remain supported. Both new configurations keep the
physical metric policy, risk cutoff0.1, upstream/action weights and data roles
unchanged. They are isolated experiments, not new deployed defaults.

`configs/training/outcome_risk_discrimination.yaml` enables:

- A per-action auxiliary `mean softplus(safe_logit - unsafe_logit)` over known
  safe/unsafe pairs. A one-class action contributes no pairs. Joint BCE remains
  unweighted. The auxiliary changes score fitting, so its sigmoid is not
  automatically a calibrated probability; independent calibration is required.
- Separate gain/risk checkpoint selection on TRAIN-derived validation keyframes:
  normalized regression for gain, proper joint log loss for risk. The loop stops
  when both criteria exhaust patience, then restores each exact selected encoder.
- Continuous monotone Platt calibration on different TRAIN keyframes, minimizing
  equal-keyframe log loss plus0.01 ridge on log-slope/intercept. Fixed bounds are
  log-slope[-2,2] and bias[-4,4], with at most80 LBFGS iterations. Identity remains
  the fallback if the penalized objective does not improve. This fits probability
  scale without searching for a risk cutoff or a preferred stereo frequency.

`configs/training/outcome_conditioned_risk.yaml` also augments risk inputs with
36 normalized predicted guard margins, derived by the canonical constraint owner
from the gain encoder's predictions. Public inputs stay the same61 pre-action
features; no measured outcome, GT, teacher target or post-action render enters
this context. Margins are detached and clipped to[-5,5]. Artifacts explicitly
list these97 risk input names, and incompatible context is rejected on load.
After selecting gain, a separate bounded risk refit uses that exact fixed gain
encoder in inference mode. This fixes the risk input distribution while learning
the small risk network; it does not freeze or retrain the upstream joint model.
Risk refit has its own maximum1,000 steps, seed, stopping history and restored
state. The total bound is therefore2,000 steps per predictor. Gain/normalization
invariance is checked. All source/partition/config identities are retained.

Reports now expose safe action recall, false-unsafe counts and the empirical
violation fraction among known regions accepted by the risk cutoff, alongside
false-safe counts. These are classifier diagnostics before gain guards and final
allocation. Unknown observations and unsupported risk sources are not reported
as successful probability estimates. Low aggregate Brier alone does not establish
useful low-risk action recognition.

### CPU replay and negative evidence

`scripts/run_outcome_risk_replay.sh` runs the historical reference and both
proposed modes on immutable native train/development tables. It requires an
existing runtime Python and table root, limits numerical libraries to one CPU
thread and creates a new external output directory. It performs no CUDA,
teacher, repair, native rendering, cache generation or upstream/joint training.

```bash
RELIABLE_ENDO_PYTHON="$RUNTIME_PYTHON" \
RELIABLE_ENDO_OUTCOME_TABLE_ROOT="$COMPLETED_NATIVE_RUN" \
RELIABLE_ENDO_RUN_ROOT="$EXTERNAL_OUTPUT_PARENT" \
bash scripts/run_outcome_risk_replay.sh
```

On the112-frame native TRAIN table, three disjoint views contain54 fit,
29 validation and29 calibration frames; development contains28 frames.
The seed-zero matched CPU replay gives:

| Mode | Gaussian / STOP decisions | Stereo Brier | Stereo safe recall at cutoff | Mean PSNR gains left/right dB |
| --- | --- | ---: | ---: | --- |
| Reference | 14 /14 | 0.102419 | 0 | +0.511065 /+0.240807 |
| Risk discrimination | 13 /15 | 0.098746 | 0 | +0.462327 /+0.227527 |
| Conditional risk | 12 /16 | 0.100069 | 0 | +0.425717 /+0.201963 |

All three select zero stereo actions and have zero measured violations among26
complete decisions, with two unknown decisions. Neither new mode improves the
selector contribution or rendering over the CPU reference. The native GPU
reference selected15 Gaussian repairs; CPU/GPU random/dropout behavior and
numeric kernels differ, so the CPU reference is used for the matched comparison.
All PSNR means use28 available render observations, including decisions whose
geometry support is incomplete. This is not a latency benchmark.

The label audit verifies joint violations equal the canonical OR of component
violations; unknown cells are excluded. Fit contains32 safe stereo regions among
195 complete regions; validation11/112 and calibration16/114. Geometry RMSE/p95
are frequent vetoes. A TRAIN-only3-neighbor diagnostic recognized3 safe validation
regions but also accepted4 unsafe ones, so it was rejected and not integrated.
The tested changes do not show reliable low-risk discrimination. Do not lower
the cutoff or relabel degraded RMSE/p95 as safe to obtain stereo diversity.
Additional pre-action evidence or genuinely safer stereo interventions need a
new identified native study. Gate II remains unpassed; these negative results
must remain evidence rather than becoming deployed defaults.
