# Local and global risk audit of the joint/frozen comparison

## Scope and evidence

The user requested checking local/global risk while retaining joint training as
the preferred direction and a separate frozen-student comparison. This pass
audits existing source, checkpoints and sealed logs. It does **not** implement
new heads, migrate checkpoints, tune thresholds or launch another training run.
The existing joint/frozen experiment is documented in
[joint/frozen comparison](joint_frozen_student_comparison.md).

Reviewed first-party source SHA-256:
`c48ef5b9c4944651d41b3f84fb3e19cfdd673b1bfba379f6222833b5e60d40d3`.

External derived evidence:
`/home/bkai/.local/share/reliableendo-gs/runs/risk-local-global-audit-41fkcjcd/`.
`report.json` records checkpoint/log hashes, resolved intervention configs,
source identity and audit script hash. `audit.py` reproduces the decomposition;
`focused-tests.log` records validation. Original artifacts remain intact.

Both branches contain 256 new training frames and 140 identical diagnostic
evaluation IDs. Their resulting actors and candidate masks differ; candidate
rows are not identical paired interventions. All diagnostic VAL images occur
in older backbone training ancestry. These results are development diagnostics,
not held-out generalization, final-test results or paper benchmark evidence.

## What the current implementation does

1. `metric_targets` uses `not feasible_changes(changes, policy)`. The canonical
   raw-unit guard checks all 12 physical components, including local depth/RGB,
   global depth, both views' PSNR/SSIM, outside RGB and geometry coverage.
   Gaussian tightening is applied through `metric_action_policy`. Thus the
   current harm label **already includes both local and global failures**.
2. `BalancedSelector` has independent stereo/Gaussian gain MLPs and independent
   stereo/Gaussian risk MLPs. Each risk MLP has **one output**, confirmed in the
   actual checkpoints by last-layer weights `[1,32]`. Its BCE learns the union
   event. There are no separately supervised local/global risk predictions.
3. Local/global gain supervision does not provide local/global risk supervision.
   The physical local gain auxiliary is local depth MAE; it is structurally zero
   for Gaussian repairs, which do not change stereo depth. Global utility may
   improve while local RGB exceeds its preservation tolerance.
4. The active 61-dimensional feature vector includes both local and global
   pre-action evidence. It is incorrect to say that all global context is
   missing. It lacks explicit predictions identifying which physical constraint
   will fail; the sufficiency of its current statistics is not established.
5. The separate offline `OutcomePredictor` supports per-component risk and
   constraint-margin objectives. That implementation exists, but these joint
   checkpoints use `BalancedSelector`. `TradeoffConfig.component_risk_weight`
   and `constraint_margin_weight` do not activate those objectives in the
   balanced joint selector merely by being present in the metric policy.
6. Local reconstruction uses left ROI plus its **fixed pre-action projected
   right footprint**, rather than the same pixel rectangle in both views.
   Missing complete metric observations are excluded from risk training and
   scoring, not silently counted as safe. GT remains separate from deployment
   inputs and detached selector features.

One scalar can mathematically learn an OR event. Its existence is not proof that
risk cannot be learned. Explicit decomposition is a proposed remedy for lost
supervision detail; it still requires a controlled comparison.

## Independent decomposition

For this audit, local consists of local depth MAE, local depth RMSE and local
RGB. Global/collateral consists of the remaining nine canonical components.
Outside RGB belongs to collateral/global. These disjoint groups exhaust all 12
components, and `harm_any = harm_local OR harm_global` is checked against the
existing canonical target for every complete observation. This is an audit
taxonomy, not a new data split or changed guard.

### All known evaluation candidates

| Branch/action | Known | Safe | Local only | Global only | Both | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint stereo | 157 | 118 | 39 | 0 | 0 | 9 |
| Joint Gaussian | 157 | 130 | 18 | 5 | 4 | 9 |
| Frozen stereo | 159 | 119 | 38 | 2 | 0 | 7 |
| Frozen Gaussian | 159 | 124 | 26 | 6 | 3 | 7 |

Joint stereo's 39 harmful candidates are all local failures: 27 local RGB,
17 local MAE and 14 local RMSE failures, with overlaps. A global-only safety
check would miss every one of them. Joint Gaussian has 22 local RGB failures
and nine global/collateral failures; four observations violate both groups.

### Actions actually selected

| Branch/action | Known selections | Local only | Global only | Both | Harmful | Mean predicted union risk | Actual harm rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint stereo | 12 | 4 | 0 | 0 | 4 | 33.35% | 33.33% |
| Joint Gaussian | 121 | 13 | 5 | 4 | 22 | 9.64% | 18.18% |
| Frozen stereo | 31 | 2 | 0 | 0 | 2 | 2.86% | 6.45% |
| Frozen Gaussian | 92 | 14 | 4 | 0 | 18 | 7.21% | 19.57% |

Three joint and four frozen selected Gaussian outcomes are unknown and excluded
from these harm-rate denominators. Four/13 STOP selections have exact zero
intervention harm by definition. STOP does not certify that base reconstruction
is good. The local and global counts must not be summed when both are present.

Matching a population's mean predicted and actual risk does not establish
discrimination. For joint selected stereo, harmful actions average **30.81%**
predicted risk versus **34.62%** for safe ones, although their overall means
match. The configured stereo cutoff is 51%, so substantial predicted harm is
allowed. This is a policy issue as well as a prediction issue.

For joint selected Gaussian, local-only harmful actions average **9.25%** risk
versus **9.91%** for safe actions. Frozen selected Gaussian similarly averages
7.40% for local-only harmful versus 7.24% for safe. Freezing the student has not
resolved this failure to distinguish local render harm.

Using the current union score, candidate union-harm ROC AUC is 0.5504/0.4416
for joint stereo/Gaussian and 0.7212/0.5553 for frozen. Selected subsets are
conditioned on the existing policy and are small; these are descriptive
statistics without confidence or significance claims. No local/global head
Brier or calibration score is reported because those predictions do not exist.

### Risk is actually learning

The new training segment provides 228 complete primary regions per action in
each branch, not 256 fully labelled frames. Joint has 200 known primary snapshots
and 800 risk updates; frozen has 199 and 796. Both exclude 77 unknown candidate
regions per action. All logged active snapshots have nonzero risk gradients and
parameter updates for **both** action heads; risk weight is 1 and each MLP is
clipped independently. The risk gradients are not absent or frozen.

Global harm labels are scarce in this segment: joint stereo six, Gaussian five;
frozen stereo three, Gaussian eleven. Multiple updates on the same snapshot do
not create independent evidence. A global risk head would need adequate safe/
harm support across FIT/CAL keyframes, not merely repeated BCE optimization.

## Recommended next implementation experiment — not implemented here

Keep the independent action branches and predict three events per action:

```text
STEREO   -> risk_local, risk_global/collateral, risk_any
GAUSSIAN -> risk_local, risk_global/collateral, risk_any
STOP     -> exact zero intervention risk
```

- Derive all labels from the single canonical physical guard owner, preserving
  action-specific tolerances and fixed before/after support. Supervise `risk_any`
  directly with the observed OR event; do not average local/global probabilities
  or assume event independence through `1-(1-p_local)*(1-p_global)`.
- Preserve geometry/render cause diagnostics within the local group. Binary
  BCE is already dimensionless. Do not multiply risk by 1,000 to match an mm loss.
  An optional margin auxiliary should use `constraint_margins_tensor`, immutable
  TRAIN scales and the same raw-unit bounds, with its own declared coefficient.
- Do not allow local and global risk to compensate for each other. If separate
  deployment gates are introduced, declare/calibrate their cutoffs on CAL and
  assess the combined decision, including whether predicted union probability
  is consistent with its component events. A maximum of probabilities is not
  automatically a calibrated union probability. Include calibration provenance
  and treat unsupported classes as unavailable rather than certified safe.
- Preserve utility weights and physical thresholds during the first head
  comparison. Report per-action/per-event Brier, discrimination, calibration,
  harm types, unknowns and achieved utility, plus depth and both-view render
  quality. Selecting all three actions is insufficient acceptance.
- Make checkpoint topology/version migration explicit. Historical single-risk
  weights and Adam cannot silently load into a new output shape. Match any
  resets between comparison arms and retain model/repair/normalizer ancestry.
- Joint remains the primary arm; apply the same risk design to the frozen
  comparison. Separately test repair-gradient feedback into the student as a
  second intervention, so a risk-head change cannot be credited with an actor
  objective change. No current evidence isolates a single causal mechanism.

## Validation in this pass

- Recomputed canonical risk/guard labels for all complete primary training and
  evaluation candidates, and confirmed selected recorded harm and per-component
  feasibility. Checked 256 training/140 unique evaluation IDs per branch.
- 96 synthetic individual-component violation/inclusive-boundary checks passed,
  plus zero-change reference checks. These execute the canonical guard; the
  audit does not reimplement guard thresholds.
- 56 existing focused tests passed in 9.72 seconds: `test_action_heads.py`,
  `test_metric_joint.py`, `test_balanced.py`. Synthetic tensor tests used the
  existing environment workaround of a small GPU0 CUDA bootstrap before CPU
  autograd. This does not certify cold CPU-only runtime portability.
- Source SHA-256 stayed unchanged. No new native evaluation, real smoke, teacher
  inference, cache generation, joint training, threshold search, full CI, final
  test8–9 or dependency installation ran. Only this documentation was added to
  the repository; derived analysis stays external.

**Verdict:** local/global failures are included in current union labels, but
explicit local/global risk learning is missing from the active joint selector.
Gaussian conditional risk remains weak even with a frozen student. Local/global
supervision is a justified next experiment, not an already validated fix.
