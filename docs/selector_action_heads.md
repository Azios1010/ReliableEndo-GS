# Independent stereo and Gaussian selector heads

The subsequent controlled comparison and opt-in checkpoint guard remediation
are recorded in [Controlled selector comparison](selector_controlled_comparison.md).
All three architectures/optimization variants can select all three actions;
that evidence does not establish safe or superior routing. Historical component
minimum checkpoint selection below remains the default. The new refit-only
`validation_guard_mode: action_objectives` guards action objectives, both risk
BCEs and ranking while permitting regression/local-auxiliary tradeoffs within
an action; it does not alter physical utility or safety thresholds.

## Scope

The user approved separating the two action heads on 2026-10-11. This is an
opt-in extension of the existing student/action joint experiment. It does not
certify the historical research gates or demonstrate improved routing.

The preceding 128-step native check completed with the shared-action selector.
Its logged 96 observed region snapshots had mean stereo/Gaussian utility
-0.1008/+0.0867. Their paired Pearson correlation was 0.0243: different marginal
distributions do not establish opposing per-region utility. Snapshots came from
moving model/action weights during TRAIN, not a frozen held-out evaluation.

## Architecture

```text
Detached pre-action features -> common frozen TRAIN feature normalization
  |-> stereo gain MLP   -> global utility, local auxiliary
  |-> stereo risk MLP   -> physical-harm logit
  |-> Gaussian gain MLP -> global utility, local auxiliary
  `-> Gaussian risk MLP -> physical-harm logit

STOP: utility=0, local auxiliary=0, risk=0; no trainable head
```

No trainable parameter is shared between actions or between gain and risk.
Each gain trunk still serves its own global/local outputs and ranking objective.
All heads receive the existing pre-action features; this change does not add
action-specific features, outcome-component prediction or new supervision.
Existing local/global counterfactuals, unknown-GT exclusion, signed utility,
raw-unit harm guards and TRAIN/CAL roles remain the same.

Feature statistics and global/local gain scales remain common across actions.
Decoded global utilities therefore remain comparable with each other and STOP.
The risk MLPs have independent parameters; their probabilities still need
held-out calibration after repair weights stop moving. Raw utility, class-prior
weights, risk thresholds and action quotas are unchanged. Action-local gradient
balancing is a change to optimization, not to the measured utility labels.

Each of the four MLPs is clipped independently, so another action's gradient
norm cannot consume its clipping budget. With `selector_balance_mode: per_action`,
global/local loss gradients have separate EMA/reference groups for stereo and
Gaussian. Each reference sees only its own action. Cross-action ranking keeps
its declared fixed coefficient and is excluded from both EMA groups. Student/
repair and selector ownership, detach boundaries and update schedule stay in place.

The two action losses still have equal mass in the total regression objective.
Feature normalization, decoded utility scales and the signed utility formula
remain common. Independence of balancing is conditional on the same fixed
normalization and known-label support; pooling TRAIN normalization is intentional.

Both action heads log normalized global/local loss, harm BCE/Brier/rate and raw
utility MAE. Joint updates also record gradient, parameter and optimizer-update
norms for each of the four MLPs. The new scalar diagnostics are read back together
to avoid a separate GPU synchronization for every added metric. No performance
improvement is claimed without a benchmark.

## Activation and checkpoint identity

Use this configuration in the existing student CLI:

```bash
--intervention-config configs/routing/intervention_metric_action_heads.yaml
```

It differs from `intervention_metric_common_utility.yaml` by
`selector_architecture: separate_action_heads` and `selector_balance_mode: per_action`.
Historical configurations retain
`shared_actions` and their prior serialized hashes/state layouts. The new
architecture and balance mode are included in the intervention/checkpoint identity.
The per-action balancer stores four EMA/weight entries instead of the shared
three. Resume across architectures or balancing modes is rejected before mutation.
Start a new selector/optimizer; existing model-only initialization remains
available through `--initialize-model-from`. Preserve old runs and artifacts.

The previous source-bound launcher must not be used unchanged after this code
change: its reviewed source fingerprint deliberately rejects changed source.
A subsequent native check needs a new reviewed binding and external output
directory. No training run is launched by this implementation.

## Whole-head selection in stationary refit

For per-action mode, refit selects a whole selector snapshot from one step:
stereo/Gaussian gain/risk are never assembled from separately selected steps.
VAL uses the fixed, unbalanced normalized objective rather than the moving
training EMA weights. A candidate must improve the joint gain/local/ranking/harm
objective by `SelectorRefitConfig.min_delta` and may not regress any action's
global/local/harm loss or joint ranking loss beyond that same declared tolerance.
The guard compares against the component minima of accepted checkpoints, so
repeated small regressions cannot accumulate unchecked. Rejected candidates do
not update the floors. The selected step is the same for gain and risk.

This is a conservative loss/ranking guard, not a physical safety guarantee or
an evaluation of deployed decisions at measured cost. It may retain an early
checkpoint when actions have incompatible validation tradeoffs; logs expose
candidate components and acceptance. Actual utility/harm, calibration and cost
still need frozen-action held-out assessment. Historical shared balancing/refit
behavior remains available for controlled comparisons.

## Verification and limitations

Focused tests cover independent parameters and gradients, output ordering,
STOP zeros, common signed gain decoding, learning in all four networks,
state round trips, configuration identity and checkpoint rejection. Existing
joint-update/resume tests run with both architectures, including missing GT
and gain-only refinement. Consumer suites exercise legacy compatibility.

Native training, held-out selector quality, risk calibration and additional
runtime overhead must be measured separately. Increased parameter isolation
does not prove that features distinguish safe stereo opportunities or that
the stereo repair itself improves metrics.

### Validation completed for the initial head separation

- 154 tests passed in 95.82 seconds: action-head isolation, selector
  optimization/balancing, metric joint learning, refit/execution/probe,
  student joint learning and runtime.
- 22 tests passed in 6.72 seconds in a targeted recheck of action heads,
  calibrated selector/refit and execution, including the added architecture
  binding regression. This recheck overlaps the preceding suite.
- `mypy src` passed all 227 source files. Scoped Ruff lint/format checks and
  `git diff --check` passed.
- Successful tests used the existing runtime environment with physical GPU0
  initialized; renderers/data/teachers were synthetic. They are not evidence
  of native renderer training or real-data selector quality.
- The initial developer-environment attempt with GPU visibility hidden aborted
  in PyTorch's first CPU backward (exit 134). Its traceback is retained. It was
  not counted as a pass; the abort cause was not independently established.
  An initial mypy variable-type error was fixed before successful validation.

External logs:
`/home/bkai/.local/share/reliableendo-gs/runs/selector-action-heads-vpvxuk49`.
No real training, teacher inference, dataset changes or dependency installation
was performed. The prior 128-step run remains unchanged.

### Files changed for this task

- `src/reliable_endo_gs/config/intervention.py`: versioned architecture setting.
- `src/reliable_endo_gs/routing/balanced.py`: independent action networks,
  parameter ownership and clipping/trunk interfaces.
- `src/reliable_endo_gs/routing/calibrated_selector.py`: architecture binding.
- `src/reliable_endo_gs/training/selector_optimization.py`, `selector_probe.py`:
  probe both gain trunks, clip each MLP independently.
- `src/reliable_endo_gs/training/selector_fitting.py`, `student_learning.py`:
  independent MLP clipping in stationary and joint learning.
- `configs/routing/intervention_metric_action_heads.yaml`: opt-in configuration.
- `tests/routing/test_action_heads.py`, `test_selector_optimization.py`:
  action isolation, compatibility and both-architecture resume coverage.
- `PLAN.md` and this document: explicit experiment/activation boundary.

Existing unrelated working-tree modifications were preserved.

## Follow-up remediation after the read-only audit

The user approved fixing action-local balancing, diagnostics and checkpoint
selection after the audit demonstrated indirect coupling through a common
balancer. The regression test fixes weights/features/stereo labels and changes
only Gaussian labels, with ranking disabled: stereo EMA weights and gradients
must be identical. A separate test keeps cross-action ranking enabled and checks
that this intended coupling remains. Additional cases cover declared balance
identity, invalid group ownership, whole-head selection, joint updates and resume
with missing GT/gain-only refinement. Follow-up validation results are recorded
separately; the initial test counts above are historical evidence.

The follow-up changes `config/intervention.py`, `routing/balanced.py`,
`routing/calibrated_selector.py`, `training/gradient_balance.py`,
`training/selector_optimization.py`, `training/selector_probe.py`,
`training/selector_fitting.py`, `training/student_learning.py`, the opt-in
action-head configuration, and the action-balance/head/optimization tests.
`PLAN.md` records the approved boundary. No utility formula, risk threshold,
dataset role or teacher/model ownership was changed.

### Follow-up validation status (2026-10-11)

- `mypy` passed all 227 source files. Scoped Ruff lint/format checks passed
  for eight source files and three test files. `git diff --check` passed.
- The first combined test attempt was stopped while reading the Windows mount.
  The retry with a Linux cache emitted 34 successful test reports, then exceeded
  its 15-minute limit (exit 137). This is an incomplete suite, not a full pass.
- A source/test/config snapshot on Linux matched SHA-256 for all 12 task files.
  The combined retry expired during PyTorch/Dynamo import (10 minutes, exit 124),
  and a separate core retry in the developer environment expired during PyTorch
  import (5 minutes, exit 124), both before test execution. These attempts do not
  establish runtime correctness. The extra balance-identity rejection cases
  added after the partial run still need a successful test run.
- No real training, teacher inference, native renderer check, calibration,
  held-out evaluation or performance measurement was run for this follow-up.
  Do not interpret the completed static checks or initial historical test
  counts as approval to launch long training with the new balancing mode.

Evidence, traceback logs and snapshot bindings are outside the repository:
`/home/bkai/.local/share/reliableendo-gs/runs/selector-action-balance-akepso0d`.

### Subsequent focused runtime check (2026-10-11)

The user requested a short trial. The follow-up finished successfully with
**44 passed, 12 deselected in 435.30 seconds; exit code 0**. It covered the
action-balance and action-head suites, optimization/configuration checks, and
all six fresh joint-update/resume variants for `separate_action_heads` with
`per_action` balancing, including unknown GT and gain-only refinement.
The twelve fresh joint/resume variants for the two shared-balancing modes
were deliberately deselected; this was not a rerun of every consumer suite.

Physical GPU0 was selected by UUID. The tests used synthetic data/teachers and
renderers; the fresh joint/resume cases exercised CUDA model, repair and
selector updates. Twelve source/test/config identities were checked against
the current checkout before running the Linux source snapshot.

The four-step synthetic stationary refit reduced loss from 1.3848138 to
1.3745148 and selected step 4 for both gain and risk. Its risk BCE remained
near 0.693; this tiny check does not demonstrate useful harm discrimination.
Real SCARED training, native renderer training, held-out utility/harm and
performance measurements were not run. Earlier interrupted attempts remain
preserved as unsuccessful attempts.

Logs, source bindings and the report derived from the console output:
`/home/bkai/.local/share/reliableendo-gs/runs/selector-action-trial-meqjjxx8`.
