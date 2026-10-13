# Fresh-snapshot selector optimization

This opt-in remediation continues the user-approved joint experiment in PLAN.md.
Student/reference, repairs and selector continue learning. It does not certify
Gate I, Gate II or calibrated uncertainty.

The fixed-table diagnostic established that the existing MLP can fit TRAIN
stereo winners, but also found worse development gain/risk errors after a long
head fit. The experiment here increases bounded optimization exposure without
retaining stale counterfactual labels or introducing GT into deployment features.

## Implementation

`intervention_metric_selector_fit.yaml` keeps the preceding stereo-specific
utility, physical depth/render guards, risk cutoffs, masks, data and seed. Its
declared changes are:

- Four selector Adam updates for each observed post-burn-in TRAIN snapshot.
- Selector-only Adam weight decay 0.01; student/action optimizers are unchanged.
- For verified safe ranking pairs, require at least half the measured normalized
  utility gap, with the existing constant margin as a floor.

All four head updates consume the same detached features and labels measured
before the model/action update. Model/action weights stay unchanged during these
head updates. No historical outcome replay is introduced. Unknown GT labels and
empty target candidates do not update the selector. TRAIN normalization remains
frozen after burn-in. Harm supervision remains the actual binary raw-unit guard
outcome; no class rebalance, label smoothing or relaxed safety threshold is used.

Ranking changes only the fitting objective. It does not alter the canonical
utility, reward stereo independently of measured outcomes, or require a minimum
number of any selected action. Prediction and the risk/budget filter are unchanged.

Optimizer counters record every real Adam step. Resume checks per-parameter Adam
steps, snapshot multiplicity and selector-balancer observations before changing
live weights. Neutral defaults omit the new fields from serialized configs,
preserving historical hashes and the one-update path.

## Validation protocol

Run focused synthetic ownership/resume/unknown-label tests, asset preflight and
a fresh bounded native pilot on the existing 56 TRAIN / 28 development worklist
for three joint epochs. Compare selected utility, raw depth/render changes,
violation/unknown counts, oracle regret and gain/risk errors against the preceding
stereo-specific pilot. Development must not fit weights, normalization or stopping.
The coefficients above are declared before the new native evaluation. The existing
risk cutoffs were chosen post hoc on earlier development results and remain
uncalibrated pilot settings; retaining them controls this comparison.

This is a test of faster current-snapshot optimization and regularization. It
does not establish that these changes resolve cohort shift or identify its cause.
If development deteriorates, retain the negative result and do not promote the
experimental config or force action diversity.

## First native result and targeted revision

The all-head pilot completed 168 joint updates and 496 selector updates.
On 28 development frames it selected STOP/stereo/Gaussian 6/1/21, with mean
utility 0.1280, four metric violations and one unknown intervention. The preceding
pilot had utility 0.1456. Gain learning improved: development stereo gain MAE
fell from 0.1506 to 0.1096 and true-safe ranking accuracy rose from 0.586 to 0.897.
However, stereo risk blocked seven of eight regions with stereo-best safe outcomes.
This revision is not accepted as a better deployment selector.

`intervention_metric_gain_fit.yaml` therefore separates optimization exposure:
four gain updates, one harm update per observed snapshot. Gain uses weight decay
0.01; harm retains its original zero decay. Extra gain calls exclude BCE entirely,
so harm gradients remain `None` and Adam must not change harm parameters, moments
or step counters. The final joint call trains both networks. Safety cutoffs and
utility remain unchanged. Checkpoints validate gain steps against the actual
selector Adam-call count and harm steps against that count divided by the declared
snapshot multiplicity. Normalization and counterfactual generation are unchanged.

This targets the observed gain underfit without also accelerating risk fitting.
It is not a calibration method; the resulting risk predictions still require
independent verification.

Native student/action weights differed across the preceding two runs, with small
gradient differences already at step one, before selector learning began. Thus
the cross-run metric difference cannot isolate the head modification. A paired
log replay on the new artifact compared both heads on identical pre-action
features/outcomes, retaining unknown GT and measured costs. Old-head utility was
0.1715 versus new-head 0.1280, each with four violations. This replay measures
decisions, not fresh runtime, and confirms the negative deployment result on that
artifact. All original runs, source snapshots and generated reports are retained.

## Gain-only native result: deployment acceptance failed

The second pilot completed three joint epochs, 168 model updates, 168 repair
updates, 496 gain Adam steps and 124 harm Adam steps on physical GPU 0, exit 0.
Native checkpoint validation and read-only development evaluation succeeded.
The evaluated checkpoint's weights were unchanged by evaluation; TRAIN overlap
was zero and no teacher inference ran. The declared configuration hash is
`64faf94cc233861cef96f85d1f4eaebd1b1d115fac7fa75b049e5c79e72a03d6`.
The exact 215-file source snapshot identity is
`b644718daa76c7706446478bf13b8d0da184ac9c9aff0d7475c418e36b93ad03`.

| Native policy, 28 development frames | STOP / stereo / Gaussian | Mean utility | Violations | Unknown |
| --- | --- | ---: | ---: | ---: |
| Gain-only joint selector | 4 / 0 / 24 | 0.1370 | 4 | 1 |
| Fixed Gaussian | 0 / 0 / 28 | 0.1783 | 4 | 1 |
| Offline oracle | 3 / 8 / 17 | 0.2031 | 0 | 0 |

Base stereo-depth MAE was 9.23719 mm and remained unchanged under the selector,
which selected no stereo action. Left/right PSNR changed from 19.58668/19.32260 dB
to 19.60788/19.33084 dB. These small render gains do not demonstrate useful
multi-action selection. This config remains an experimental diagnostic and is
not promoted to the default path.

### Confirmed findings

- **High: risk discrimination/transfer is inadequate.** Stereo risk was
  0.5237–0.6527 across development candidates, so the unchanged 0.51 cutoff
  rejected every stereo action. Predicted mean risk was 0.5912 versus observed
  harm rate 0.2667; Brier error was 0.2979. In last logged TRAIN snapshots,
  observed stereo harm was 35/59 (0.5932), close to the predicted mean 0.6029.
  These data show a harm-rate shift and predictions close to the TRAIN prior.
  They do not establish whether feature insufficiency, sample scarcity or
  changing label generators caused it.
- **High: gain ordering still misses stereo opportunities.** Nine known safe
  stereo-best regions had positive predicted stereo utility, but predicted
  Gaussian utility exceeded stereo in all nine. Lowering risk alone therefore
  cannot fix their gain ordering. Offline true-safe ranking counts three of nine
  correct only because it masks genuinely harmful Gaussian alternatives using
  GT; deployment does not have that information.
- **Medium: harm labels reflect real local regressions.** On last TRAIN
  snapshots, stereo violated local depth RMSE 24 times, local depth MAE 20 times
  and local RGB 16 times. Development counts were 5, 3 and 3. Counts overlap.
  Coverage was not a cause. Gaussian's five harmful development regions all
  violated local RGB preservation; one also violated right PSNR and one outside
  RGB preservation. These labels must not be discounted merely to select more
  repairs.
- **Medium: native run differences confound cross-run attribution.** The old
  head replayed on the new frozen artifact achieved utility 0.1519 versus the
  new head's 0.1370, with four violations each. Thus even a paired comparison of
  decisions does not support promoting the revised selector. Replayed timings
  are not a fresh benchmark.

The next discriminating work is a TRAIN outcome table measured under one pinned
final model/action artifact, with separate internal TRAIN fitting and calibration
roles and enough independent examples to assess conditional gain/risk. That table
has not been collected here. Merely tuning the stereo cutoff, adding an action
bonus or requiring all three actions would not resolve the measured failure.

### Validation actually run

- 51 focused selector/config/utility/ownership/resume tests passed. After final
  type fixes, the three gain-only ownership/resume/unknown-label cases passed
  again. Four CPU historical-hash/ranking checks also passed.
- Ruff lint/format, mypy on six affected source files, and Git whitespace checks
  passed. Initial mypy errors were corrected and rechecked.
- Two native 168-update pilots, asset preflight, dedicated TRAIN action-cost
  calibration, native development comparisons and read-only head replay ran.
- Two preliminary local test invocations hit their 90-second limit; the focused
  runtime/GPU invocations above completed. Full CPU backward regression tests
  and full CI were not run.
- No full-cohort training, final dataset_8/9 evaluation, teacher inference,
  dependency installation or changes to dataset roles occurred.

**Verdict:** engineering paths execute; selector effectiveness remains NO-GO.
Preserve both negative pilots rather than promote either experimental config.
