# Stationary gain and risk refit

## Status and scope

This is an isolated remediation of the authorized masked joint experiment. It
does not establish Gate I, Gate II, final test performance, or a latency claim.
The student and both learned repairs retain the exact selected joint checkpoint.
Future joint training remains possible; any model/action update invalidates the
refit and requires new outcomes and calibration.

The previous gain-only optimization pilot did not improve deployment utility.
It selected STOP/STEREO/GAUSSIAN = 4/0/24 on 28 development frames, with mean
physical utility 0.137004, four metric violations, and one unknown intervention.
Its uncalibrated stereo risk exceeded 0.51 everywhere on development. Observed
stereo harm was about 59% on the last moving TRAIN snapshots and 27% on
development. Those different snapshots were not stationary training labels for
the final model/actions. Development evidence diagnosed the failure; it is not
used to fit this remediation.

## Implemented protocol

1. Pin the source checkpoint by SHA-256. Validate the original manifests, role
   hash, native FP32 grid, deployment ROI index and dedicated TRAIN cost artifact.
2. Before reading GT, partition the existing 56-frame TRAIN worklist by whole
   keyframe, ordered by `SHA256(seed:keyframe)`, with seed 0. Fit/validation/
   calibration receive 32/8/16 frames from 8/2/4 keyframes. Original dataset
   roles and manifests are unchanged; original SCARED 8–9 remain untouched.
3. Execute the canonical STOP, stereo repair and Gaussian repair on the same
   frozen model/action artifact. Compute deployment features before loading
   separate offline GT. Persist full outcomes, including unknown labels. Unknown
   regions are excluded from supervision, never assigned a safe label.
4. Refit the existing small gain and harm MLPs using keyframe-balanced weights.
   Feature normalization and shared gain scales use FIT only. Keep signed gain,
   STOP=0, local auxiliary loss, global utility and safe-pair ranking. Harm uses
   the existing binary guard event and proper BCE without class reweighting.
5. Use TRAIN validation to select the gain encoder and risk encoder independently.
   AdamW, gradient clipping and bounded early stopping limit fitting. Local gain
   remains an auxiliary target and is not added again to decision utility.
6. On the separate TRAIN calibration role, fit a regularized monotone affine
   gain correction and the canonical bounded Platt risk calibrator. No risk
   cutoff or action quota is fitted. Each calibrated action requires at least
   eight known regions, with safe and harmful labels each represented by two
   distinct keyframes. Unsupported actions abstain.
7. Save a separate immutable selector artifact bound to model/action state,
   source, config, partition, masks and outcome hashes. Never rewrite the joint
   checkpoint or its optimizer. Deployment rejects a changed model/action state.
8. Run GT-free selected inference, then native original/refit evaluation on the
   same 28 development IDs with the same original model/actions and cost artifact.
   Decisions precede GT loading. Check all scientific weights remain unchanged.

The calibration sample is small. Calibration support and an improved proper
loss do not guarantee development safety or distribution-shift robustness.
The hash split happens to place both validation keyframes in dataset_6; this
was determined before labels and limits the strength of model selection evidence.
Development shares keyframes with TRAIN and is a diagnostic, not final test.

## Entry point

```bash
python -m reliable_endo_gs.cli.selector_refit --help
```

The CLI requires explicit external data, original TRAIN and development
manifests, roles, bounded worklist, target mask index, pinned checkpoint digest,
TRAIN cost artifact, refit config and a new output directory. Use
`--preflight-only` for deployment-file checks without loading GT or fitting.
No teacher inference, cache generation, joint retraining, dataset copying or
dependency installation is part of this path.

## Files changed in this remediation

| File | Change |
| --- | --- |
| `configs/routing/selector_refit.yaml` | Bounded declared fit/stopping/calibration policy |
| `src/reliable_endo_gs/config/selector_refit.py` | Config validation and sealed hash partition |
| `src/reliable_endo_gs/routing/balanced.py` | Optional keyframe row weights; default objective preserved |
| `src/reliable_endo_gs/routing/calibrated_selector.py` | Bound gain/risk calibration, support checks and abstention |
| `src/reliable_endo_gs/training/selector_outcomes.py` | Frozen shared interventions and canonical outcome reader |
| `src/reliable_endo_gs/training/selector_fitting.py` | Independent head selection and separate TRAIN calibration |
| `src/reliable_endo_gs/training/selector_refit.py` | Preflight, frozen collection, refit, smoke and native comparison |
| `src/reliable_endo_gs/training/student_learning.py` | Attach refit and reject changed action state; diagnostic support flags |
| `src/reliable_endo_gs/evaluation/intervention_selection.py` | Optional refit evaluation with immutable-state checks |
| `src/reliable_endo_gs/cli/selector_refit.py` | Thin first-party CLI |
| `tests/routing/test_selector_refit.py` | Six focused contract, fitting and calibration tests |
| `docs/selector_gain_risk_refit.md` | Protocol, negative evidence, identities and reproduction |

## Validation evidence

### Native results: NO-GO for selector promotion

Run: `/home/bkai/.local/share/reliableendo-gs/runs/selector-refit-gpu0-cb4fXMZH`.
Exit code 0. GPU 0, RTX 5070 Ti, PyTorch 2.7.1+cu128, CUDA 12.8, synchronous
CUDA debugging. A source snapshot and the complete console log are retained.

| Policy on the same 28 development frames | STOP / stereo / Gaussian | Mean physical utility | Metric violations | Unknown interventions |
| --- | --- | --- | --- | --- |
| Original joint selector | 4 / 0 / 24 | 0.137004 | 4 | 1 |
| Stationary refit + TRAIN calibration | 1 / 0 / 27 | 0.168954 | 4 | 1 |
| Fixed Gaussian | 0 / 0 / 28 | 0.178292 | 4 | 1 |
| Offline oracle diagnostic | 3 / 8 / 17 | 0.203065 | 0 | 0 |

Utility increases by about 23.3% over the original selector, but remains below
fixed Gaussian; the four violations persist. This does not establish a useful
learned action selector. The new artifact is an isolated diagnostic, not an
approved default deployment artifact.

On the 30 known development regions:

| Head | Original gain MAE | Refit gain MAE | Original risk Brier | Refit calibrated risk Brier |
| --- | --- | --- | --- | --- |
| Stereo | 0.111187 | 0.124649 | 0.297936 | 0.326219 |
| Gaussian | 0.120101 | 0.116788 | 0.152367 | 0.141712 |

Stereo gain does not improve. There are nine regions where safe stereo has the
best positive measured utility; both the raw refit and its calibrated version
predict Gaussian gain above stereo in all nine. The gain ranking error is
independent of risk rejection.

Raw refit stereo risk has development Brier 0.222037 and rejects none of these
nine regions at 0.51. TRAIN calibration changes it to Brier 0.326219 and rejects
all nine. Mean calibrated stereo risk is 0.619343 versus observed development
harm 0.266667. This isolates an additional calibration generalization failure.
It does not prove stereo cannot be selected or that the action cannot help.

Known stationary TRAIN regions have stereo harm counts of 5/30 in FIT, 4/15 in
validation and 8/14 in calibration. Calibration is concentrated in four
keyframes; its harm distribution differs from FIT and development. On that
calibration role, stereo Brier improves from 0.306155 to 0.238162, and Gaussian
from 0.169197 to 0.111327. An improved calibration-role loss is therefore
insufficient evidence of development calibration. The small, heterogeneous
sample and gain ranking error require further evidence; lowering the risk
cutoff or multiplying all gains cannot resolve both failures.

Both independently selected encoder snapshots happen to be step 25. Early
stopping finishes at step 175 and restores those snapshots. At step 25,
validation gain objective / BCE are 0.393808 / 0.581882; by step 175 they rise
to about 0.649829 / 1.849547. Continuing to fit this small set overfits validation.
These observations support collecting more diverse stationary TRAIN outcomes
and testing action-specific feature sufficiency before another long run. They
do not establish that available deployment features are mathematically
insufficient.

Selected deployment smoke uses `dataset_3/keyframe_1/frame_data000097`, chooses
STOP, has finite left/right RGB, no budget overrun, and no GT/teacher input.
Development base/selected stereo depth MAE remain 9.237194 mm because the
selector chooses no stereo action. Selected left/right PSNR change from
19.607877/19.330843 to 19.610301/19.332803 dB. All model, repair and selector
weights remain unchanged during native evaluation; train/development overlap
is zero. The two evaluation passes take 65.47 and 45.23 seconds, with peak
allocated VRAM 0.686 and 0.689 GiB. These are diagnostic passes including oracle
work, not comparable FPS measurements.

### Identities

- Original checkpoint SHA-256:
  `01ee7a0d065127bb50f996e1d9247610e7826c6da54c9bbdcddc3a02679c32c3`.
- Original training source SHA-256:
  `b644718daa76c7706446478bf13b8d0da184ac9c9aff0d7475c418e36b93ad03`.
- Current first-party source SHA-256:
  `5ef8ac5ec004e27bd356586f1f68fcf67a0193d86b46429348c3f66036a13f46`.
- Refit artifact SHA-256:
  `c8e0606fab1bc6a25de2f180df7f22d57ddf83c7bf38e71780d855605564b210`.
- Internal TRAIN partition SHA-256:
  `ef56ee3380ff120e08c15570f08b732b6224b75caf29783ccc2b19359e2d7442`.
- Refit config hash:
  `059d44399fa1e9672c1e3d3fd6bc2f5d650d2d7328dc781224ad92868e9b719b`.
- Cost artifact SHA-256:
  `0f34afc81746d69b9cb20e5246fb1daead7f0c3838285022ce24ad87c81acd4b`.

Other manifest, mask and outcome hashes are in `pilot/result.json`,
`pilot/train_outcomes/result.json` and `pilot/selector.pt`. The paired diagnostics
are `pilot/paired_head_diagnostics.json` and
`pilot/raw_vs_calibration_diagnostic.json`. Generated outcomes and medical
inputs remain outside the repository.

### Checks executed

- Six new selector refit tests pass, including canonical labels, unknown-label
  exclusion, immutable collection, deployment feature construction before GT,
  whole-keyframe splits, separate normalization/stopping, bounded calibration,
  abstention and artifact binding/round-trip.
- Sixty focused regression tests pass across balanced selector, metric joint
  learning and fresh selector optimization. GPU 0 runs the gradient tests;
  test runner uses existing dev pytest with runtime PyTorch. No package install.
- Ruff check/format checks, mypy on nine affected source files, CLI help, shell
  syntax, Git whitespace and forbidden-import/path checks.
- Real preflight, 56 frozen TRAIN sample collections, selector fitting/calibration,
  one GT-free deployment smoke and two native 28-frame development passes.
- A preliminary test worker was stopped to recover its output capture; its
  completed replacement is recorded. A preliminary native invocation was
  stopped after its first TRAIN outcome to fix immutable-mapping JSON
  serialization; that partial collection is unused, and its exit 143 log
  remains alongside the final successful run. The final run independently
  collects all 56 TRAIN samples under its recorded source identity.
- No full CI suite, teacher inference, cache generation, joint retraining,
  long training, final test on datasets 8–9, or production latency benchmark.

Reproduce locally with the external helper:

```bash
bash /tmp/run-reliable-endo-selector-refit-gpu0.sh --preflight-only
bash /tmp/run-reliable-endo-selector-refit-gpu0.sh
```

The helper creates a new output directory on each invocation. The completed
run retains `command.json`, the helper, check logs and the exact source snapshot.
Pinned checkpoint and cost-artifact copies are saved as `input_checkpoint.pt`
and `input_cost_calibration.pt` outside the repository so their `/tmp` originals
are not the sole surviving copies. External datasets and masks are not copied.
Existing successful outputs and checkpoints are preserved.
