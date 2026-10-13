# Fixed-feature selector fit diagnostic

This head-only diagnostic follows the user-approved joint experiment. It does
not change the student, repairs, masks, data roles, loss implementation or active
router. It does not certify either research gate.

## Inputs and fitting boundary

`reliable_endo_gs.cli.selector_probe` reads an existing completed pilot directory:
`training/final.pt`, `training/train.jsonl`, and the matching
`development/result.json` / `development/samples.jsonl`.

The checkpoint bytes, config, training manifest and target-mask identity must
match the development result. TRAIN and development frame IDs must be disjoint,
and only original SCARED folds 1–7 are admitted. The last logged TRAIN snapshot
per frame is retained; unknown labels are excluded without falling back to an
older snapshot. Every observed utility/local/harm triple is checked against
`metric_targets`. Existing TRAIN-only normalization is retained.

The diagnostic resets the two existing selector MLPs and fits the unchanged
objective, gradient balancer, Adam learning rate and per-network clipping on a
fixed full TRAIN batch. The step count is declared before fitting. Development
does not fit normalization, weights or stopping thresholds. Outputs must be in
a new directory outside the repository. Source joint checkpoints are immutable.

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  "$RELIABLE_ENDO_RUNTIME_PYTHON" -m reliable_endo_gs.cli.selector_probe \
  --pilot-dir "$RELIABLE_ENDO_PILOT_DIR" \
  --output-dir "$RELIABLE_ENDO_SELECTOR_PROBE_OUTPUT" \
  --device cuda:0 --steps 1000 --seed 0
```

## Interpretation

TRAIN fixed rows originate from different pre-update model/action snapshots.
They are not newly collected outcomes of one final frozen model. Full-batch
fitting also increases region exposure compared with the online joint run, so
success cannot isolate label stationarity from optimization exposure.

True-safe gain ranking is reported to isolate gain learning from harm prediction;
it uses offline labels and is not a deployment policy. Separately, policy replay
uses only predicted gains/risks, original validity and original remaining budgets
before consulting outcomes. Unknown GT regions stay eligible. The original head
must reproduce the original logged action counts, violations, unknown count and
mean utility before the new head is evaluated. Reused costs/timings do not measure
the new head's latency or FPS.

## Recorded 2026-10-09 diagnostic

The approved 1000-step GPU-0 fit completed on 60 observed TRAIN regions from 52
frames. The full original worklist contained 56 TRAIN frames; four final-snapshot
regions lacked GT labels. Development fit metrics used 29 known regions from 27
frames. Policy replay retained all 28 development frames, including unknown GT.

| Quantity | Original joint head | Fixed-table fit |
| --- | ---: | ---: |
| TRAIN stereo gain MAE | 0.2242 | 0.00263 |
| TRAIN stereo winners ranked correctly | 0/17 | 17/17 |
| Development stereo gain MAE | 0.1506 | 0.2018 |
| Development stereo harm Brier | 0.2502 | 0.3710 |
| Replayed STOP / stereo / Gaussian | 3 / 1 / 24 | 3 / 5 / 20 |
| Replayed mean utility, known outcomes | 0.1456 | 0.1717 |
| Replayed metric violations / unknown interventions | 4 / 1 | 4 / 1 |

The current architecture/objective can fit these TRAIN labels; this contradicts
a claim that the stereo action is inherently unlearnable from this table. The
development error increase is evidence of poor transfer despite better selected
utility on this small cohort. Overfitting, label-generator snapshot differences
and cohort shift remain plausible contributors. No independent final-test claim
or production-head promotion follows from this result.

The next discriminating experiment should collect TRAIN/development outcomes
under one frozen model/action checkpoint, separate TRAIN fit/validation/calibration
roles, and compare controlled optimization exposure and regularization. Retain
the joint run and this negative transfer evidence.
