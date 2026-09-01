# Plan 05 — Phase I Learned Uncertainty and Calibration

## Status

IMPLEMENTATION: COMPLETE
LOCAL SYNTHETIC/ANALYTIC VALIDATION: PASS
LEARNED PROVIDER SCIENTIFIC SELECTION: PENDING
REAL-DATA VALIDATION: PENDING
SCIENTIFIC ACCEPTANCE: PENDING

## Research stage

Phase I

## Recommended executor

Sol. Model-interface selection, probabilistic loss, calibration separation, checkpointing, and proxy-relative scientific interpretation require careful research judgment.

## Estimated implementation risk

MEDIUM/HIGH. The upstream feature boundary may be awkward, and a learned scale can inflate, miscalibrate, or appear strong through depth/test leakage.

## Objective

Implement and validate a lightweight Laplace disparity-uncertainty predictor, calibrate it on validation sequences, and decide whether it replaces or merely complements the best Plan 04 proxy.

## Scientific question

Does a learned and calibrated stereo uncertainty predictor provide better probabilistic calibration and useful error ranking than cheap proxy baselines?

## Why this milestone exists

Plan 04 first proves signal/headroom. This milestone adds capacity only if justified, while keeping raw prediction, calibration, and test evaluation separable so calibration gains cannot be mistaken for representation gains.

## Prerequisites

- Plan 04 explicitly approves learned uncertainty.
- Immutable baseline artifact, frozen split, target/mask definition, and proxy results.
- Verified upstream feature source that can be exposed without changing baseline disparity.

## Inputs

- Baseline hidden/correlation features if actually available, otherwise the approved minimal feature set.
- Ground-truth disparity and validity masks on training/validation sequences.
- Best proxy baselines and Plan 04 evaluation protocol.

## Outputs

- Raw uncertainty-head checkpoint and training record.
- Separately fitted calibration artifact.
- Calibrated `sigma_d` provider satisfying `StereoPrediction` semantics.
- Proxy-versus-learned report and decision for Plan 06.

## Scope

- Select the smallest head compatible with actual upstream features and keep baseline prediction behavior identifiable.
- Train a Laplace scale predictor with explicit stability/regularization diagnostics.
- Fit scalar temperature first; consider a monotonic method only if validation evidence shows systematic heteroscedastic miscalibration.
- Evaluate raw and calibrated predictions separately on untouched held-out sequences.
- Preserve the best cheap proxy as a required baseline and fallback.

## Non-goals

- No covariance, renderer change, cross-view loss, repair, router, test calibration, or claim that uncertainty prediction itself is novel.
- Do not require arbitrary RAFT internals or jointly train the head to optimize later rendering/router labels.

## Files expected to be created

- `src/reliable_endo_gs/uncertainty/laplace.py`
- `src/reliable_endo_gs/uncertainty/head.py`
- `src/reliable_endo_gs/uncertainty/calibration.py`
- `src/reliable_endo_gs/training/uncertainty.py`
- `scripts/train_uncertainty.py`
- `scripts/calibrate_uncertainty.py`
- `configs/uncertainty/laplace_head.yaml`
- `configs/training/uncertainty.yaml`
- `configs/experiment/p1b_uncertainty_calibration.yaml`
- `tests/uncertainty/test_laplace.py`
- `tests/uncertainty/test_calibration.py`
- `tests/integration/test_uncertainty_training_smoke.py`

## Files expected to be modified

- `src/reliable_endo_gs/baseline/adapter.py` only for an already-audited optional feature output.
- Artifact/manifest schema for uncertainty and calibration identities.
- `docs/contracts.md` only if the calibrated `sigma_d` parameterization needs a compatible clarification.

## Interfaces and contracts

The raw predictor maps verified pre-render stereo features to positive Laplace scale `b` and diagnostics. The calibrator maps raw `sigma_d` plus its own immutable parameters to calibrated `sigma_d'`. Training, calibration, and evaluation entry points require explicit split roles and reject test data for fitting.

## Scientific formulation

Owned by `uncertainty/laplace.py`:

```text
b_i = softplus(h_phi(z_i)) + epsilon
L_NLL,i = |d_i* - d_hat_i| / b_i + log(b_i)
sigma_d,i^2 = 2 b_i^2
```

Owned by `uncertainty/calibration.py`:

```text
sigma'_d,i = tau sigma_d,i
```

The Laplace likelihood and temperature scaling are standard prior formulas/supporting components, not proposed novelty. Tests use analytic residuals and verify gradients/positivity. The proposed program contribution remains downstream two-covariance semantics and reliability-to-action evidence.

## Numerical and coordinate conventions

`b`, `sigma_d`, and residuals share verified disparity units/resolution. `epsilon` is named/configured and tested, not a hidden clamp. Masks precede reduction; invalid pixels do not contribute. Track minimum/maximum scale, saturation, non-finite values, and uncertainty inflation by depth bin.

## Configuration changes

Define feature source, head size, softplus epsilon, optimizer/schedule, loss weights, masks, seed, checkpoint policy, calibration method, and validation selection metric. Configs identify baseline artifact and split hash explicitly.

## Tests

### Unit

- Laplace NLL values/gradients, positive scale, variance conversion, temperature scaling, masks, and empty cases.
- Checkpoint and calibration schema compatibility.

### Contract

- Calibrated output matches `sigma_d` shape/unit and remains distinct from raw `b`/score.
- Test split cannot be supplied to fit methods.

### Integration

- Baseline evidence -> head -> calibration -> uncertainty evaluation with a frozen baseline.

### Smoke

- Tiny CPU/GPU training-calibration cycle with deterministic seed and artifact output.

### Regression

- Synthetic residual calibration and fixed checkpoint inference summaries.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/uncertainty tests/integration
pytest -q
python -m compileall src
python scripts/train_uncertainty.py --config configs/experiment/p1b_uncertainty_calibration.yaml
git diff --check
```

## Experiments

1. Train the minimal head across required seeds on training sequences.
2. Fit calibration on validation sequences only.
3. Compare raw, calibrated, and proxy signals on calibration, ranking, depth bins, and runtime overhead.
4. Freeze the chosen uncertainty provider before Plan 06/test evaluation.

## Metrics

- Laplace NLL, AUSE, AUROC, empirical coverage, calibration error.
- Correlation and depth-bin/per-sequence behavior.
- Scale inflation/saturation and valid count.
- Head parameters, latency, and peak-memory overhead.

## Acceptance criteria

- Learned uncertainty is finite, stable, and calibration is fit without test data.
- It outperforms the strongest cheap proxy on predeclared primary uncertainty criteria with consistent sequence/depth behavior, or the milestone explicitly selects the proxy fallback.
- Raw and calibrated checkpoints/artifacts are independently identified.
- Baseline disparity behavior remains unchanged unless a separately declared fine-tuning experiment is approved.

## Failure conditions

- Learned output fails to beat the proxy, is materially miscalibrated after approved validation-only calibration, inflates scale, or requires invasive baseline changes.

## Pivot / rollback path

Retain the negative learned result and use the best calibrated Plan 04 proxy for Plan 06. Consider proxy distillation or monotonic calibration only as a separately justified ablation; do not force the learned head into the Phase I contribution.

## Artifact/provenance requirements

Record baseline artifact ID, feature schema, architecture/config hash, training/calibration split hashes, seeds, raw checkpoint hash, calibration parameters/hash, proxy comparator identities, environment/hardware, and full metrics.

## Completion checklist

- [ ] Learned path was approved by Plan 04.
- [ ] Minimal head and Laplace loss are tested.
- [ ] Calibration uses validation sequences only.
- [ ] Raw, calibrated, and proxy evaluations are separate.
- [ ] Provider selection or proxy pivot is recorded.
- [ ] Artifact identities and overhead are complete.

## Handoff to next plan

Plan 06 may assume exactly one frozen calibrated `sigma_d` provider—learned or proxy—with explicit units, masks, schema, and artifact identity. It may not retrain or recalibrate that provider.
