# Plan 04 — Phase I Uncertainty Proxies and Oracle

## Status

PLANNED

## Research stage

Phase I

## Recommended executor

Sol. This milestone defines the first falsifiable uncertainty experiment, leakage-safe oracle signal, calibration evaluation, and go/no-go decision for learned uncertainty.

## Estimated implementation risk

MEDIUM. Signals are accessible only if the pinned backbone exposes them correctly, and an apparently strong proxy can reflect invalid pixels, depth scale, or test leakage.

## Objective

Measure whether stereo-derived cheap proxies and a ground-truth-derived oracle uncertainty signal predict downstream disparity/depth error, and define covariance-headroom inputs before investing in a learned head.

## Scientific question

Does stereo-derived uncertainty contain calibrated or rank-useful information about disparity, metric depth, and eventual center error beyond trivial depth/range effects?

## Why this milestone exists

Learned uncertainty and probabilistic Gaussians are unjustified if even oracle/proxy evidence has no useful signal. Running this milestone first exposes low-cost pivots and fixes evaluation/calibration protocols before model capacity is added.

## Prerequisites

- Plan 03 baseline artifact accepted.
- Frozen training/validation/test sequences and metric definitions.
- Verified availability map for upstream intermediate evidence.

## Inputs

- Immutable baseline predictions and optional disparity iterations.
- SCARED ground-truth disparity/depth and masks where available.
- Left/right images, calibration, baseline renders, and split identity.

## Outputs

- Versioned raw proxy records and oracle uncertainty targets.
- Comparative calibration/detection report by sequence and depth bin.
- A frozen protocol describing the oracle covariance-headroom experiment for Plans 06-07, without implementing covariance here.
- Decision: approve Plan 05, select a calibrated proxy fallback, or stop uncertainty-dependent Phase I.

## Scope

- Implement only evidence actually supported by the pinned source: final recurrent update magnitude; disagreement among final disparity iterations; left-right consistency; correlation entropy only if exposed with clear meaning; photometric or feature-warp residual.
- Define oracle uncertainty from valid absolute disparity/depth residuals without exposing ground truth at inference.
- Fit any proxy scale/calibration on validation sequences only.
- Evaluate raw ranking separately from probabilistic calibration.
- Stratify by depth, texture, occlusion/specularity masks where defined, and validity.
- Specify what `sigma_d` records Plan 06 will consume and how Plan 07 will compare oracle covariance to opacity/fixed filtering after propagation exists.

## Non-goals

- No learned uncertainty head, covariance propagation, Gaussian covariance, renderer change, action, oracle routing, or test-set calibration.
- Do not require RAFT correlation features absent from upstream.
- Oracle error is diagnostic/supervisory only and cannot enter deployable inference.

## Files expected to be created

- `src/reliable_endo_gs/uncertainty/proxies.py`
- `src/reliable_endo_gs/uncertainty/oracle.py`
- `src/reliable_endo_gs/uncertainty/calibration_metrics.py`
- `src/reliable_endo_gs/evaluation/uncertainty.py`
- `scripts/evaluate_uncertainty_proxies.py`
- `configs/uncertainty/proxies.yaml`
- `configs/experiment/p1a_uncertainty_proxy.yaml`
- `configs/experiment/p1c_covariance_oracle.yaml` as a future experiment specification consumed only after Plan 07.
- `tests/uncertainty/test_proxies.py`
- `tests/uncertainty/test_oracle_targets.py`
- `tests/evaluation/test_uncertainty_metrics.py`

## Files expected to be modified

- Baseline adapter only to expose verified optional iterations/evidence through `StereoPrediction`, without changing predictions.
- Contract schema only through a reviewed compatible optional-field extension if real evidence requires it.
- Runtime/artifact manifest support for uncertainty evidence identities.

## Interfaces and contracts

Each proxy is a named, versioned estimator mapping pre-action baseline evidence to a disparity uncertainty score and valid mask. Calibrators are fit objects with training split identity. Oracle target generation requires ground truth explicitly and produces a separate diagnostic record that deployable loaders reject.

Raw scores, calibrated `sigma_d`, and oracle targets remain distinct. Evaluation accepts a prediction, target, and masks and returns metrics with valid counts, bin definitions, and aggregation policy.

## Scientific formulation

Proxy definitions are supporting baselines, not proposed contributions. Ground-truth oracle error is diagnostic. Metric implementations belong in `evaluation/uncertainty.py`; proxy calculations belong in `uncertainty/proxies.py`. AUSE uses a clearly defined sparsification/error curve, AUROC uses a validation-frozen high-error threshold, and NLL is reported only when a score has a valid probabilistic scale.

## Numerical and coordinate conventions

Compare disparity residuals in verified pixels and depth residuals in verified metric units. Apply masks before aggregation. Handle zero-valid batches explicitly. Normalize iteration/update magnitudes only by a validation-defined rule. Never turn invalid disparity into a large “uncertainty” target without labeling it separately.

## Configuration changes

Define evidence source, iteration window, warp method, calibration method, validation-frozen bin edges/high-error definition, masks, and aggregation. Experiment `p1a` compares proxies; `p1c` records the later oracle covariance comparison but cannot execute before Plans 06-07.

## Tests

### Unit

- Known iteration disagreement/update examples.
- Left-right warp sign and out-of-view masking.
- Oracle residual/mask behavior, NLL/AUSE/AUROC/correlation, coverage, bins, and empty cases.

### Contract

- Proxy output shape/device/mask and explicit raw-versus-calibrated identity.
- Ground-truth oracle records cannot be loaded as inference features.

### Integration

- Frozen baseline artifact -> proxy extraction -> validation calibration -> report.

### Smoke

- Small validation subset produces all available proxies and a traceable evidence artifact.

### Regression

- Fixed synthetic residual rankings and metric values.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/uncertainty tests/evaluation
pytest -q
python -m compileall src
python scripts/evaluate_uncertainty_proxies.py --config configs/experiment/p1a_uncertainty_proxy.yaml
git diff --check
```

## Experiments

1. Compare all available proxies and oracle targets on validation sequences.
2. Measure disparity- and depth-bin behavior and failure-mode slices.
3. Freeze the later oracle-covariance headroom protocol: oracle `sigma_d` source, masks, comparison variants, metrics, and no-test tuning.

## Metrics

- NLL when probabilistically meaningful.
- AUSE and sparsification error.
- AUROC for validation-defined high error.
- Spearman/Pearson correlation with absolute error, reported with caveats.
- Empirical interval coverage/calibration error for calibrated scores.
- Per-depth-bin and per-sequence results with valid counts.

## Acceptance criteria

- At least one deployable proxy or calibratable signal shows meaningful, repeatable error ranking/coverage beyond trivial controls on validation data, using criteria frozen before test evaluation.
- Oracle uncertainty establishes a clear upper-bound protocol and the exact inputs needed by Plans 06-07.
- Learned-head decision and proxy fallback are documented before Plan 05.
- No ground-truth oracle information leaks into inference or test tuning.

## Failure conditions

- Neither oracle nor available stereo evidence predicts valid error beyond trivial range/validity effects.
- Results disappear under sequence grouping or depth stratification.
- Required evidence needs an upstream scientific modification.

## Pivot / rollback path

If learned investment is unsupported but a proxy works, skip Plan 05 and calibrate/freeze the best proxy for Plan 06. If only oracle works, use it solely to test representation headroom and limit deployable claims. If oracle covariance later lacks headroom, stop covariance work at Gate I rather than forcing a head.

## Artifact/provenance requirements

Record baseline artifact ID, dataset/split hashes, sample IDs, proxy versions/configs, calibration split, mask/bin schemas, metric code revision, seeds, and raw per-sequence outputs. Oracle records carry a non-deployable marker.

## Completion checklist

- [ ] Every available proxy is defined from pre-action evidence.
- [ ] Oracle targets and leakage controls are explicit.
- [ ] Calibration/detection metrics and depth bins are validated.
- [ ] Covariance-headroom protocol is frozen conceptually.
- [ ] Learned-path/proxy/stop decision is recorded.
- [ ] No covariance or learned head was implemented.

## Handoff to next plan

Plan 05 may assume a fixed uncertainty target/evaluation protocol only if learned prediction was approved. Plan 06 may instead consume the selected calibrated proxy; neither plan may use test sequences to choose calibration.
