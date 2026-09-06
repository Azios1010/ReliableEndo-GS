# Revised Phase I Preregistration: ProbStereo-EndoGS

Status: **DEFINED / NOT IMPLEMENTED**

This document is a protocol definition only. It authorizes no code, training,
checkpoint creation, or RiskRoute work.

## Research question

Can a stereo-to-Gaussian reconstruction system represent where its geometry is
unreliable in a way that remains meaningful after Gaussian rendering and across
views?

The working scientific premise is narrower than the original roadmap:

> Stereo uncertainty, prediction error, and intervention utility are distinct
> quantities. Phase I tests whether uncertainty propagated from disparity into
> 3D Gaussian geometry is more predictive of reconstruction reliability than
> raw disparity uncertainty alone.

This is a hypothesis, not an established result.

## Gate-0 implications

Gate 0 moderately supported an uncertainty–repairability mismatch, observed a
weak same-view Gaussian-side compensation signal, and did not support action
specialization. Therefore Phase I focuses on uncertainty representation and
geometry propagation. Repair actions, oracle routing labels, routers, and
budget allocation are out of scope.

The strong claim that cross-view error systematically exposes Gaussian
compensation is not carried forward. Cross-view error begins as an independent
evaluation signal.

## Uncertainty source comparison

| signal | definition | units | inference availability | GT required |
| --- | --- | --- | --- | --- |
| iteration update magnitude | `abs(d_K - d_(K-1))` | disparity pixels | yes | no |
| iteration disagreement | standard deviation of valid `d_k` | disparity pixels | yes | no |
| left-right consistency | `abs(d_L - warp(d_R, d_L))` | disparity pixels | conditional on right prediction | no |
| photometric residual | channel mean of left/right warp residual | normalized RGB | yes | no |
| correlation entropy | entropy of normalized correlation distribution | nats | only if exact tensor is exposed | no |

Correlation entropy is not added unless the pinned upstream tensor contract is
verified. Ground-truth error may be used only as an evaluation target, never as
an inference proxy.

## Calibration

The primary calibration object is a risk–coverage curve: retain the lowest
uncertainty regions at increasing coverage and measure empirical error. Secondary
metrics are Spearman rank correlation, high-error AUROC/AUPRC, and a quantile
reliability curve.

The high-error event is fixed as absolute error above the 90th percentile of the
development calibration split. No threshold is tuned on the final untouched
sequences. ECE is used only if the representation defines a probability; NLL is
used only if it defines an explicit predictive distribution.

All intervals are sequence- or frame-grouped. Pooled regions are not treated as
independent subjects.

## Geometry propagation

For valid positive disparity:

```text
D = fx B / d
r = K^-1 [u, v, 1]^T
mu = D r
J_d = -(fx B / d^2) r
Sigma_geo = J_d sigma_d^2 J_d^T
sigma_D = |fx B / d^2| sigma_d
```

The first geometry gate compares `sigma_d`, `sigma_D`, `trace(Sigma_geo)`, and
the largest eigenvalue of `Sigma_geo` against disparity error, depth error, 3D
center error, same-view rendering error, and cross-view rendering error.

Invalid or non-positive disparity is excluded before depth or covariance
calculation; no arbitrary clipping is introduced to make a metric finite.

## Probabilistic Gaussian hypotheses

Surface/support covariance and center-location covariance remain distinct:

```text
Sigma_surf = R(q) diag(s^2) R(q)^T
Sigma_geo  = center-location uncertainty from disparity propagation
```

The exploratory combination is:

```text
Sigma_eff = Sigma_surf + Sigma_geo
```

It is not declared to be the final renderer rule. The protocol compares:

1. covariance enlargement only;
2. covariance enlargement plus the determinant-ratio amplitude candidate;
3. an alternative mass-preserving rule only if it is mathematically derived
   and documented before comparison.

The candidate determinant correction is:

```text
kappa = sqrt(det(Sigma_surf) / det(Sigma_eff))
```

No candidate is selected before Gate I-C, and every candidate is tested for
blur, coverage, geometry, and reconstruction trade-offs.

## Sequential ablation ladder

| stage | definition | entry condition |
| --- | --- | --- |
| P0 | frozen deterministic baseline | already complete |
| P1 | proxy measurement, no rendering change | protocol checks |
| P2 | proxy calibration | Gate I-A evidence available |
| P3 | analytic `Sigma_geo`, no renderer change | Gate I-B measurement |
| P4 | uncertainty-aware Gaussian representation | Gate I-B passes |
| P5 | opacity/mass candidate comparison | Gate I-C branch only |
| P6 | cross-view supervision | Gate I-D passes; otherwise evaluation-only |

A failed gate stops its branch. Negative evidence is retained and does not
authorize a compensating redesign within the same run.

## Gates

### Gate I-A — uncertainty signal

Pass only if an inference-available signal ranks held-out development geometry
error better than chance and better than trivial baselines with sequence-aware
support.

### Gate I-B — geometry propagation

Pass only if geometry-space quantities derived from `Sigma_geo` add reliability
information beyond raw `sigma_d` without final-sequence tuning.

### Gate I-C — probabilistic Gaussian

Pass only if adding uncertainty to the Gaussian representation improves a
predeclared reliability or reconstruction criterion without unacceptable blur or
geometry degradation.

### Gate I-D — cross-view

Pass only if cross-view evidence adds reproducible information beyond same-view
metrics, supported by sequence-cluster bootstrap evidence without reversal on a
final sequence. Otherwise it remains evaluation-only.

## Data policy

Development uses the existing v3 training sequences:

- `dataset_1/keyframe_1`
- `dataset_2/keyframe_1`
- `dataset_3/keyframe_1`

The existing v3 validation sequences are development calibration/selection
cases, not pristine final tests:

- `dataset_7/keyframe_2`
- `dataset_4/keyframe_4`

The untouched final evaluation is preregistered as:

- `dataset_5/keyframe_1`
- `dataset_8/keyframe_2`

These candidates passed the existing integrity audit and were not part of the
v3 baseline selection. `dataset_9/keyframe_3` is excluded because the existing
integrity audit recorded zero-valid disparity frames. A new immutable manifest
containing the final sequences must be created and hashed before Phase-I
implementation; no data is processed by this definition task.

## Metrics and compute

Stereo metrics are EPE, 1px, and 3px. Geometry metrics are depth MAE/RMSE and
3D center error. Reconstruction metrics are RGB L1, SSIM, and audit-only PSNR.
Reliability metrics are rank correlation, risk–coverage, high-error AUROC/AUPRC,
and proper NLL only for an explicit predictive distribution. Runtime metrics are
latency, peak VRAM, and additional parameters.

Proxy and analytic gates are capped at two GPU-hours total. Any learned variant
is capped at three preregistered seeds and 10,000 updates per variant, and may
start only after its upstream gate passes. No routing compute budget is defined.

## Claim boundary and parked work

The intended Phase-I claim is that disparity uncertainty, propagated geometry
uncertainty, and reconstruction reliability can be empirically distinguished.
The contribution is not “add uncertainty to Gaussian splatting,” universal
SCARED generalization, clinical reliability, or a guaranteed cross-view
compensation law.

RiskRoute remains **PARKED** because Gate 0 found no action specialization and
no compute-aware routing advantage. It may be reconsidered only after future
evidence creates heterogeneous intervention regimes.
