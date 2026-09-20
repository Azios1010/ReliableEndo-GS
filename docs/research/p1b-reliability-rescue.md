# P1b — Reliability Signal Rescue

Status: **PREREGISTERED DEVELOPMENT-ONLY STUDY**

P1b is a new study.  It does not revise, overwrite, or reinterpret historical
P1.  Historical U1, U2, corrected U3, and U4 remain controls exactly as
measured.  P1b asks whether derived signals from the same information sources
generalize across development sequences without GT at inference time.

## Frozen source and data boundary

- Stage-2 output path: `GPS-GS_stage2_deterministic_v3_best_step003000.pth`
- Checkpoint SHA-256: `32D42DE10A4C683C31FB4B3E0DE831F900E11197CDF1769B18CF3BBAFA5EE16F`
- Pinned Endo-E2E-GS: `186fa2b4a2159b28393492f6df1aa444b54391a8`
- Fit sequences: `dataset_1/keyframe_1`, `dataset_2/keyframe_1`,
  `dataset_3/keyframe_1`
- Holdouts: `dataset_7/keyframe_2`, `dataset_4/keyframe_4`
- Final sequences are forbidden and are rejected by the runner.

The original d3 prediction remains the Stage-2 output.  The runner compares
the original forward path with the diagnostic iteration path and records the
maximum absolute difference.  P1b never changes Gaussian parameters, opacity,
or renderer inputs; renderer invocation count is recorded as zero.

## Independent channels

P1b keeps three evidence channels separate:

1. model dynamics: ordered recurrent trajectory and convergence structure;
2. geometry consistency: corrected positive-magnitude LR correspondence;
3. observation consistency: direct bidirectional and robust photometric
   agreement.

No scalar fusion, weighted sum, MLP, calibration, `sigma_d`, `sigma_D`,
`Sigma_geo`, Gaussian covariance mutation, or opacity mutation is authorized.

### Dynamics candidates

With `v1=d2-d1` and `v2=d3-d2`, the preregistered maps are:

```text
U1 historical              |v2|
U1b path                   |v1|+|v2|
U1b decay                  |v2|/(|v1|+eps), direction unknown
U1b decay deviation        |log((|v2|+eps)/(|v1|+eps))|
U1b acceleration           |v2-v1|
U1b directional disagreement
                           |v2-v1|/(|v1|+|v2|+eps)
U1b directional agreement  1 - directional disagreement
U2 historical              population std(d1,d2,d3)
U2b normalized dispersion  std(d1,d2,d3)/(|d3|+eps)
U2b path efficiency        |d3-d1|/(|v1|+|v2|+eps)
U2b oscillation             1[v1*v2<0]
```

The convergence-state vector remains a vector.  It is not fused for P1b.

### Geometry candidates

The corrected U3 control uses the audited swapped-output positive-magnitude
semantics.  P1b adds:

- strict finite/in-view/bilinear-support validity;
- relative residual
  `|dL-dR|/(|dL|+|dR|+eps)`;
- a fixed GT-free visibility mask based on unique rounded target columns and
  local monotonic order.

Coverage and residual quality are reported separately.  Excluded pixels are
not assigned a large uncertainty value.

### Observation candidates

U4 controls remain raw RGB L1 from the left-to-right warp.  U4b evaluates:

- bidirectional mean, maximum, and left/right asymmetry on intersection support;
- absolute left/right residual disagreement on intersection support;
- fixed 5x5 local per-channel normalized intensity residual;
- fixed central finite-difference gradient residual;
- fixed 5x5 SSIM-like structural residual;
- the raw residual on U3b visibility support;
- the raw residual excluding a fixed high-intensity/low-saturation probable
  specularity mask.

The fixed heuristic uses normalized-RGB intensity `>=0.95` and saturation
`<=0.20`; these thresholds are not selected on d7/d4.

## Evaluation and decision rule

Every candidate is evaluated on identical samples at pixel, frame, and fixed
4x4-region levels.  The primary comparison level is region-level.  Metrics are
Spearman rank correlation, frame-rho distribution, risk-coverage at
100/90/80/70/50%, frame-cluster bootstrap, sequence-cluster bootstrap, and
coverage.  A fixed 1-pixel disparity error threshold is used for high-error
diagnostics.

A candidate is `PASS` only when both d7 and d4 have positive region-level
association and lower risk at 50% than at 100% retained coverage.  Partial
evidence is `MARGINAL`; otherwise it is `FAIL`.  This rule does not authorize
holdout-derived weights or a learned fusion.

After independent evaluation, pairwise rank correlations among the best
dynamics, geometry, and observation candidates are descriptive only.  A
failure-state table uses fit-side medians as fixed descriptive thresholds; it
does not train a classifier or produce a routing label.

## Shadow iteration study

Disabled by default.  If separately enabled, the runner requests at most
`K=6`, verifies that the first three predictions remain bitwise/numerically
unchanged, records extra forward cost, and uses later iterations only as
`EXTRA_COMPUTE_SIGNAL` diagnostics.

## Stereo-consensus observer

`L+R -> shared Z -> L_hat/R_hat` is design-only in P1b.  It is scientifically
justified as a possible P1c feasibility study, not as a P1b implementation or
result.  The design is recorded in
`docs/research/p1b-stereo-consensus-feasibility.md`.
