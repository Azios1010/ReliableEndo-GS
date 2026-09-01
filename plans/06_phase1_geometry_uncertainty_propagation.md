# Plan 06 — Phase I Geometry Uncertainty Propagation

## Status

IMPLEMENTATION COMPLETE

Local analytic/synthetic validation: PASS (focused Plan 06 geometry tests).
Real-data validation: PENDING.
Scientific acceptance: PENDING.

## Research stage

Phase I

## Recommended executor

Sol. Canonical geometry, Jacobians, frames, units, masks, rank structure, and stabilization require numerical-method rigor.

## Estimated implementation risk

MEDIUM. The formulas are standard, but disparity singularities and convention errors can produce plausible yet physically wrong covariances.

## Objective

Implement the single canonical path from disparity and calibrated disparity variance to metric depth, camera-frame center, and 3D center covariance, with analytic and finite-difference verification.

## Scientific question

Does the chosen uncertainty provider yield physically interpretable, numerically stable center uncertainty after verified stereo triangulation?

## Why this milestone exists

Plan 07 requires observation-derived covariance with known meaning. Centralizing geometry before renderer integration prevents duplicated formulas and separates transformation correctness from the hypothesis that covariance improves reconstruction.

## Prerequisites

- Plan 04 proxy fallback or Plan 05 learned/calibrated provider frozen.
- Plans 01-03 camera/disparity conventions and baseline artifact.
- Analytic camera contract tests from Plan 00.

## Inputs

- `StereoPrediction.disparity`, calibrated `sigma_d`, valid masks.
- Left-camera intrinsics, verified stereo baseline, pixel grid, transforms, and units.

## Outputs

- Canonical disparity/depth, backprojection, and covariance-propagation modules.
- Derived depth uncertainty and `cov_center`/`Sigma_geo` with diagnostics.
  `Sigma_geo` is propagated center-position covariance, not Gaussian support covariance.
- Numerical policy for invalid/near-zero disparity, PSD, rank, eigenvalues, and optional localization term.
- Geometry validation report and fixtures for Plan 07.

## Scope

- Implement batched disparity-to-depth and inverse conversion.
- Implement pixel-ray construction and camera-frame backprojection.
- Implement analytic disparity Jacobian and first-order covariance propagation.
- Keep the initial disparity-only covariance rank-1 except for an explicitly configured `epsilon I` stabilization.
- Treat pixel/localization covariance as an optional later term, disabled initially unless independently justified.
- Inspect eigenvalue/depth-bin distributions on validation data; freeze any cap from validation percentiles before held-out evaluation.
- Return masks and diagnostics rather than giant covariances for invalid disparity.

## Non-goals

- No surface covariance, renderer integration, learned uncertainty changes, repair, or novelty claim for triangulation/propagation.
- No silent arbitrary clipping or coordinate-frame conversion in consumers.

## Files expected to be created

- `src/reliable_endo_gs/geometry/__init__.py`
- `src/reliable_endo_gs/geometry/disparity.py`
- `src/reliable_endo_gs/geometry/backprojection.py`
- `src/reliable_endo_gs/geometry/covariance_propagation.py`
- `src/reliable_endo_gs/geometry/pixels.py`
- `configs/geometry/stereo_geometry.yaml`
- `tests/geometry/test_disparity.py`
- `tests/geometry/test_backprojection.py`
- `tests/geometry/test_covariance_propagation.py`
- `tests/geometry/test_analytic_cameras.py`
- `scripts/audit_geometry_uncertainty.py`

## Files expected to be modified

- `src/reliable_endo_gs/contracts/gaussian.py` to support verified optional `cov_center` semantics if not already complete.
- Phase I artifact diagnostics schema.
- `docs/contracts.md` with resolved conventions only.

## Interfaces and contracts

Functions accept explicit tensors and convention-bearing camera metadata, return values plus validity/diagnostics, preserve batch/spatial shapes, and never read global camera constants. `cov_center` is expressed in the same declared frame as Gaussian means. All consumers call these owners rather than repeating equations.

## Scientific formulation

Owned by `geometry/disparity.py` and `geometry/backprojection.py`:

```text
D = f_x B / d
r = K^-1 [u, v, 1]^T
mu = D r
```

Owned by `geometry/covariance_propagation.py`:

```text
J_d = -(f_x B / d^2) r
Sigma_geo = J_d sigma_d^2 J_d^T
```

An optional later extension is `J_uv Sigma_uv J_uv^T + epsilon I`. These are standard stereo/first-order propagation formulas and implementation transformations, not proposed novelty. The tests compare the analytic Jacobian with finite differences and verify covariance properties.

## Numerical and coordinate conventions

- Use verified pixel coordinates, focal length in pixels, baseline and depth in the same metric unit, and camera-frame rays.
- Mask non-finite, invalid, wrong-sign, and below-threshold disparity before division.
- Configure the near-zero threshold, epsilon, and optional eigenvalue cap; record affected counts.
- Test symmetry, PSD within tolerance, near-rank-1 structure, and the viewing-ray principal eigenvector.
- Mixed precision is not accepted until compared to a float64 reference on analytic cases.
- SCARED-C real-data calibration, units, rectification, and depth semantics remain unresolved;
  no real-data validation is claimed by this plan.

## Configuration changes

Define disparity validity threshold, output units, epsilon policy, optional pixel covariance, eigenvalue-cap method/value, and diagnostic bins in `configs/geometry/`. Values derived from validation carry the calibration split identity.

## Tests

### Unit

- Known disparity/depth and inverse pairs.
- Analytic backprojection at principal point/off-axis.
- Finite-difference Jacobian across safe disparities.
- Symmetry, PSD, rank/eigenvector, variance scaling, masks, NaN/Inf, and near-zero disparity.

### Contract

- Shapes, frames, units, `cov_center` semantics, and mask propagation.

### Integration

- Frozen prediction/uncertainty -> depth -> means -> covariance on a small SCARED batch.

### Smoke

- CPU analytic-camera pipeline and GPU batch path agree within declared tolerance.

### Regression

- Float64 analytic fixtures and validation-distribution summaries.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/geometry tests/contracts
pytest -q
python -m compileall src
python scripts/audit_geometry_uncertainty.py --config configs/geometry/stereo_geometry.yaml
git diff --check
```

## Experiments

Audit `sigma_D` and `Sigma_geo` magnitude/eigenstructure by depth, sequence, validity, and uncertainty provider. This verifies transformation behavior; representation benefit is tested in Plan 07.

## Metrics

- Analytic/finite-difference Jacobian error.
- PSD violation and non-finite rates.
- Effective rank and principal-ray alignment.
- Invalid/capped fraction by depth bin.
- Depth and center uncertainty coverage where ground truth supports it.

## Acceptance criteria

- Analytic tests and finite differences pass within predeclared dtype tolerances.
- Valid outputs are finite, symmetric, PSD within tolerance, and approximately rank-1 as expected.
- Invalid disparities are masked, and every stabilization count is reported.
- Coordinate frame and units match Gaussian means and baseline behavior.
- One canonical implementation is used by training/evaluation.

## Failure conditions

- Analytic and numerical Jacobians disagree, frames/units remain ambiguous, covariance is unstable over valid ranges, or useful samples require arbitrary caps.

## Pivot / rollback path

Stop Plan 07 integration. Correct conventions or uncertainty units first. If first-order propagation is unstable only in identifiable extremes, freeze a validation-justified validity/cap policy and report exclusions. Consider optional pixel covariance only as a separately tested extension.

## Artifact/provenance requirements

Record uncertainty-provider artifact, formula/schema version, camera convention, units, thresholds/caps, calibration split, dtype/device, code/config hashes, and distribution diagnostics.

## Completion checklist

- [x] Canonical depth, backprojection, and propagation owners exist.
- [x] Analytic camera and finite-difference tests pass on local synthetic fixtures.
- [x] Invalid and near-zero disparity policies are explicit.
- [x] PSD/rank/eigenvalue diagnostics are retained.
- [x] No surface covariance or renderer logic was added.

Real-data validation and scientific acceptance remain pending, including unresolved SCARED-C
real-data semantics.

## Handoff to next plan

Plan 07 may assume verified camera-frame means and `Sigma_geo` with explicit units, masks, provider identity, and stabilization metadata. It may not alter propagation formulas inside renderer code.
