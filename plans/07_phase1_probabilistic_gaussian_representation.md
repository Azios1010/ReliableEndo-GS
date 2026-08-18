# Plan 07 — Phase I Probabilistic Gaussian Representation

## Status

PLANNED

## Research stage

Phase I

## Recommended executor

Sol. This central milestone combines representation semantics, linear algebra, custom renderer integration, numerical stability, and high-stakes ablations.

## Estimated implementation risk

HIGH. Covariance may only blur outputs, determinant correction may be unstable, or the production renderer may not expose a compatible covariance path.

## Objective

Keep learned surface support distinct from observation-derived center uncertainty, construct an effective per-primitive covariance and amplitude correction, integrate it before nonlinear compositing, and measure headroom against simpler alternatives.

## Scientific question

Does stereo-derived center covariance improve geometry or cross-view reliability beyond opacity-only confidence and fixed/Mip-style filtering while preserving feed-forward inference?

## Why this milestone exists

This is the key Phase I representation hypothesis. It follows verified propagation so any gain or failure can be attributed to how uncertainty enters Gaussian primitives rather than to incorrect geometry.

## Prerequisites

- Plans 03, 04/05, and 06 accepted.
- Verified upstream Gaussian parameterization and production renderer API.
- Oracle uncertainty/headroom protocol from Plan 04.

## Inputs

- Baseline `GaussianField` attributes and renderer.
- Means and `Sigma_geo` from Plan 06.
- Frozen baseline, uncertainty provider, SCARED split, and evaluation metrics.

## Outputs

- Canonical surface-covariance and marginalization modules.
- Renderer-facing `cov_effective` integration plus deterministic CPU reference subset where feasible.
- Stability diagnostics and full representation ablation report.
- Candidate Phase I representation checkpoint/config for Plan 08/Gate I.

## Scope

- Convert verified rotations/scales into `Sigma_surf`.
- Combine `Sigma_surf` and `Sigma_geo` without losing their separate fields/provenance.
- Implement determinant-based amplitude/opacity correction and rank-1 optimized path only after parity with the full formula.
- Prefer the production rasterizer's precomputed 3D covariance interface; otherwise use a tested factorization consistent with upstream conventions.
- Compare baseline, loss/confidence weighting where applicable, opacity-only, fixed/isotropic, covariance-only, Mip-style filtering, and stereo-derived covariance plus correction.
- Execute oracle-`sigma_d` headroom before relying on learned uncertainty results.

## Non-goals

- Do not claim covariance addition, Gaussian convolution, determinant correction, Mip filtering, or standard surface covariance as new.
- Do not claim exact expectation of the full rendered image; alpha compositing is nonlinear and primitives interact.
- No cross-view training yet, action, oracle routing, or per-scene/Monte Carlo inference.

## Files expected to be created

- `src/reliable_endo_gs/probabilistic_gs/__init__.py`
- `src/reliable_endo_gs/probabilistic_gs/support_covariance.py`
- `src/reliable_endo_gs/probabilistic_gs/marginalization.py`
- `src/reliable_endo_gs/probabilistic_gs/stability.py`
- `src/reliable_endo_gs/rendering/production.py`
- `src/reliable_endo_gs/rendering/reference.py`
- `src/reliable_endo_gs/rendering/protocol.py`
- `configs/probabilistic_gs/two_covariance.yaml`
- `configs/rendering/production.yaml`
- `configs/experiment/p1d_probabilistic_gs.yaml`
- `scripts/evaluate_phase1_representation.py`
- `tests/probabilistic_gs/test_support_covariance.py`
- `tests/probabilistic_gs/test_marginalization.py`
- `tests/rendering/test_covariance_contract.py`
- `tests/integration/test_probabilistic_rendering.py`

## Files expected to be modified

- Baseline adapter only at its documented renderer boundary.
- `GaussianField` contract/schema for verified covariance fields.
- Evaluation/profiling to report representation variants and overhead.

## Interfaces and contracts

`surface_covariance(rotations, scales)` owns `cov_surface`. `marginalize(cov_surface, cov_center, opacity, policy)` returns `cov_effective`, effective opacity, masks, and diagnostics while preserving both inputs. A renderer request declares which covariance it consumes. Production and reference backends implement the supported common contract; only production results support scientific performance claims.

## Scientific formulation

Owned by `probabilistic_gs/support_covariance.py`:

```text
Sigma_surf = R(q) diag(s^2) R(q)^T
```

Owned by `probabilistic_gs/marginalization.py`:

```text
Sigma_eff = Sigma_surf + Sigma_geo
kappa = sqrt(det(Sigma_surf) / det(Sigma_eff))
alpha_eff = clip(alpha kappa, 0, 1)
```

For `Sigma_geo = v v^T`, test the candidate optimization:

```text
kappa = (1 + v^T Sigma_surf^-1 v)^(-1/2)
```

Surface covariance and determinant identities are standard prior mathematics/implementation transformations. The proposed contribution is the calibrated observation-derived center covariance, its explicit separation from surface support, and its evaluated role in feed-forward endoscopic GS. Marginalization is exact per primitive before compositing, not exact expected full-image rendering.

## Numerical and coordinate conventions

Normalize/validate rotation under verified ordering; scales are strictly positive in the Gaussian frame. Covariances share the means' frame and dtype. Use stable log-determinant/solve rather than explicit inverse where possible. Configure epsilon and eigenvalue policy, report corrections, handle singular/invalid entries through masks, and verify alpha parameterization before clipping.

## Configuration changes

Define representation variant, covariance source, correction on/off, stability epsilon, factorization path, rank-1 optimization, renderer backend, and ablation identity. `p1c_covariance_oracle.yaml` and `p1d_probabilistic_gs.yaml` name immutable upstream artifacts.

## Tests

### Unit

- Known rotations/scales, covariance symmetry/PSD/positive definiteness.
- Zero center uncertainty exactly recovers baseline representation.
- Effective covariance addition and determinant correction stability.
- Rank-1 formula matches full determinant result.
- Invalid scale/opacity/covariance and gradient checks.

### Contract

- Covariance meanings remain distinct and renderer declares consumption.

### Integration

- Baseline field -> propagation -> marginalization -> production/reference render on small scenes.

### Smoke

- Deterministic CPU reference case and minimal GPU production case.

### Regression

- Analytic Gaussian renders and pinned variant summaries under declared tolerances.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/probabilistic_gs tests/rendering tests/integration
pytest -q
python -m compileall src
python scripts/evaluate_phase1_representation.py --config configs/experiment/p1d_probabilistic_gs.yaml
git diff --check
```

## Experiments

1. Oracle-uncertainty covariance headroom versus opacity-only and fixed filtering.
2. Full ablation set with cross-view training off.
3. Full versus rank-1 computation and stability analysis.
4. Geometry/rendering/overhead evaluation by sequence and failure mode.

## Metrics

- Depth/point/normal/edge errors and outlier rate.
- Left/right PSNR, SSIM, LPIPS where current model permits diagnostic right render.
- Covariance eigenvalues, invalid/stabilized fraction, opacity change.
- Latency/FPS/peak memory and covariance/rasterization overhead.

## Acceptance criteria

- Oracle covariance demonstrates predeclared headroom beyond opacity-only and fixed/isotropic filtering on geometry or right-view reliability.
- Zero-uncertainty recovery, PSD, determinant, and renderer contract tests pass.
- Stereo-derived covariance results are attributable and reported against all required simpler variants.
- Overhead and numerical interventions are measured, not hidden.

## Failure conditions

- Oracle covariance lacks headroom, only produces blur/left-view gain with geometry harm, or renderer integration requires changing baseline semantics.
- Numerical stabilization dominates a material share of valid primitives.

## Pivot / rollback path

Stop the two-covariance claim and retain calibrated uncertainty, opacity-only/confidence-aware GS, or fixed filtering as the documented Phase I pivot. Use rank-1 caps only if validation-justified. Do not proceed to Gate I pretending covariance succeeded.

## Artifact/provenance requirements

Record baseline/uncertainty/geometry artifact IDs, formula and renderer versions, variant config, split, seeds, checkpoint, upstream commit, stability counts, backend/hardware, raw metrics, and oracle-versus-predicted uncertainty identity.

## Completion checklist

- [ ] Surface and center covariance remain semantically separate.
- [ ] Full and rank-1 formulas are tested.
- [ ] Production/reference renderer contracts are verified.
- [ ] Oracle headroom and required ablations are complete.
- [ ] Numerical and overhead diagnostics are retained.
- [ ] Claim boundary excludes exact full-image expectation.

## Handoff to next plan

Plan 08 may assume a validated representation candidate and renderer path, but must evaluate cross-view supervision independently and preserve a cross-view-off checkpoint/config for Gate I attribution.
