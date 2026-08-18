# Plan 10 — Phase II Regions and Repair Actions

## Status

PLANNED

## Research stage

Phase II

## Recommended executor

Sol. Region isolation, recurrent stereo reuse, Gaussian residual parameterization, immutable state transitions, and action attribution are tightly coupled to the audited upstream model.

## Estimated implementation risk

HIGH. Local execution may not be supported efficiently, repair heads may have no independent gain, and halo/scatter errors can invalidate intervention attribution.

## Objective

Implement a fixed-grid region MVP and three independent actions—STOP, one stereo repair, and one Gaussian attribute repair—using a common contract suitable for direct calls, training, oracle collection, profiling, and deployment.

## Scientific question

Can stereo geometry and Gaussian attributes be repaired independently on identifiable subsets of a frozen Phase I state, with measurable benefit and bounded cost?

## Why this milestone exists

Gate I provides stable evidence. Gate II requires real interventions, so action mechanics and region write semantics must be fixed and tested before any oracle label or router exists.

## Prerequisites

- Gate I PASS and immutable Phase I artifact from Plan 09.
- Upstream recurrent/feature and Gaussian parameterization audit.
- Frozen raw Phase II evidence semantics; no router encoding.

## Inputs

- Immutable `ReconstructionState` from the frozen Phase I artifact.
- Verified baseline features/checkpoints and renderer.
- Training/validation sequences only for action development.

## Outputs

- `RegionSelection`, `RegionState`, gather/scatter, halo policy, and action registry.
- `RepairAction`, `ActionResult`, and `CostEstimate` implementations/schemas.
- Exact STOP, one selected stereo action, and one Gaussian attribute action.
- Independent action checkpoints, profiling evidence, and subset-gain report.

## Scope

- Use a fixed grid plus context halo; freeze core/halo size before oracle collection.
- Make core write ownership explicit when halos overlap.
- STOP returns a new explicit no-change result.
- Prefer extra recurrent stereo iterations when upstream state reuse is faithful and measurable; otherwise select one local residual disparity head. Do not implement both as coequal MVP actions.
- Gaussian repair predicts local residual rotation, log-scale, and opacity-logit updates while holding means/depth and `Sigma_geo` fixed.
- Recompute every scientifically dependent field after an allowed change and enumerate `changed_fields`.
- Train action heads before oracle collection, with Phase I frozen.

## Non-goals

- No original seven-action space, adaptive region proposal, learned router, chained multi-action labels, second routing round, center-changing Gaussian action, or joint Phase I/action training.
- Router code cannot appear in actions; oracle code cannot duplicate action mechanics.

## Files expected to be created

- `src/reliable_endo_gs/regions/__init__.py`
- `src/reliable_endo_gs/regions/grid.py`
- `src/reliable_endo_gs/regions/gather.py`
- `src/reliable_endo_gs/regions/scatter.py`
- `src/reliable_endo_gs/regions/state.py`
- `src/reliable_endo_gs/actions/__init__.py`
- `src/reliable_endo_gs/actions/base.py`
- `src/reliable_endo_gs/actions/registry.py`
- `src/reliable_endo_gs/actions/stop.py`
- `src/reliable_endo_gs/actions/stereo_repair.py`
- `src/reliable_endo_gs/actions/gaussian_repair.py`
- `src/reliable_endo_gs/training/actions.py`
- `configs/regions/fixed_grid.yaml`
- `configs/actions/stop.yaml`
- `configs/actions/stereo_repair.yaml`
- `configs/actions/gaussian_repair.yaml`
- `configs/experiment/p2a_actions.yaml`
- `scripts/train_actions.py`
- `scripts/evaluate_actions.py`
- `tests/regions/` and `tests/actions/` focused suites.

## Files expected to be modified

- Scientific contracts to add versioned Phase II `RegionState`, `RepairAction`, `ActionResult`, and `CostEstimate` implementations.
- Profiling and artifact manifests for action identity/checkpoints.
- Baseline adapter only through its documented recurrent interface, if selected.

## Interfaces and contracts

Each action has stable identity and implements `apply(state, regions) -> ActionResult` and `estimate_cost(state, regions) -> CostEstimate`. Inputs are immutable. `ActionResult` records affected regions, post-state/output, changed fields, measured cost when executed under profiling, failure/validity, and provenance. The registry resolves the exact same object in training, oracle, evaluation, runtime, and profiling.

## Scientific formulation

Stereo residual implementation owner: `actions/stereo_repair.py`.

```text
d' = d_0 + Delta d
D' = f_x B / d'
mu' = backproject(D')
```

Gaussian residual implementation owner: `actions/gaussian_repair.py`.

```text
r' = normalize(r ⊕ Delta r)
s' = s * exp(Delta s)
alpha' = sigmoid(logit(alpha) + Delta alpha)
```

These are implementation transformations/supporting repair designs, not standalone novelty. The exact rotation composition follows the verified upstream representation and is tested analytically. Proposed Phase II contribution depends on intervention-measured source utility, not these residual identities.

## Numerical and coordinate conventions

Stereo repair reuses canonical geometry and propagation owners; it never duplicates formulas. Validate disparity/scale/opacity/rotation ranges and finite values. Gaussian repair preserves means and `cov_center` bitwise or within an explicitly justified serialization tolerance. Halo reads cannot write outside authorized cores. Empty/invalid regions produce explicit ineligible results.

## Configuration changes

Define grid/core/halo, action eligibility, one stereo implementation choice, iteration count or head structure, Gaussian residual bounds, training configs, checkpoint identities, and profiling context. No router features or budget policy yet.

## Tests

### Unit

- Grid coverage, halo/core mapping, gather/scatter, overlap resolution, empty/degenerate regions.
- STOP exact no-op.
- Stereo action changes only disparity-dependent fields.
- Gaussian action preserves centers/depth/`Sigma_geo` and changes only allowed attributes.

### Contract

- Stable identities, immutable inputs, changed-field completeness, failures, and cost units.
- Registry returns the same implementation for every caller role.

### Integration

- Frozen Phase I state -> each independent action -> rerender -> metrics/profile.

### Smoke

- All three actions run independently from identical serialized state.

### Regression

- No-op checksum, write-isolation fixtures, and selected action-output summaries.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/regions tests/actions tests/integration
pytest -q
python -m compileall src
python scripts/evaluate_actions.py --config configs/experiment/p2a_actions.yaml
git diff --check
```

## Experiments

Train/evaluate each repair independently on training/validation regions. Compare each with STOP, full-frame/fixed extra refinement where applicable, and measure local/global gain by failure-mode subset and region size/halo.

## Metrics

- Local/global depth/geometry and rendering deltas.
- Changed/touched pixel and primitive counts.
- Action success/failure/eligibility rates.
- Measured latency mean/median/p90, peak memory, and estimated-cost calibration.
- Gain distribution by uncertainty, edge, texture, visibility, occlusion, and specularity.

## Acceptance criteria

- STOP is an exact no-op and all actions preserve input state.
- Core/halo write isolation and changed-field tests pass.
- Stereo and Gaussian actions each show independent, repeatable gain on some validation subset or are explicitly marked unsupported before oracle work.
- Gaussian repair preserves center/depth; stereo repair uses canonical recomputation.
- One implementation identity serves all execution roles.

## Failure conditions

- Local execution is not faithful/efficient, actions mutate unauthorized fields, neither repair has subset gain, or Gaussian repair can improve only by moving centers.

## Pivot / rollback path

Fall back from recurrent reuse to one local disparity residual head if justified. Use coarser regions/full-frame batching if local overhead dominates. If Gaussian repair has no headroom, carry only STOP/stereo into a documented stereo-only Gate II analysis; do not retain a decorative action.

## Artifact/provenance requirements

Record Phase I artifact ID, region/action schema versions, action code/config/checkpoint hashes, training split/seeds, upstream identity, profiling protocol/hardware, eligibility/failure rules, and independent per-action results.

## Completion checklist

- [ ] Gate I authorization and frozen Phase I identity verified.
- [ ] Fixed-grid/core/halo semantics are tested.
- [ ] STOP, one stereo, and one Gaussian action are independent.
- [ ] Gaussian centers remain fixed under Gaussian repair.
- [ ] Same registry implementations serve every caller.
- [ ] Action checkpoints, gains, costs, and failures are retained.

## Handoff to next plan

Plan 11 may assume frozen region/action schemas and exact action implementations/checkpoints. It must reset identical starting state for each one-step intervention and may not tune the actions while collecting final oracle records.
