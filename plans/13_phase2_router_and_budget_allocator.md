# Plan 13 — Phase II Router and Budget Allocator

## Status

PLANNED

## Research stage

Phase II

## Recommended executor

Sol. Router objective selection depends on oracle label/margin structure, while measured-cost allocation and leakage-safe features require careful statistical and systems reasoning.

## Estimated implementation risk

HIGH. Class imbalance, router collapse, utility miscalibration, feature leakage, and allocation overhead can erase oracle headroom.

## Objective

Train a Gate-II-approved hierarchical router on validated pre-action evidence and allocate repair actions under measured-cost budgets without modifying Phase I or embedding action mechanics.

## Scientific question

Can a lightweight policy recover useful oracle headroom and improve the quality-cost frontier over uncertainty-only and stereo-only routing?

## Why this milestone exists

Only Gate II can establish that decisions are worth learning. This milestone converts measured intervention evidence into deployment decisions while preserving frozen representation and action semantics.

## Prerequisites

- Explicit Plan 12 router approval, action set, and scope.
- Immutable Phase I and oracle artifacts.
- Frozen action registry, cost units, data partitions, and no-test tuning protocol.

## Inputs

- Pre-action `RegionState` raw evidence and versioned feature-encoder candidates.
- Oracle utilities/labels/margins and measured action costs.
- Approved budget/lambda grid and strong deterministic baselines.

## Outputs

- Versioned raw-to-feature encoder selected after oracle analysis.
- Hierarchical router: STOP versus REPAIR, then STEREO versus GAUSSIAN if dual-source was approved.
- Measured-cost budget allocator and runtime coordinator.
- Immutable router artifact with calibration, leakage audit, cost/budget, and validation reports.

## Scope

- Candidate cheap features may include `sigma_d`, `sigma_D`, trace/max eigenvalue of `Sigma_geo`, left-right/photometric/cross-view/render residuals, visibility, validity, Gaussian statistics, and action cost estimates.
- Freeze exact features from training/validation oracle evidence; do not use post-action/test-only values.
- Select classification, utility regression, pairwise ranking, or a justified hybrid based on label imbalance, margins, and utility noise.
- Implement hierarchical routing for an approved dual-source space; simplify automatically to the Gate II-approved binary scope.
- Allocate action-region pairs with measured/calibrated costs, initially using positive utility-per-cost greedy selection; compare with exact small knapsack to measure regret before adding complexity.
- Keep decision and execution separate; runtime coordinator validates then invokes the action registry.

## Non-goals

- No joint Phase I/action/router training, router-owned repair logic, exact Jacobian/Fisher feature by default, second routing round, test tuning, or feature vector frozen before oracle analysis.

## Files expected to be created

- `src/reliable_endo_gs/routing/__init__.py`
- `src/reliable_endo_gs/routing/features.py`
- `src/reliable_endo_gs/routing/hierarchical.py`
- `src/reliable_endo_gs/routing/utility.py`
- `src/reliable_endo_gs/routing/budget.py`
- `src/reliable_endo_gs/routing/runtime.py`
- `src/reliable_endo_gs/training/router.py`
- `configs/router/hierarchical.yaml`
- `configs/training/router.yaml`
- `configs/experiment/p2d_router.yaml`
- `scripts/train_router.py`
- `scripts/evaluate_router.py`
- `tests/routing/` and `tests/integration/test_routed_actions.py`.

## Files expected to be modified

- Router/RoutingDecision contract implementations and artifact manifests.
- Evaluation/profiling for router regret, overhead, and budget accounting.
- Action registry only through its stable execution API.

## Interfaces and contracts

Feature encoding is a separate versioned transformation from raw `RegionState`. `Router.predict(features_or_state, budget) -> RoutingDecision` returns action/region selections, predicted utility/cost, and provenance; it never executes. `BudgetAllocator.allocate(candidates, budget)` guarantees feasibility under one declared cost model. Runtime executes decisions only through registered actions and reports actual cost.

## Scientific formulation

Owned by `routing/hierarchical.py`/`routing/utility.py`:

```text
p_repair = P(max_{a != STOP} U(a) > 0 | s)
p_source = P(U(stereo) > U(gaussian) | s, repair)
U_hat(a) = h_psi(s, a)
```

Owned by `routing/budget.py`:

```text
maximize sum_i U_hat_i(a_i) subject to sum_i C_i(a_i) <= B
score_i(a) = max(0, U_hat_i(a)) / (C_i(a) + epsilon)
```

Hierarchical intervention-supervised utility routing is the proposed Phase II decision mechanism. Greedy/knapsack optimization and classification/ranking losses are standard tools. Tests verify objective signs, masking, feasibility, and greedy regret on small exact cases.

## Numerical and coordinate conventions

Normalize features with training-only statistics and explicit missing indicators. Costs share units/hardware profile. Zero budget returns STOP. Invalid/ineligible actions are masked before normalization/argmax. Handle NaN/Inf and ties explicitly; probability/utility calibration uses validation sequences.

## Configuration changes

Define feature schema, encoder, approved action set, learning objective, imbalance handling, utility/cost targets, normalization artifact, budget grid, allocator, calibration, seeds, and checkpoint selection. Configs reference Phase I/oracle/action identities.

## Tests

### Unit

- Feature order/missingness/normalization and leakage checks.
- Hierarchical label logic, objectives, action masks, zero budget, budget never exceeded.
- Greedy versus exact knapsack on small cases and deterministic ties.

### Contract

- RoutingDecision contains decisions only; schemas/action IDs/budget units validate.

### Integration

- Frozen state -> features -> decision -> coordinator -> exact registered action -> final render.

### Smoke

- Tiny train/calibrate/infer flow with STOP/stereo/Gaussian and several budgets.

### Regression

- Fixed decisions and budget selections on synthetic oracle data.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/routing tests/integration
pytest -q
python -m compileall src
python scripts/evaluate_router.py --config configs/experiment/p2d_router.yaml
git diff --check
```

## Experiments

1. Choose classification/regression/ranking objective from training/validation oracle properties.
2. Feature-family ablations and leakage audit.
3. Flat versus hierarchical router if dual-source approved.
4. Greedy allocation regret versus exact small problems.
5. Compare deterministic uncertainty, consequence, stereo-only, and oracle policies across measured budgets.

## Metrics

- Repair/STOP and source-selection accuracy, balanced metrics, utility ranking/calibration.
- Oracle regret and oracle-gain recovery.
- Action distribution/failure by sequence and failure mode.
- Budget violations, predicted-versus-measured cost error, allocator regret.
- Router/feature/region/action overhead and net latency/memory.

## Acceptance criteria

- Router uses only inference-available pre-action evidence and approved actions.
- Budget is never exceeded under the decision cost contract; actual-cost deviations are reported.
- Router improves over strong deterministic and stereo-only baselines on validation at matched measured cost and recovers meaningful oracle headroom under predeclared criteria.
- Total overhead is measured and Phase I/action weights remain frozen.
- Immutable router artifact validates.

## Failure conditions

- Router collapses, fails simple baselines, has high oracle regret, violates budgets, depends on leaked features, or overhead erases quality-cost benefit.

## Pivot / rollback path

Use class-balanced/ranking objectives only when justified. Simplify to binary repair/STOP, deterministic uncertainty/consequence policy, or no learned router according to Gate II evidence. Use coarser regions/batched actions if overhead dominates. Preserve the oracle artifact and negative router result.

## Artifact/provenance requirements

Record Phase I/oracle/action IDs, raw and encoded feature schemas, normalization, router/checkpoint/config hashes, action registry/cost-profile identity, splits/seeds, code/environment/hardware, leakage audit, validation metrics, and budget/allocator behavior.

## Completion checklist

- [ ] Gate II authorization is validated programmatically.
- [ ] Feature schema follows oracle analysis and excludes leakage.
- [ ] Objective choice is evidence-based.
- [ ] Hierarchical decisions remain separate from execution.
- [ ] Budget/cost tests and strong-baseline comparisons pass.
- [ ] Router artifact and overhead report are immutable.

## Handoff to next plan

Plan 14 may assume the approved router artifact, or the explicit Plan 12/13 simplified-policy result if routing fails. All final comparisons must use frozen artifacts and matched measured cost.
