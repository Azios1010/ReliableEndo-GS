# Plan 12 — Phase II Gate and Oracle Analysis

## Status

PLANNED

## Research stage

Gate II

## Recommended executor

Sol. The milestone must distinguish genuine heterogeneous repair utility from noise, imbalance, cost artifacts, and validation overfitting before authorizing a router.

## Estimated implementation risk

HIGH. Scientific risk dominates: one action may dominate, Gaussian repair may lack support, or oracle advantage may be too small to justify routing overhead.

## Objective

Analyze the immutable intervention artifact and decide whether evidence supports a dual-source router, a simpler stereo-only/binary policy, or no learned router.

## Scientific question

Is heterogeneous dual-source repair necessary and learnable enough to outperform fixed or stereo-only policies at matched measured cost?

## Why this milestone exists

A router cannot create intervention headroom. Gate II prevents classifying noisy oracle argmax labels or building a complex policy when one fixed action already captures the available utility.

## Prerequisites

- Plan 11 oracle artifact validated.
- Gate II protocol, pilot/final split roles, cost units, and comparison policies defined.

## Inputs

- Pre-action state evidence and independent STOP/stereo/Gaussian outcomes.
- Measured costs, utility versions, failure statuses, sequence/failure-mode metadata.
- Fixed, top-K uncertainty, consequence, stereo-only, and oracle policy definitions.

## Outputs

- Gate II report with best-action distributions, margins, diversity, predictability diagnostics, and matched-cost oracle comparisons.
- Frozen quantitative criteria selected from pilot data before final held-out gate evaluation.
- Explicit decision: dual-source router approved, stereo-only/binary simplification, deterministic policy, or no router.

## Scope

- Analyze overall/per-sequence STOP, stereo, and Gaussian support and confidence intervals.
- Measure utility margins, ties, failures, and sensitivity to lambda/budget.
- Compare oracle to fixed/uniform, uncertainty-threshold/top-K, consequence proxy, GS-only, and stereo-only policies at matched measured cost.
- Check action preference by low texture, depth edge, specularity, occlusion, uncertainty, residual, and visibility.
- Assess whether pre-action state contains source-discriminative signal without training the final router.
- Use pilot analysis to choose non-arbitrary support/headroom criteria, freeze them, then evaluate on a held-out gate split.

## Non-goals

- No final router, budget allocator training, post-action feature leakage, action retuning, arbitrary thresholds chosen to force pass, or test-set inspection.

## Files expected to be created

- `src/reliable_endo_gs/oracle/analysis.py`
- `src/reliable_endo_gs/evaluation/gate2.py`
- `scripts/analyze_oracle.py`
- `scripts/review_phase2_gate.py`
- `configs/evaluation/gate2.yaml`
- `configs/experiment/p2c_oracle_analysis.yaml`
- `reports/phase2/gate2_decision.md`
- `tests/oracle/test_analysis.py`
- `tests/evaluation/test_gate2.py`

## Files expected to be modified

- Oracle artifact readers for analysis-only indexed access, if needed.
- `PLAN.md` status only after decision approval.

## Interfaces and contracts

Analysis is read-only and returns versioned summary records with population, eligibility, confidence interval, policy, cost, utility, and upstream artifact identity. Gate decision names the approved action set and router scope. Plan 13 must reject execution unless this decision explicitly authorizes it.

## Scientific formulation

Use the canonical `oracle/utility.py` outputs; do not recompute a private utility. Policy cost/quality aggregation lives in `oracle/analysis.py`. Bootstrap confidence intervals group by sequence/frame group. Oracle advantage and source support are evaluation quantities. Gate criteria are frozen decision rules, not proposed model contributions.

## Numerical and coordinate conventions

Compare only rows with compatible metric/cost schemas and hardware profiles. Treat action failures/ineligibility separately from losses. Report ties within a predeclared numerical tolerance and sensitivity to that tolerance. Avoid frame-independent uncertainty assumptions.

## Configuration changes

Define pilot and held-out gate partitions, policy set, budget/lambda grid, tie/margin rules, bootstrap grouping/repetitions, minimum evidence coverage, and criteria-freeze manifest. Thresholds are not filled arbitrarily in advance; pilot-derived values and rationale are committed before held-out evaluation.

## Tests

### Unit

- Best-action distribution, ties, margins, policy utility/cost, confidence intervals, missing/failure handling.

### Contract

- Gate rejects incompatible oracle/Phase I/action schemas and unapproved held-out access.

### Integration

- Immutable oracle artifact -> analysis tables/plots -> decision record.

### Smoke

- Synthetic cases exercise dual-source PASS, stereo-only pivot, dominant-action simplification, and no-headroom STOP.

### Regression

- Frozen synthetic oracle tables with known policy orderings.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/oracle tests/evaluation
pytest -q
python -m compileall src
python scripts/review_phase2_gate.py --config configs/evaluation/gate2.yaml
git diff --check
```

## Experiments

1. Pilot support/margin/cost sensitivity and criteria freeze.
2. Held-out Gate II policy comparison at multiple measured budgets.
3. Failure-mode and sequence-stratified oracle analysis.
4. Cheap-state source-predictability diagnostic without final router tuning.

## Metrics

- Best-action and positive-utility support with confidence intervals.
- STOP support; stereo/Gaussian support among repairable regions.
- Utility margin/tie/failure distributions.
- Oracle advantage over fixed, uncertainty, consequence, GS-only, and stereo-only policies.
- Cost-normalized quality frontier and provisional source predictability.

## Acceptance criteria

- Dual-source approval requires nontrivial, repeatable support for both repair sources, meaningful oracle advantage over stereo-only/fixed policies at matched measured cost, and pre-action evidence plausibly predictive under pilot-frozen criteria.
- Criteria and held-out decision are traceable and not chosen to ensure success.
- Decision explicitly declares permitted Plan 13 scope/action set.

## Failure conditions

- Gaussian repair rarely wins: dual-source hypothesis unsupported.
- One action dominates almost everywhere: hierarchical multiclass routing unnecessary.
- Oracle barely beats a simple fixed policy after cost: learned routing unjustified.
- Apparent diversity comes from failures, incompatible costs, or sequence leakage.

## Pivot / rollback path

- Gaussian support absent: stereo-only adaptive STOP/repair.
- One repair dominates: binary repair/stop or fixed schedule.
- Oracle advantage negligible: no complex router; retain oracle/action analysis as output.
- State not source-discriminative despite diverse oracle: regress best utility directly or drop diagnosis claim, only if a separately approved simplified plan remains justified.

## Artifact/provenance requirements

Record oracle/Phase I/action IDs, analysis and gate config hashes, pilot/final partitions, frozen criteria, metric/cost/utility schemas, code revision, bootstrap seed, all plots/source tables, decision, dissent/limitations, and selected pivot.

## Completion checklist

- [ ] Action distributions, margins, failures, and costs analyzed.
- [ ] Oracle compared with strong simple/stereo-only policies.
- [ ] Criteria frozen after pilot and before held-out gate run.
- [ ] Gate decision and allowed action/router scope are explicit.
- [ ] Negative result selects a viable simpler path.
- [ ] No final router was implemented.

## Handoff to next plan

Plan 13 may start only with an explicit Gate II router approval and must use the approved action set, feature availability, cost semantics, and immutable oracle artifact. Otherwise Plan 14 evaluates the approved simplified policy directly.
