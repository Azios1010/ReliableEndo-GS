# Plan 14 — Final Matched-Cost Evaluation

## Status

PLANNED

## Research stage

Final evaluation

## Recommended executor

Sol. The primary program claim depends on fair multi-artifact comparisons, statistical aggregation, device profiling, and disciplined claim reduction after negative results.

## Estimated implementation risk

HIGH. Small preprocessing or hardware mismatches, budget leakage, or selective reporting can produce an invalid Pareto claim even when individual models work.

## Objective

Run the frozen SCARED comparison suite at matched measured cost and determine which Phase I/Phase II claims are supported by calibration, geometry, rendering, routing, and end-to-end efficiency evidence.

## Scientific question

At matched measured compute, does the approved ReliableEndo-GS path improve the reconstruction quality-cost frontier over Endo-E2E-GS, fixed refinement, uncertainty routing, and single-source policies?

## Why this milestone exists

Quality alone cannot support selective-compute claims, and oracle/router metrics alone cannot support reconstruction claims. This milestone evaluates the whole chain under shared data, metrics, and hardware.

## Prerequisites

- Plan 12 Gate II decision.
- Plan 13 router artifact if approved/successful, or a frozen simplified policy.
- Immutable baseline, Phase I, action, oracle, and applicable router artifacts.
- Test sequences untouched by calibration/model/policy selection.

## Inputs

- Frozen artifact chain, experiment configs, split manifests, metric schemas, cost profiles, and paper claim matrix.
- Identical production hardware/backend and evaluation conditions for matched-cost comparisons.

## Outputs

- Immutable final evaluation/report artifact with source tables, plots, Pareto frontiers, confidence intervals, failures, and claim decisions.
- Reproduction commands for every reported table/figure.
- Negative-result and limitation report where gates/policies failed.

## Scope

Compare where applicable: Endo-E2E-GS baseline; frozen Phase I/no repair; fixed additional refinement; uncertainty-threshold/top-K refinement; stereo-only adaptive routing; GS-only fixed/triggered repair; full dual-source router; and oracle upper bound. Match measured latency/compute budgets rather than FLOPs alone. Evaluate multiple operating points and include the full overhead of uncertainty, region extraction, features, router, allocation, actions, rendering, and synchronization.

Run required Phase I ablations and Phase II policy comparisons only from frozen configs. Report per-sequence and grouped uncertainty, failure modes, actual action distributions, exclusions, and at least three seeds for stochastic training where feasible as specified by the proposal.

## Non-goals

- No model/action/router tuning on test data, no new method variant after results, no manual metric copying, no clinical/safety claim, and no efficiency claim from selected pixels/FLOPs alone.

## Files expected to be created

- `src/reliable_endo_gs/evaluation/matched_cost.py`
- `src/reliable_endo_gs/evaluation/statistics.py`
- `src/reliable_endo_gs/evaluation/pareto.py`
- `src/reliable_endo_gs/evaluation/reporting.py`
- `configs/evaluation/final.yaml`
- `configs/experiment/p2e_pareto.yaml`
- `scripts/run_final_evaluation.py`
- `scripts/build_report_tables.py`
- `scripts/build_pareto_plots.py`
- `reports/final/` generated source manifests/templates.
- `tests/evaluation/test_matched_cost.py`
- `tests/evaluation/test_statistics.py`
- `tests/evaluation/test_pareto.py`
- `tests/integration/test_final_report_lineage.py`

## Files expected to be modified

- Artifact/report schemas for final lineage.
- Profiling aggregation for consistent end-to-end cost.
- Documentation claims/limitations only after evidence is verified.

## Interfaces and contracts

Each evaluation record binds method artifact IDs, dataset/split, seed, operating point, measured cost context, metric schema, and validity counts. Matched-cost selection uses a predeclared interpolation/nearest-feasible rule. Report builders consume source records and cannot alter metrics.

## Scientific formulation

Primary claim quantity is a quality-cost Pareto comparison, implemented in `evaluation/pareto.py`; matched-budget comparisons are owned by `evaluation/matched_cost.py`. Oracle gain recovery is:

```text
(Q_router - Q_base) / (Q_oracle - Q_base)
```

These are evaluation definitions, not new scientific formulas. Handle near-zero oracle denominators explicitly. All method formulas remain with their canonical owners.

## Numerical and coordinate conventions

All methods share resolution, masks, coordinate/metric definitions, precision policy, renderer, and test sample set. Latency uses the same hardware, warm-up, synchronization, repetitions, and aggregation; report mean, median, and p90 where variable. Bootstrap groups correlated frames by sequence/frame group.

## Configuration changes

Freeze method artifact IDs, operating budgets/lambdas, primary/guardrail metrics, matched-cost rule, seeds, failure masks, confidence-interval method, profiler, report destinations, and claim tests in `configs/evaluation/final.yaml` and `p2e_pareto.yaml`.

## Tests

### Unit

- Matched-budget selection, Pareto dominance/frontier, bootstrap grouping, oracle-gain denominator, missing/failure/exclusion handling.

### Contract

- All rows share compatible split/metric/cost schemas and trace to immutable artifacts.

### Integration

- Frozen artifacts -> evaluation records -> source tables/plots -> report manifest without mutation.

### Smoke

- Tiny multi-policy evaluation creates a traceable report.

### Regression

- Synthetic known Pareto frontiers and table values; do not freeze scientifically incorrect outputs.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/evaluation tests/integration
pytest -q
python -m compileall src
python scripts/run_final_evaluation.py --config configs/experiment/p2e_pareto.yaml
git diff --check
```

## Experiments

- Frozen final SCARED test evaluation for all applicable baselines/ablations/policies across measured budgets.
- Required stochastic seeds and per-sequence grouped bootstrap intervals.
- Failure-mode slices: low texture, depth edges, specularity, occlusion, far range, visibility, deformation where defined before test.
- End-to-end profiling and overhead decomposition on the same target hardware.

## Metrics

- Stereo/depth: EPE, D1/bad pixel, depth MAE/RMSE.
- Calibration: NLL, AUSE, AUROC, empirical coverage/calibration error.
- Geometry: depth/point error, normal error if supported, depth-edge metrics, outlier ratio.
- Rendering: PSNR, SSIM, LPIPS for left/right and masked/full where applicable.
- Routing: oracle regret, repair/source accuracy, utility ranking/calibration, oracle-gain recovery, action distribution.
- Efficiency: latency, FPS, peak memory, routing/action/region overhead, measured action cost, net compute.

## Acceptance criteria

- Every comparison uses the same frozen split, preprocessing, metric code, and matched-cost hardware protocol.
- Final report traces each value to immutable source records/artifacts.
- The primary Pareto claim is made only if supported across predeclared operating points and guardrails; otherwise the report states the reduced/negative conclusion.
- Oracle is labeled upper bound, not deployment.
- Failures/exclusions, per-sequence results, intervals, and complete overhead are reported.

## Failure conditions

- Router fails simple policies at matched cost, net overhead erases savings, improvements violate geometry/rendering guardrails, results are sequence-specific without support, or lineage/comparison parity is incomplete.

## Pivot / rollback path

Report the strongest supported upstream result: Phase I only, stereo-only adaptive repair, deterministic policy, or oracle headroom analysis. Do not change budgets/metrics after test results. Preserve negative router/dual-source evidence and narrow claims.

## Artifact/provenance requirements

Record the complete baseline -> Phase I -> oracle -> router/simplified-policy lineage, configs/hashes, code/upstream commits, split, seeds, metric/cost schemas, hardware/environment, raw rows, table/plot source hashes, confidence procedure, exclusions, and claim decision.

## Completion checklist

- [ ] Frozen methods and test protocol are unchanged.
- [ ] All applicable baselines and ablations are included.
- [ ] Cost is measured end to end on shared hardware.
- [ ] Metrics, intervals, failures, and action distributions are complete.
- [ ] Report values trace to immutable records.
- [ ] Claims match evidence and preserve negative results.

## Handoff to next plan

Plan 15 may assume one immutable final report artifact and an evidence-limited claim matrix. It may validate externally and package releases, but cannot broaden claims beyond compatibility and measured results.
