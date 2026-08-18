# Plan 09 — Phase I Gate and Artifact Freeze

## Status

PLANNED

## Research stage

Gate I

## Recommended executor

Sol. This is a scientific decision and evidence-synthesis milestone, not a routine coding task.

## Estimated implementation risk

HIGH. Scientific risk dominates: a technically functional Phase I may fail calibration, representation headroom, geometry evidence, or feasibility, requiring a reduced claim or stop.

## Objective

Evaluate predeclared Gate I subconditions, preserve negative evidence, and either freeze one immutable Phase I artifact for Phase II or stop/pivot the program explicitly.

## Scientific question

Has Phase I established a useful and feasible reliability representation that warrants intervention research without relying on Phase II labels?

## Why this milestone exists

Actions and oracle collection would otherwise be built on a moving representation. Gate I converts evidence, not code availability, into a formal artifact boundary.

## Prerequisites

- Plan 04 evidence and Plan 05 result or documented proxy skip.
- Plans 06-08 completed with required ablations.
- Immutable baseline artifact and frozen evaluation protocol.

## Inputs

- Raw per-sequence uncertainty, calibration, geometry, rendering, cross-view, stability, latency, and memory results.
- All baseline/opacity/fixed-filtering/covariance/cross-view ablations.
- Candidate checkpoints/configs and artifact manifests.

## Outputs

- Gate I review package with tables, plots, claim matrix, negative results, and signed decision record.
- If passed, one immutable Phase I artifact and frozen Phase II feature/evidence schema.
- If conditional, a proxy-based or reduced-scope artifact with explicit claim/pivot.
- If failed, a stop report and no authorization for Plan 10.

## Scope

Review four subconditions:

- **I-A uncertainty usefulness:** learned/calibrated uncertainty or an approved proxy predicts error meaningfully under frozen calibration/ranking criteria.
- **I-B representation headroom:** stereo-derived covariance benefits geometry or right-view reliability beyond opacity-only confidence and fixed/isotropic/Mip-style filtering.
- **I-C geometry/cross-view evidence:** intended reliability improves without unacceptable degradation elsewhere; photometric-only gain is labeled accordingly.
- **I-D practical feasibility:** deterministic feed-forward inference has acceptable measured latency/memory overhead; the proposal's 10-15% inference-overhead target is evaluated under a preselected operating definition, not silently reinterpreted.

Freeze criteria/aggregation before final held-out evaluation, run the review, and prohibit post-result tuning within the same gate decision.

## Non-goals

- No new scientific model, action, oracle, router, or rescue experiment invented after seeing final test results.
- No automatic pass because code/checkpoints exist.
- Do not combine incompatible best results from different checkpoints into one artifact.

## Files expected to be created

- `src/reliable_endo_gs/evaluation/gates.py`
- `src/reliable_endo_gs/runtime/artifacts.py` or the approved artifact owner.
- `scripts/review_phase1_gate.py`
- `scripts/freeze_phase1_artifact.py`
- `configs/evaluation/gate1.yaml`
- `reports/phase1/gate1_decision.md`
- `tests/evaluation/test_gate1.py`
- `tests/runtime/test_phase1_artifact.py`

## Files expected to be modified

- Artifact manifest/schema definitions.
- `docs/artifact_lifecycle.md` only with verified implementation details.
- `PLAN.md` status only after the evidence decision is approved.

## Interfaces and contracts

Gate evaluation consumes immutable report inputs and produces a decision record; it cannot train or mutate candidates. The Phase I loader accepts artifact identity, verifies payload hashes/upstream schemas, and returns frozen reconstruction/feature producers. Phase II feature exports include `sigma_d`, optional `sigma_D`, `Sigma_geo` summaries, cross-view/render residuals, visibility, validity, and Gaussian diagnostics without router encoding.

## Scientific formulation

No new method equation. Gate aggregations and confidence intervals are evaluation procedures owned by `evaluation/gates.py`. Each reviewed equation points back to its canonical Plan 05-08 owner and classification. Gate rules are scientific governance, not learned objectives.

## Numerical and coordinate conventions

All compared variants use the same split, resolution, masks, units, renderer, metrics, and hardware where cost is compared. Report valid counts and stabilization/exclusion rates. Confidence intervals group by sequence/frame group rather than treating adjacent frames as independent.

## Configuration changes

`configs/evaluation/gate1.yaml` declares evidence artifact IDs, primary/guardrail metrics, aggregation, subcondition logic, overhead definition, and report outputs. Values are frozen before held-out evaluation and retained even if a condition fails.

## Tests

### Unit

- Pass/fail/conditional logic, missing evidence, incompatible schema, and boundary cases.

### Contract

- Phase I artifact manifest, upstream lineage, payload hashes, feature schema, and immutable loading.

### Integration

- Candidate artifacts -> gate report -> freeze/load round trip.

### Smoke

- Synthetic evidence exercises PASS, PIVOT, and STOP without touching scientific state.

### Regression

- Frozen gate-decision fixture and report-source lineage.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/evaluation tests/runtime
pytest -q
python -m compileall src
python scripts/review_phase1_gate.py --config configs/evaluation/gate1.yaml
git diff --check
```

## Experiments

No new exploratory experiment. Execute the frozen Gate I evaluation and synthesize existing approved runs. Any missing required run sends the gate back to its owning plan before test access.

## Metrics

- I-A: NLL/AUSE/AUROC/coverage and depth-bin consistency.
- I-B: geometry/right-view gain over opacity and fixed-filter controls.
- I-C: depth/normal/edge/outlier plus left/right masked/full rendering guardrails.
- I-D: latency mean/median/p90, FPS, peak memory, and overhead decomposition.

## Acceptance criteria

- Every applicable subcondition has complete, traceable evidence and a predeclared decision.
- PASS freezes exactly one loadable Phase I artifact and raw-evidence schema.
- A learned failure with proxy success yields an explicitly proxy-based artifact, not a hidden substitution.
- Covariance without headroom cannot pass the two-covariance claim.
- Phase II authorization is explicit and machine/readme visible.

## Failure conditions

- I-A has no useful signal; I-B lacks covariance headroom; I-C shows only misleading photometric gain or unacceptable harm; I-D exceeds the research feasibility bound without an approved simpler path; or evidence/provenance is incomplete.

## Pivot / rollback path

- Learned fail/proxy pass: freeze proxy-based Phase I.
- Covariance fail: pivot to calibrated confidence-aware GS and do not claim two-covariance success.
- Cross-view fail: remove/limit the term and claim.
- Feasibility fail: evaluate rank-1/mixed-precision or opacity-only fallback as a new candidate through the same gate.
- No acceptable representation: stop before Phase II and publish the negative analysis.

## Artifact/provenance requirements

The Phase I artifact records baseline and uncertainty identities, checkpoint, resolved config/hash, split hash, contract/feature/formula schemas, calibration, renderer/backend, code/upstream commits, seeds, environment/hardware, all gate metrics, decision, limitations, and payload hashes. No mutable checkpoint aliases.

## Completion checklist

- [ ] Gate criteria were frozen before final held-out evaluation.
- [ ] I-A through I-D evidence is complete.
- [ ] Negative results and pivots are preserved.
- [ ] Decision explicitly authorizes or blocks Phase II.
- [ ] PASS artifact loads immutably with validated lineage.
- [ ] No Phase II code was created.

## Handoff to next plan

Only a Gate I PASS/approved reduced Phase I artifact authorizes Plan 10. Plan 10 may assume frozen Phase I state/evidence semantics and must not fine-tune or redefine them.
