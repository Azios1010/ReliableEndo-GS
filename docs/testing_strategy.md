# Testing strategy

## Purpose and current scope

ReliableEndo-GS uses tests to protect software correctness, scientific semantics, and reproducibility separately. The current CPU-safe infrastructure and data tests remain unchanged. The scientific tests described below are **PLANNED** and are added only with their owning implementation milestones.

## Test layers

### Unit tests

Unit tests cover one owning function or class with small deterministic inputs. Planned examples include analytic disparity/depth conversion, camera backprojection, covariance construction and propagation, mask handling, utility computation, budget accounting, and configuration validation.

Formula tests use hand-computable cases, dimensional checks, symmetry/positive-semidefinite checks where mathematically required, and finite-difference or independent reference calculations where useful. They do not duplicate the production formula line for line.

### Contract tests

Contract tests validate public shapes, dtypes, units, masks, coordinate metadata, optional-field behavior, immutability, schema compatibility, and provenance requirements. Each producer is tested against the contract it publishes and each consumer against supported versions.

The baseline adapter receives special contract tests so third-party types cannot leak downstream. Renderer backends share contract cases for their supported subset. Action results must enumerate changed fields and leave input states unchanged.

### Integration tests

Integration tests join a small number of real components across an intentional boundary. Planned examples are:

- dataset adapter -> sample index -> loader contract;
- baseline adapter -> `StereoPrediction`;
- uncertainty -> geometry -> Gaussian field;
- Gaussian field -> deterministic reference renderer;
- frozen reconstruction state -> region evidence;
- runtime coordinator -> router decision -> registered action;
- action -> profiler -> oracle utility;
- artifact writer -> validator -> downstream consumer.

Integration tests use tiny synthetic or licensed fixtures and avoid requiring full external datasets in default CI.

### Smoke tests

Smoke tests answer whether a supported entry point can complete a minimal execution with valid outputs and provenance. CPU CI uses synthetic/tiny configurations. GPU smoke tests run in a separately declared environment when available and are never silently skipped while reporting GPU validation as successful.

A Phase I smoke test must end at `ReconstructionState` without importing Phase II. A Phase II smoke test starts from a frozen fixture artifact and executes STOP plus available repair actions through the same runtime action registry.

### Regression tests

Regression tests protect previously verified semantics, not arbitrary floating-point bytes. They use versioned golden fixtures and declared tolerances for baseline-normalized outputs, geometry, reference rendering, artifact manifests, cost-accounting logic, and selected end-to-end metrics.

Updating a golden value requires a reviewed explanation of the semantic or numerical change. A changed upstream commit, schema, formula, stabilization threshold, or backend produces new fixtures rather than silently replacing old evidence.

### Experiment validation

Experiment validation checks scientific comparisons beyond ordinary software tests:

- fixed dataset and split-manifest identities;
- separation of training, validation, calibration, oracle-label generation, and test use;
- baseline reproduction within approved tolerance;
- seeds and run counts sufficient for the claim;
- identical metrics and preprocessing across compared methods;
- measured or calibrated matched-cost budgets;
- complete failures, exclusions, validity counts, and confidence summaries;
- artifact lineage and schema compatibility;
- leakage checks between Phase I, oracle targets, router features, and held-out evaluation;
- gate criteria evaluated exactly as approved.

Experiment validation can fail a promotion even when all unit tests pass.

## Numerical safety matrix

Numerical behavior is owned by the component that introduces the quantity. The default is explicit rejection, masking, or a diagnosed stabilization path—not an unreported clamp.

| Condition | Expected test behavior |
| --- | --- |
| Invalid, zero, or near-zero disparity | Verify configured validity threshold, no uncontrolled depth division, correct mask/count diagnostics. |
| NaN or infinity | Verify detection at the producer/consumer boundary and propagation as an explicit failure or invalid entry. |
| Singular or indefinite covariance | Verify the owning module's rejection or named regularization policy and record affected counts. |
| Nonpositive Gaussian scales | Verify parameterization prevents them or contract validation rejects them. |
| Invalid opacity/logit | Verify range or parameterization semantics and explicit failure behavior. |
| Empty mask or zero valid elements | Verify a defined empty result or clear error; never a misleading aggregate. |
| Out-of-view projection | Verify visibility masking and absence of invalid memory/index access. |
| Degenerate/empty regions | Verify router/action eligibility and STOP behavior are deterministic. |

Stabilization constants are configuration values owned next to the formula. Tests cover values below, at, and above their boundary. Diagnostics record how often stabilization occurs so it cannot become a hidden model behavior.

## Geometry test policy

Geometry tests start with analytic cameras:

- identity and known rigid transforms;
- known focal length, principal point, and metric stereo baseline;
- points on the optical axis and off-axis;
- hand-computable disparity/depth pairs;
- known uncertainty Jacobians and covariance orientation;
- round-trip checks under declared tolerances.

Dataset-derived cases are added only after conventions are verified independently. A test that merely reproduces the baseline's output does not prove that its convention is correct.

## Renderer test policy

A future deterministic CPU reference backend serves small correctness tests. The production GPU backend is checked against the same contract and selected reference scenes within declared tolerances. CPU timing is not used to predict GPU performance, and the reference renderer is not a scientific performance baseline unless explicitly justified.

Renderer tests declare whether they consume surface or effective covariance, and verify that changing center uncertainty has the expected effect only when the chosen rendering semantics say it should.

## Action, oracle, and routing tests

The action registry is tested independently from routing. For every action:

- direct, oracle, routed, evaluation, and profiling paths resolve the same implementation identity;
- declared affected fields change and prohibited fields do not;
- input states remain unchanged;
- failure and ineligible-region behavior is explicit;
- estimated and measured costs use declared units and contexts.

Oracle tests independently recompute small quality-gain/cost examples and confirm that STOP participates in comparison. Router tests verify that decisions respect supported actions and budgets, but do not test repair mechanics inside router code. Leakage tests ensure oracle-selected targets are not present in inference features.

## Artifact and provenance tests

Artifact tests verify content hashes, immutable finalization, required metadata, upstream lineage, schema rejection/migration, incomplete-write handling, and refusal of mutable `latest` references. Reproduction tests must detect changed configs, dirty upstream code, split mismatches, and missing payloads.

## Data tests

The existing data tests remain the foundation for registry parsing, environment-root resolution, path safety, adapter discovery, and deterministic split manifests. Future loader tests use small fixtures and preserve sequence/patient grouping. Default CI does not require the private server data root and must not log patient-identifying paths or contents.

## Failure-oriented scientific tests

The architecture must remain executable under negative outcomes:

- learned uncertainty can be disabled in favor of a calibrated proxy;
- center covariance can be ablated without changing surface covariance semantics;
- Gaussian repair can be absent while STOP and stereo repair remain valid actions;
- router training can be skipped while oracle headroom and action reports remain reproducible.

Tests cover these configurations so a failed hypothesis does not force unrelated code paths or artifact schemas to break.

## Continuous integration tiers

Default CPU CI runs formatting, linting, static typing, unit/contract tests, minimal integration/smoke tests, and source compilation without external datasets or GPU dependencies. Later GPU CI may add production renderer/baseline smoke checks behind an explicit environment marker. Long experiments and statistical gate evaluations run as tracked research workflows and publish artifacts; they are not disguised as ordinary CI tests.

The repository-level validation command set remains:

```text
ruff check .
ruff format --check .
mypy src
pytest -q
python -m compileall src
git diff --check
git status --short
```
