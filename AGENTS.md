# ReliableEndo-GS Agent Instructions

## Project identity

- Project: ReliableEndo-GS
- Domain: feed-forward stereo endoscopic Gaussian Splatting
- Research program: ProbStereo-EndoGS -> RiskRoute-GS

ProbStereo-EndoGS establishes a reliable uncertainty-aware representation. RiskRoute-GS may use the frozen representation for selective repair and compute allocation.

## Source of truth

Use this precedence order:

1. Explicit task instruction from the user
2. Approved research proposal
3. Repository governance and scientific rules
4. Approved architecture documentation
5. Approved implementation plans
6. Experiment configurations
7. Code and tests
8. Comments and historical notes

When sources conflict, the higher-priority source wins. Never silently change the research formulation to resolve a conflict. Identify the conflict, preserve the higher-priority source, make the smallest compatible change, and disclose the conflict in the completion report.

The proposal contains hypotheses, expected contributions, candidate methods, gates, and pivots. These are research intentions, not established results. Before evidence exists, write "the proposal hypothesizes that stereo-derived center covariance may improve geometry," not "the covariance improves geometry."

## Program order and stage isolation

Do not reinterpret this execution order:

```text
baseline reproduction
-> Phase I proxy/oracle uncertainty
-> Phase I full implementation
-> Gate I
-> freeze Phase I artifact
-> Phase II repair actions
-> Phase II oracle interventions
-> Gate II
-> router
-> matched-cost final evaluation
```

### Baseline

Baseline reproduction must remain scientifically unmodified. Research changes must not be silently introduced into baseline code.

### Phase I: ProbStereo-EndoGS

Phase I owns stereo uncertainty, calibration, geometry uncertainty propagation, center covariance, surface-support covariance, probabilistic Gaussian representation, cross-view supervision, and Phase I evaluation.

Phase I must not depend on repair actions, oracle routing labels, router predictions, a budget allocator, or a Phase II loss.

### Phase II: RiskRoute-GS

Phase II may consume frozen Phase I outputs such as `sigma_d`, `sigma_D`, center-covariance statistics, cross-view residuals, visibility, validity masks, and Gaussian diagnostics. It must not change Phase I scientific semantics to make routing easier.

### Gates

- Gate I: Phase II must not become the default research path until an approved Phase I artifact is frozen. Existing Phase I code does not imply that Gate I passed.
- Gate II: do not implement or train the final dual-source router until oracle interventions show meaningful heterogeneous utility among STOP, stereo repair, and Gaussian repair.
- If a gate fails, follow the approved simplification or pivot. Do not force the original hypothesis.

## Architecture rules

The conceptual dependency direction is:

```text
contracts + config
-> data
-> baseline
-> uncertainty
-> geometry
-> probabilistic_gs
-> rendering
-> regions
-> actions
-> oracle
-> routing
-> evaluation + profiling
```

This graph describes future boundaries; it does not authorize empty package creation.

Dependency constraints:

- Third-party isolation: only the future baseline integration layer may directly depend on Endo-E2E-GS internals. Prefer `third_party/endo_e2e_gs -> baseline adapter -> ReliableEndo-GS contracts`. Core packages must not import from `third_party`.
- Actions are independent of routing. One action implementation must serve oracle intervention, training, inference, and profiling.
- Oracle code evaluates the real action implementations; it must not maintain private duplicate repairs.
- Evaluation must not mutate scientific model state.
- Files under `scripts/` are thin entry points. Scientific logic belongs under `src/reliable_endo_gs/`.

Create a scientific package only when a real implementation task requires it. Do not pre-create empty `actions`, `oracle`, `routing`, or other future packages.

Use stable concept names such as `uncertainty`, `geometry`, `probabilistic_gs`, `rendering`, `actions`, `oracle`, `routing`, and `profiling`. Avoid chronology or throwaway names such as `phase1_new`, `model_v2`, `new_model`, `final_final`, `experiment_new`, `fix2`, and `temporary_final`. Phase names belong in plans, experiment configurations, reports, and documentation, not in the core source layout.

## Scientific formula ownership

Each scientific transformation has one canonical implementation. This includes disparity to depth, depth to 3D center, disparity uncertainty to center covariance, surface covariance construction, and utility computation. Other modules call that implementation. Document numerical approximations beside the canonical formula.

## Data governance

Real datasets live outside the repository. Resolve future locations through configuration or environment variables such as `RELIABLE_ENDO_DATA_ROOT`; never embed developer-specific absolute paths.

Do not commit raw datasets, copied subsets, or private medical data; download datasets during import; or silently change splits. Dataset adapters, schemas, configurations, and immutable split manifests may be committed when explicitly requested.

## Experiment integrity and evidence

Future experiments must be traceable to the Git commit, resolved configuration and hash, dataset split and hash, seed, checkpoint/artifact, and relevant hardware metadata. Do not overwrite successful outputs by default or manually edit generated metrics. Derived reports must retain provenance.

Efficiency claims require measured runtime evidence. FLOPs, parameter count, selected regions, and processed pixels are insufficient alone. Account for the base model, uncertainty, routing, region extraction, repairs, rendering, and memory; record hardware for latency measurements.

Preserve required ablations and matched comparisons. At minimum, future Phase I work must remain able to compare baseline, scalar confidence weighting, opacity-only, fixed/isotropic filtering, stereo-derived covariance, and cross-view off/on. Future Phase II work must remain able to compare baseline, fixed refinement, uncertainty thresholding, stereo-only routing, GS-only repair, dual-source routing, and oracle performance.

## Coding rules

Future Python code must use Python 3.10+, type annotations for public APIs, `pathlib.Path` for paths, and explicit tensor-shape documentation.

Do not use wildcard imports, mutable global scientific state, import-time GPU initialization, import-time dataset access, `sys.path.append` hacks, silent exception swallowing, hidden machine constants, duplicated scientific formulas, giant monolithic training scripts, or scientific logic that exists only in notebooks. Prefer clear code over clever abstractions.

## Change discipline and execution protocol

Make the smallest coherent change required by the task. Do not perform unrelated refactors or opportunistically implement later proposal stages. Before changing an established interface, search for all consumers.

Every implementation task follows:

1. Inspect the repository.
2. Understand current contracts and documentation.
3. Identify the task boundary.
4. Make the minimal coherent implementation.
5. Add or update tests.
6. Run relevant validation.
7. Inspect the Git diff and status.
8. Report exactly what changed.

Completion reports must distinguish implemented, tested, not tested, blocked, and planned. Never fabricate test results, metrics, dataset availability, checkpoint compatibility, or experiment completion; never silently skip failed validation.

Negative results are valid. Preserve gate failures and negative evidence. If oracle covariance does not beat opacity-only, Gaussian repair almost never wins, or router overhead exceeds savings, reconsider the relevant claim and follow an approved pivot instead of distorting the architecture.

## Conceptual file ownership

Most scientific source areas below are planned and must not be created until requested.

| Area | Responsibility | Status |
| --- | --- | --- |
| `src/reliable_endo_gs/contracts/` | Shared scientific and data interfaces | Planned |
| `src/reliable_endo_gs/data/` | Dataset identity, paths, split contracts, and adapter scaffolds; root `data/` is runtime-only | Active: no parsing/tensors |
| `src/reliable_endo_gs/baseline/` | Upstream Endo-E2E-GS isolation | Planned |
| `src/reliable_endo_gs/uncertainty/` | Uncertainty estimation and calibration | Planned |
| `src/reliable_endo_gs/geometry/` | Camera and stereo geometry | Planned |
| `src/reliable_endo_gs/probabilistic_gs/` | Gaussian uncertainty representation | Planned |
| `src/reliable_endo_gs/rendering/` | Renderer adapters and rendering losses | Planned |
| `src/reliable_endo_gs/regions/` | Region partition, gather, and scatter | Planned |
| `src/reliable_endo_gs/actions/` | Repair implementations | Planned |
| `src/reliable_endo_gs/oracle/` | Counterfactual intervention and evidence | Planned |
| `src/reliable_endo_gs/routing/` | Learned and deterministic policies | Planned |
| `src/reliable_endo_gs/evaluation/` | Scientific metrics and reports | Planned |
| `src/reliable_endo_gs/profiling/` | Runtime, memory, and action cost | Planned |
| `configs/` | Declarative experiment settings | Skeleton only |
| `plans/` | Approved implementation plans | Directory exists |
| `docs/` | Stable project specification | Active |
