# Execution pipeline

## Scope

This document specifies how implemented infrastructure and planned scientific components will be composed. It does not create experiment plans, training code, tensor loaders, baseline integration, or scientific packages.

## End-to-end runtime

The intended runtime sequence is:

```text
Stereo pair
  -> baseline stereo prediction
  -> disparity and intermediate evidence
  -> Phase I uncertainty
  -> depth and center geometry
  -> Gaussian attributes
  -> surface covariance + center covariance -> effective covariance
  -> rendering and cross-view evidence
  -> Phase I ReconstructionState
  -> regions
  -> raw RiskRoute evidence
  -> router feature encoding
  -> budget-aware router decision
  -> STOP | stereo repair | Gaussian repair
  -> final Gaussian field
  -> final render
```

Phase I terminates at `ReconstructionState` and must execute independently. Everything after that boundary is optional Phase II behavior. The runtime coordinator, not the router, invokes the selected action and accounts for budget. Detailed diagrams are in `docs/architecture.md`.

## Training flows

Training and calibration are separate stage-specific programs, not modes of one monolithic trainer.

### Baseline reproduction

Inputs are the pinned upstream source/checkpoint, an explicit dataset and split, and a baseline reproduction config. Outputs are normalized predictions, metrics, profiling evidence, and a baseline artifact. This flow establishes preprocessing, disparity, camera, and renderer semantics before project-specific models are trained.

### Phase I uncertainty

This flow consumes an accepted baseline artifact. It first evaluates proxy/oracle uncertainty and establishes calibration targets. Learned uncertainty is introduced only with an explicit target, calibration protocol, and comparison to simple proxies. The flow cannot import actions, oracle intervention labels, router features, or budgets.

### Phase I probabilistic Gaussian representation

This flow consumes accepted baseline/uncertainty outputs and implements geometry, center covariance, surface covariance, marginalization, and rendering experiments. It produces the evidence for Gate I. A promoted Phase I artifact is frozen before Phase II data generation.

### Phase II action training and measurement

Each action is developed and evaluated independently against a frozen Phase I artifact. The action code used here is the same implementation later called by the oracle and runtime. Action training, if required, cannot update the frozen Phase I artifact in the MVP.

### Oracle intervention generation

For each eligible state/region/budget, the oracle executes the supported actions, measures post-action quality and actual cost, and computes utility through the owned formula. It produces a versioned oracle intervention artifact and Gate II headroom analysis. The oracle never recreates repair logic.

### Router training

Router training starts only after Gate II. It consumes versioned raw evidence, an explicit feature encoder, oracle labels, action identities, and budget semantics. It trains selection behavior only; no action mechanics or Phase I updates occur inside this flow.

### Matched-cost evaluation

Final evaluation compares STOP, always-repair policies, individual actions, oracle upper bounds, learned routing, and other approved baselines at matched measured cost or a justified calibrated budget. It reads immutable artifacts and produces a report artifact without mutating them.

## Execution modes

### Development mode

Development mode is fast, local, and diagnostically rich. It may use synthetic data, tiny checked fixtures, deterministic CPU reference components, reduced samples, and mocked external boundaries. It guarantees contract validation, provenance capture, and explicit labeling as development output. It does not support scientific performance claims.

### Research experiment mode

Research experiment mode executes an approved comparison on external data and supported hardware. It requires resolved configuration, pinned input artifacts, explicit split identity, seeds, measured resource context, and finalized output manifests. It may optimize throughput but cannot relax scientific contract or provenance checks. Partial/failed runs remain distinguishable from promoted results.

### Reproduction mode

Reproduction mode reconstructs a named artifact or report from immutable identities. It rejects dirty or incompatible dependencies unless the reproduction protocol explicitly records and permits them. It guarantees that configurations, splits, upstream revisions, checkpoint hashes, schemas, and metric definitions match the target evidence, or reports the precise mismatch. It never resolves an unnamed “latest” artifact.

## Configuration responsibilities

Current typed configuration and hashing infrastructure remains the foundation. Future configuration files should be divided by responsibility:

| Future config area | Responsibility |
| --- | --- |
| `configs/data/` | Dataset registry key, split manifest, selection, and loader parameters. |
| `configs/baseline/` | Pinned baseline adapter/checkpoint behavior and normalized evidence selection. |
| `configs/uncertainty/` | Proxy/learned estimator, calibration, loss, and thresholds. |
| `configs/geometry/` | Verified conventions and numerical stabilization parameters. |
| `configs/probabilistic_gs/` | Gaussian attributes and covariance/marginalization behavior. |
| `configs/rendering/` | Backend and render request parameters. |
| `configs/regions/` | Region construction and evidence aggregation. |
| `configs/actions/` | One reusable config per action implementation. |
| `configs/oracle/` | Intervention grid, metrics, measured-cost policy, and utility. |
| `configs/routing/` | Feature schema, router, action set, and budget semantics. |
| `configs/training/` | Stage-specific optimizer, schedule, checkpoint, and reproducibility behavior. |
| `configs/evaluation/` | Metrics, comparisons, aggregation, and reporting. |
| `configs/experiment/` | One thin config per approved scientific comparison. |

Component configs are reusable and own component behavior. An experiment config selects and overrides components for one comparison; it must not become a second implementation of their semantics. A configuration-composition framework is intentionally not introduced now. The existing loader may be extended only when an implementation milestone demonstrates a concrete need.

## Data flow and external storage

The implemented data control plane remains authoritative:

```text
RELIABLE_ENDO_DATA_ROOT -> dataset YAML -> resolver -> registry
                           -> adapter -> split manifest
                           -> future sample index / tensor loader
```

Future sample indexes and loaders extend that path rather than bypassing it. Dataset roots come from `RELIABLE_ENDO_DATA_ROOT`; raw datasets, large derived data, and checkpoints remain external. Sequence/patient grouping is preserved before split assignment. See `docs/data_architecture.md` for adapter and split semantics.

## Run initialization and provenance

Every executable flow initializes through the implemented runtime foundation and records a resolved configuration snapshot, configuration hash, Git metadata, run identity, mode, and logs. Scientific stages add data/artifact/schema/backend identities required by `docs/artifact_lifecycle.md`.

A downstream flow names every upstream artifact explicitly. A stage does not discover scientific input by scanning a directory for the newest timestamp.

## Gate enforcement

Gate checks occur at artifact promotion boundaries:

- baseline reproduction blocks Phase I promotion when reference behavior or conventions are unresolved;
- Gate I blocks freezing a Phase I artifact when uncertainty/probabilistic reconstruction criteria are unmet;
- a frozen Phase I identity is required for action and oracle work;
- Gate II blocks router training when measured interventions have insufficient cost-aware oracle headroom;
- final claims require matched-cost evaluation and complete lineage.

Failing a gate produces a reportable artifact and triggers the approved rollback/pivot analysis. It does not silently weaken the gate or start a dependent stage.

## Failure and pivot paths

- Learned uncertainty failure falls back to the strongest calibrated proxy/oracle representation while retaining the same Phase I contract.
- Lack of covariance headroom can remove center-covariance use from the promoted representation while preserving the baseline Gaussian path and reporting the negative ablation.
- Weak Gaussian-repair utility can reduce Phase II to STOP and stereo repair because actions are independently registered.
- Weak router performance can stop at the oracle artifact; the intervention upper bound and action analysis remain valid outputs.

Every pivot produces a new configuration/artifact identity and is reported as a design decision. Pivots do not rewrite earlier results.
