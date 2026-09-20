# Execution pipeline

## Scope

This document specifies how implemented infrastructure and planned scientific
components will be composed. Plan 03A now provides a development-only
baseline preflight, contract-level evaluation, and synchronized profiling; it
does not authorize official reproduction, training, or later scientific
packages.

The current project execution order is recorded in
[the master roadmap](../PLAN.md#current-execution-checkpoint-2026-09-20).
It requires matched patched Stage2 fixed-10k validation before the current Phase I
audit and full SCARED-C development rerun. The runtime boundaries below do not
authorize skipping that decision or entering fallback methods early.

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

The current Plan 03A entry point is explicitly `mode: development` and uses
the distinct SCARED-C development protocol. It loads and indexes the mounted
sample, then fails closed when the authorized checkpoint, native input bridge,
or pinned runtime is unavailable. Its content-addressed artifact is marked
`scientific_status: development_only` and cannot be used as an accepted
baseline dependency.

### Current deterministic foundation validation

Stage1 60k is provisionally frozen and supplies the fixed Stage2 input. The old
Stage2 10k is diagnostic only because its Gaussian render branch plateaued:
`Softplus(beta=100) -> clamp_max(0.01)` caused near-dead scale gradients. The
patched variant uses `scale = 0.01 * sigmoid(raw_scale)`. Its completed 3k run
showed partial recovery but used a 3k OneCycle schedule, so it is not accepted
or directly comparable to the old 10k control and must not be resumed.

Audit behavioral parity except scale parameterization and diagnostics. Start
a fresh patched Stage2 fixed-10k from the same Stage1 60k checkpoint, retaining
the same split, seed, loss, optimizer, and fixed-10k scheduler/training protocol.
Evaluate matched 422-frame validation with separate render/disparity metrics.
Freeze the deterministic foundation only if Stage2 is acceptable; retain the
prior artifacts and label the patched variant separately from the upstream
reference. See [Plan 03](../plans/03_baseline_reproduction.md).

### Phase I uncertainty

After patched Stage2 fixed-10k acceptance, audit/fix and harden the existing
Phase I reliability/uncertainty implementation, then run the current Phase I
once more on the full SCARED-C development protocol. Do not move directly to
probabilistic fallback methods. Keep `dataset_6` completely untouched for final
evaluation; it cannot select architecture, loss, calibration, or checkpoints.

This flow consumes the accepted deterministic artifact and retains explicit
uncertainty targets, calibration protocols, and proxy comparisons. Evaluate
mean disparity quality separately from uncertainty quality, and calibration
separately from correlation/ranking. The flow cannot import actions, oracle
intervention labels, router features, or budgets.

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
- the current full-SCARED-C development Phase I decision selects continuation or explicit predictive-uncertainty fallback; it does not itself pass Gate I or freeze a Phase I artifact;
- Gate I blocks freezing a Phase I artifact when uncertainty/probabilistic reconstruction criteria are unmet;
- a frozen Phase I identity is required for action and oracle work;
- Gate II blocks router training when measured interventions have insufficient cost-aware oracle headroom;
- final claims require matched-cost evaluation and complete lineage.

Failing a gate produces a reportable artifact and triggers the approved rollback/pivot analysis. It does not silently weaken the gate or start a dependent stage.

## Failure and pivot paths

- If the current full-SCARED-C development Phase I passes, continue that direction and evaluate downstream reliability utility. An existing validated proxy may remain the provider within this passing path.
- If it fails, stop further heuristic uncertainty-proxy search and freeze the strongest deterministic Stage1 mean predictor. Evaluate Gaussian heteroscedastic disparity uncertainty, then Laplace, against that same mean; compare calibration, NLL, coverage/width, sequence transfer, and selective prediction.
- Only if parametric uncertainty is insufficient, implement conditional residual diffusion and test its added value against the same parametric baselines. Only after uncertainty is validated, propagate disparity uncertainty -> depth -> 3D positional uncertainty -> Stage2 integration. Details are in [Phase I Fallback Directions](ReliableEndoGS_PhaseI_Fallback_Directions.md).
- Lack of covariance headroom can remove center-covariance use from the promoted representation while preserving the baseline Gaussian path and reporting the negative ablation.
- Weak Gaussian-repair utility can reduce Phase II to STOP and stereo repair because actions are independently registered.
- Weak router performance can stop at the oracle artifact; the intervention upper bound and action analysis remain valid outputs.

Every pivot produces a new configuration/artifact identity and is reported as a design decision. Pivots do not rewrite earlier results.

Throughout this sequence, positional uncertainty covariance is distinct from
Gaussian surface/support covariance, and diffusion sample variance must not be
claimed as total uncertainty. Preserve upstream geometry semantics, including
`disp2depth` and `depth2pc`, rather than changing them to improve metrics.
