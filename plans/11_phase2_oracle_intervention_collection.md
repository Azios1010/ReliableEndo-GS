# Plan 11 — Phase II Oracle Intervention Collection

## Status

PLANNED

## Research stage

Phase II

## Recommended executor

Sol. This is compute/data-intensive counterfactual collection where state identity, action independence, utility correctness, resumability, and leakage control are scientifically critical.

## Estimated implementation risk

HIGH. Compute/data risk dominates: three production action renders per region can be expensive, and partial/resumed collection can silently mix states, costs, or code versions.

## Objective

Build a resumable, deterministic intervention dataset by applying STOP, stereo repair, and Gaussian repair independently to the same immutable Phase I region state and measuring quality change and device cost.

## Scientific question

What is the measured counterfactual utility of each available repair for each region, and does the data contain heterogeneous source-specific headroom?

## Why this milestone exists

Router labels must come from real downstream consequences, not difficulty heuristics. Collection must precede Gate II and use exactly the action code intended for deployment.

## Prerequisites

- Plan 10 action implementations/checkpoints and region schema frozen.
- Immutable Phase I artifact and training/validation collection split.
- Frozen metric, profiling, and utility protocol.

## Inputs

- Serialized or reproducibly reconstructable `state_0` and raw `RegionState`.
- Action registry containing exact STOP/stereo/Gaussian implementations.
- Quality metrics, measured-cost profiler, budgets/lambda grid defined without test tuning.

## Outputs

- Versioned intervention-table schema and validated immutable oracle artifact.
- Deterministic intervention IDs, shards, completion index, failure records, and resumable collector.
- Per-action before/after outputs, `Delta Q`, measured cost, and utility.
- Collection QA report; no router yet.

## Scope

- For each selected region, restore the same `state_0` before each action; never chain actions for one-step labels.
- Measure local and global rendering/geometry consequences with declared masks.
- Profile real latency/memory on target hardware using a frozen synchronization/warm-up protocol.
- Include STOP as a measured zero-change reference.
- Store pre-action raw evidence separately from post-action outcomes.
- Make collection sharded, atomic, resumable, duplicate-safe, and schema validated.
- Choose a tabular backend only during implementation after row/tensor size and access-pattern benchmarks; preserve backend-neutral schema semantics.

## Non-goals

- No router features from post-action outputs, router training, action chaining, joint action tuning, test-set labels, or manual deletion of failures.
- Oracle is an upper-bound/supervisory artifact, not deployable performance.

## Files expected to be created

- `src/reliable_endo_gs/oracle/__init__.py`
- `src/reliable_endo_gs/oracle/schema.py`
- `src/reliable_endo_gs/oracle/collector.py`
- `src/reliable_endo_gs/oracle/identity.py`
- `src/reliable_endo_gs/oracle/utility.py`
- `src/reliable_endo_gs/oracle/storage.py`
- `configs/oracle/interventions.yaml`
- `configs/experiment/p2b_oracle_collection.yaml`
- `scripts/collect_oracle_interventions.py`
- `scripts/validate_oracle_artifact.py`
- `tests/oracle/test_identity.py`
- `tests/oracle/test_utility.py`
- `tests/oracle/test_collector.py`
- `tests/oracle/test_resume.py`
- `tests/integration/test_oracle_collection_smoke.py`

## Files expected to be modified

- Artifact manifest/schema for oracle lineage and shard hashes.
- Profiling code for action-scoped measured costs.
- Action registry only for stable serialization/identity, not new mechanics.

## Interfaces and contracts

Each row includes sample/sequence/region IDs, Phase I artifact ID, region schema, raw state evidence or immutable reference, action ID/version, before/after metrics, `Delta Q`, measured cost/context, utility/version, validity/failure, changed fields, and provenance. Intervention ID is a deterministic hash of scientific identity fields, not path/time.

Collector accepts an immutable state plus action set, deep/restores state per action, validates no mutation, writes a staged row atomically, and can resume without recomputing validated rows or accepting incompatible ones.

## Scientific formulation

Owned only by `oracle/utility.py`:

```text
Delta Q_i(a) = Q_i(after a) - Q_i(before)
U_i(a) = Delta Q_i(a) - lambda C_i(a)
a_i^oracle = argmax_a U_i(a)
```

Measured counterfactual utility across the two repair sources is part of the proposed Phase II method/evidence. The linear utility form is an explicit design choice, not a claim of universal utility. Tests verify sign, units, ties, infeasible actions, STOP, and argmax.

## Numerical and coordinate conventions

Quality direction must be normalized explicitly so positive `Delta Q` means improvement. Cost units/hardware are fixed per comparable table or converted by a declared model. Handle failed actions, missing metrics, NaN/Inf, empty masks, and timing outliers explicitly. No latency claim without device synchronization.

## Configuration changes

Define Phase I/action artifact IDs, collection split, sampling/region policy, quality composition, local/global weights, lambda/budget grid, profiling protocol, repetitions, shard size, resume rules, and output root. Avoid selecting a storage library in the planning file.

## Tests

### Unit

- Utility sign, cost penalty, ties, STOP, invalid actions, and argmax.
- Deterministic IDs under field ordering and changed IDs under semantic changes.
- Atomic write, checksum, duplicate, corruption, and resume behavior.

### Contract

- Required row fields, schemas, lineage, cost units, and raw-versus-post evidence separation.

### Integration

- All actions start from byte/semantic-equivalent `state_0`; deployment registry identity equals collection identity.

### Smoke

- Interrupt and resume a tiny multi-region collection without duplicates or mixed schemas.

### Regression

- Synthetic action outcomes and known oracle labels/utilities.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/oracle tests/actions tests/integration
pytest -q
python -m compileall src
python scripts/validate_oracle_artifact.py --artifact <immutable-id>
git diff --check
```

## Experiments

Collect approved training/validation interventions across sequences, regions, failure modes, and a frozen budget/lambda grid. Run pilot throughput/storage QA before full collection; the pilot may freeze engineering parameters but cannot inspect final test outcomes.

## Metrics

- Rows/actions per second, completion and failure rates, duplicates/corruption.
- Local/global `Delta Q` and utility distributions.
- Latency mean/median/p90 and peak memory by action/region.
- STOP/stereo/Gaussian provisional support by sequence/failure mode.

## Acceptance criteria

- Every action for a comparison starts from identical state and exact deployed implementation identity.
- Intervention IDs and resumed outputs are deterministic; artifact shards validate.
- Before/after metrics, measured cost, utility, failures, and lineage are complete.
- No post-action feature enters pre-action router state.
- Dataset is sufficient for the predeclared Gate II analysis or transparently reports coverage limits.

## Failure conditions

- State reset/action identity cannot be guaranteed, collection mixes incompatible versions, costs are not reproducible, failure/missingness is systematic and hidden, or storage cannot be validated/resumed.

## Pivot / rollback path

Reduce sampled regions, use coarser grids, batch actions, or narrow the action set while preserving identical-state semantics. Recollect from scratch under a new artifact identity after any action/utility/schema change. Never patch rows manually.

## Artifact/provenance requirements

Record Phase I/action/checkpoint identities, region/raw-evidence/action/utility/metric schemas, split/sample selection, code/config hashes, seeds, renderer, hardware/profiling context, shard hashes, completion/failure ledger, and storage backend/version.

## Completion checklist

- [ ] Identical starting-state invariant is tested.
- [ ] Same deployment actions are used.
- [ ] Schema includes all required identity, outcome, cost, and failure fields.
- [ ] Utility and oracle argmax are verified.
- [ ] Collection is atomic, deterministic, and resumable.
- [ ] Immutable oracle artifact validates.

## Handoff to next plan

Plan 12 may assume one validated oracle intervention artifact with frozen utility/cost semantics. It may analyze but not rewrite outcomes or tune action mechanics on the final analysis split.
