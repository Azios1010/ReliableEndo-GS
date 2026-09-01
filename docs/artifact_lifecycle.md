# Artifact lifecycle and provenance

## Purpose

Scientific stages communicate through immutable, identified artifacts rather than path conventions or an implicit “latest” checkpoint. This makes stage freezing enforceable and allows negative results, reruns, and downstream decisions to be traced to exact evidence.

The development-only baseline artifact schema is implemented under
`reliable_endo_gs.evaluation.provenance`. Its `scientific_status` is
`development_only` and it cannot represent an accepted baseline until the
scientific reproduction gate is satisfied. Phase I, Phase II, and report
artifacts remain planned. Existing run metadata and configuration hashing
provide the shared infrastructure.

## Dependency graph

```mermaid
flowchart LR
    B[Baseline artifact] --> P[Phase I artifact]
    P --> O[Oracle intervention artifact]
    O --> R[Router artifact]
    R --> F[Final evaluation / report artifact]
```

The arrows mean “records and depends on the immutable identity of.” Dependencies are transitive, so the final report can trace the full chain. A report may intentionally stop at an earlier stage when a gate fails; it then records the available upstream identities, the failed gate, and excludes unsupported downstream claims.

## Artifact identity

An artifact identity is content-addressed or equivalently immutable and includes a stable artifact type, schema version, and digest. Human-friendly names may point to an identity for convenience, but they cannot replace it in provenance.

Every artifact manifest records:

- artifact type, schema name, schema version, creation time, and immutable ID;
- ReliableEndo-GS code revision and dirty state;
- complete resolved configuration and configuration hash;
- data registry entry, split-manifest identity, and sample-selection policy;
- random seeds and determinism settings;
- hardware, environment, and relevant backend versions;
- all upstream artifact identities and their expected schemas;
- produced payload files with hashes and sizes;
- metrics with definitions, aggregation policy, and validity counts;
- warnings, numerical stabilization counts, and known limitations;
- gate decision and the evidence used for it, when applicable.

An artifact is written to a staging location, validated, and then finalized under its identity. Finalized payloads are never overwritten. Failed or incomplete creation remains visibly incomplete and cannot be consumed as a promoted artifact.

The names `latest.pt`, `best.pt`, or similar mutable filenames are not valid dependency references. A convenience pointer called “best” may exist only if it resolves to an immutable artifact and records the selection rule and comparison set.

## Baseline artifact

The baseline artifact establishes reproduced Endo-E2E-GS behavior before project-specific scientific changes.

It contains or references:

- official upstream and optional fork URLs;
- pinned upstream commit and patch identities;
- baseline adapter and contract schema versions;
- checkpoint identity/hash and acquisition provenance;
- preprocessing, camera, disparity, renderer, and evaluation conventions;
- resolved dataset split identity;
- reproduced metrics, tolerances, profiling context, and logs;
- normalized sample outputs sufficient for contract/regression checks.

Promotion requires an approved reproduction comparison and resolved convention audit. A baseline that does not reproduce remains a diagnostic artifact and blocks Phase I promotion.

## Phase I artifact

The Phase I artifact represents the complete, independently executable reconstruction stage. It depends on one or more explicit baseline artifact identities.

It contains or references:

- uncertainty model/calibration parameters and uncertainty schema;
- geometry and coordinate-convention version;
- Gaussian representation and covariance formula versions;
- renderer/backend identity;
- Phase I model weights and configuration;
- reconstruction contract schema and representative outputs;
- uncertainty calibration, reconstruction quality, covariance ablations, and numerical diagnostics;
- Gate I criteria, measurements, and decision.

After Gate I, the promoted Phase I artifact is frozen for Phase II. Phase II inputs reference that exact identity. Any change to Phase I weights, formulas, preprocessing, or contract semantics creates a new Phase I artifact and invalidates dependent oracle/router artifacts until regenerated.

## Oracle intervention artifact

The oracle intervention artifact records measured counterfactual action outcomes against a frozen Phase I artifact. It is not just a label file.

It contains or references:

- Phase I artifact identity;
- region and raw-evidence schema versions;
- action identities, versions, and configurations;
- per-action pre/post metrics, affected regions, failures, and changed fields;
- measured cost with units, hardware, warm-up, synchronization, and repetition policy;
- utility formula/version and derived best-action labels;
- tie, infeasible-action, missing-measurement, and budget policies;
- Gate II headroom analysis and decision.

Oracle labels are regenerated if action mechanics, utility semantics, cost policy, regions, Phase I identity, or relevant evaluation metrics change. Gate II must demonstrate actionable, cost-aware oracle headroom before router training.

## Router artifact

The router artifact is trained only from an accepted oracle intervention artifact and its frozen Phase I dependency.

It contains or references:

- oracle artifact and Phase I artifact identities;
- raw evidence and encoded router-feature schema versions;
- feature normalization/encoding parameters;
- router architecture, weights, training configuration, and seed;
- supported action identities and budget representation;
- calibration, decision, and cost-prediction metrics;
- leakage audit and matched-cost validation results.

A router is incompatible if its action set, feature semantics, budget unit, or upstream artifact identity is unsupported. It cannot trigger automatic migration by field position or shape coincidence.

## Report artifact

The report artifact is an immutable evidence package for a table, figure set, paper result, or gate review. It references all contributing artifacts and includes:

- evaluation code revision and metric schema versions;
- exact comparison configurations and matched-cost protocol;
- table/figure source data, aggregation scripts or identities, and rendered outputs;
- exclusions, failed runs, seeds, confidence intervals where applicable, and negative results;
- machine-readable lineage sufficient to trace every reported value.

A final report never points to a mutable working checkpoint. If the router fails to improve on matched-cost baselines, the report may validly terminate with Phase I and oracle evidence and must say so.

## Schema-version matrix

The following semantics are versioned independently:

| Schema | Examples of breaking changes |
| --- | --- |
| Scientific contracts | Shape, units, masks, coordinate frames, covariance meaning. |
| Dataset/split manifests | Sample identity, grouping, eligibility, or partition semantics. |
| Phase I artifact | Uncertainty parameterization, geometry formula, Gaussian representation, renderer-consumed covariance. |
| Oracle intervention | Action set/mechanics, region meaning, utility, metric, or cost protocol. |
| Router features | Raw evidence definition, encoding, order, normalization, missing-value behavior. |

Project package versions can be recorded as useful metadata, but they do not replace these schema versions. Breaking semantic changes require a schema bump even when serialized files remain loadable.

## Storage and retention policy

Source control stores manifests, small fixtures, and report-ready summaries when permitted; it does not store external datasets or large checkpoints by default. Artifact storage location is environment-specific and configured explicitly. Manifests use stable logical identities so results do not depend on one machine's absolute path.

Retention favors evidence needed to reproduce promoted artifacts and gate decisions. If payload deletion is necessary, the manifest and reason remain, and the artifact is marked unavailable rather than silently redirected.

## Consumption checks

Before consuming an artifact, a stage verifies identity, payload hashes, schema compatibility, upstream lineage, data split, and required provenance. Downstream execution fails closed on a missing or conflicting field. Compatibility adapters, when justified, are named and tested transformations that produce a new manifest; they are never hidden inside a loader.
