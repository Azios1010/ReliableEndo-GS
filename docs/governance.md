# Repository Governance

## Purpose

ReliableEndo-GS is a staged research repository. Governance exists to keep implementation decisions traceable to the approved research proposal, protect experimental identifiability, and let the codebase evolve without prematurely committing to unverified hypotheses.

## Source-of-truth hierarchy

The user's explicit task has the highest authority, followed by the approved research proposal, repository governance, approved architecture, approved plans, experiment configurations, code and tests, and finally comments or historical notes. This ordering prevents stale implementation details from silently changing the scientific program.

When a conflict appears, work stops at the conflicting boundary long enough to identify it. The higher-priority source is preserved, the smallest compatible implementation change is made, and the discrepancy is reported. Scientific semantics are never reconciled silently in code.

## Research-stage isolation

The program proceeds from an honest Endo-E2E-GS baseline through ProbStereo-EndoGS and only then to RiskRoute-GS. Phase I establishes and evaluates uncertainty-aware representation semantics. Phase II consumes an approved frozen Phase I artifact to study repair decisions and computation allocation.

Gate I protects Phase II from depending on an unstable representation. Gate II protects the final router from being trained before oracle evidence demonstrates that STOP, stereo repair, and Gaussian repair have meaningfully different utility. Code availability alone cannot satisfy either gate.

## Architectural dependency rules

Dependencies flow from shared contracts and data boundaries through baseline, uncertainty, geometry, probabilistic representation, rendering, regions, actions, oracle, routing, and evaluation/profiling. This is a design direction, not a request to create every package now.

Endo-E2E-GS internals remain behind a baseline adapter. Repair actions are reusable independently of routing. Oracle analysis calls those same actions rather than forking private versions. Evaluation observes models without mutating them, and scripts delegate scientific work to importable source modules. Canonical formulas have one owner so implementations cannot drift between training, inference, and evaluation.

## Change-management protocol

Changes should be small, coherent, and limited to the active task. Established interfaces require a consumer search before modification. New packages require real behavior and a clear dependency boundary; empty placeholders are intentionally avoided.

A change report records what was implemented, validated, left untested, blocked, and merely planned. Failures and conflicts are part of the report, not conditions to hide by expanding scope.

## Agent workflow

The Research Guardian reviews scientific meaning and gates. The Repo Architect reviews boundaries and contracts. The Implementation Engineer makes and tests the approved change. The Experiment Auditor evaluates provenance and claims. Trivial work may use only the relevant role, while new scientific modules and publication evidence need broader review.

## Experiment governance

Reproducible runs retain commit identity, resolved configuration and hash, split and hash, seed, artifacts, and relevant hardware. Successful outputs are not overwritten by default, generated metrics are not hand-edited, and derived reporting keeps links to source evidence.

Comparisons preserve shared data, preprocessing, resolution, evaluation, and hardware conditions where compute is claimed. Required ablations remain possible throughout implementation. Efficiency is evaluated end to end rather than inferred from proxy quantities alone.

## Data governance

Real and private medical datasets remain outside the repository. Paths are supplied by environment or server-specific configuration, never embedded as developer-specific constants. Imports do not trigger downloads or dataset access. Split changes are explicit, versioned, and hashable.

## Release and reproduction philosophy

A release should separate available code from executed experiments, verified results, and supported claims. Reproduction material should include the information needed to reconstruct the exact comparison without exposing restricted data. Negative results and failed gates are retained because they determine valid pivots and constrain scientific claims.
