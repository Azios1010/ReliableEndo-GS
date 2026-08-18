# Third-party Endo-E2E-GS integration

## Scope and current status

This document defines how the future Endo-E2E-GS baseline is incorporated without allowing upstream implementation details to become ReliableEndo-GS architecture. No repository is cloned, no upstream commit is selected, and no submodule or baseline adapter is created by this specification.

The required integration chain is:

```text
official upstream
    -> research fork, only if required
    -> pinned commit in third_party/endo_e2e_gs
    -> ReliableEndo-GS baseline adapter
    -> ReliableEndo-GS contracts
```

## Ownership boundary

Only the future `baseline` adapter may import modules from `third_party/endo_e2e_gs`. All other packages consume normalized ReliableEndo-GS contracts such as `StereoBatch`, `StereoPrediction`, `GaussianField`, and `RenderOutput`.

The adapter owns:

- translating ReliableEndo-GS inputs into the pinned baseline's input form;
- calling the supported baseline entry points;
- translating outputs into versioned ReliableEndo-GS contracts;
- declaring image normalization, resize, disparity, camera, and coordinate conventions;
- exposing only the intermediate evidence deliberately supported by the selected upstream revision;
- rejecting an unsupported or dirty upstream revision when reproducibility requires it;
- collecting upstream commit and adapter provenance;
- containing compatibility workarounds and clearly identifying any patch dependency.

The adapter does not own Phase I uncertainty semantics, covariance propagation, repair actions, oracle labels, routing, or evaluation metrics.

## Source selection and pinning workflow

The future integration milestone must perform a documented review before selecting code:

1. Identify the official upstream repository and authoritative paper release.
2. Record the upstream license, model-weight terms, dataset assumptions, citation, and attribution obligations.
3. Reproduce the supported baseline from unmodified upstream when practical.
4. Create a research fork only when changes cannot be isolated in a wrapper or local patch.
5. Pin one reviewed commit through a Git submodule or an equivalently immutable mechanism approved for the repository.
6. Record the pinned commit, origin URL, fork relationship, and dirty-state check in every dependent artifact.

There is no floating branch dependency. Tags alone are insufficient unless resolved and recorded as an immutable commit. Updating the pin is a reviewed change with reproduction evidence, adapter contract tests, and artifact compatibility analysis.

The choice between a direct upstream submodule and a research-fork submodule remains deferred until the baseline API and required changes are inspected. This document does not guess the commit.

## License and attribution

Before third-party code or weights enter the repository workflow, the integration milestone must verify that their licenses permit the intended research use and redistribution behavior. Required notices and citations remain adjacent to the integration metadata and in resulting reports where appropriate.

Dataset access terms are independent from source-code terms. A usable code license does not authorize committing datasets, weights, derived patient data, or artifacts. Any ambiguity is resolved before distribution.

## Patch policy

Wrappers and adapters are preferred because they preserve a clean upstream history. If a patch is unavoidable, it is stored under the future directory:

```text
patches/endo_e2e_gs/
```

Each patch must be minimal and contain or accompany:

- the exact upstream commit to which it applies;
- a stable patch identifier;
- the scientific or operational rationale;
- a description of changed behavior;
- a focused test or reproduction check;
- instructions for applying and verifying it;
- a note explaining why an adapter-only solution was insufficient.

Patches cannot silently alter baseline metrics, preprocessing, camera semantics, or evaluation behavior. Patched baseline reproduction is reported separately from an official unmodified baseline when the change affects scientific behavior.

The integration test matrix retains an unmodified upstream baseline path wherever the selected revision supports it. This distinguishes adapter correctness from behavior introduced by a research patch.

## Preventing source-copy drift

Copying selected upstream source files into `src/reliableendo_gs/` is prohibited. It obscures origin, makes upstream comparison unreliable, and encourages local edits without an auditable patch history. Likewise, duplicating baseline formulas or preprocessing in downstream packages is prohibited when the adapter can expose the normalized result.

Vendored generated binaries, checkpoints, and caches are not source integration. Their acquisition, hashes, license terms, and storage location must be handled as external artifact dependencies.

## Provenance requirements

Every baseline-dependent run and artifact records at least:

- official upstream URL;
- research-fork URL when applicable;
- pinned commit SHA;
- submodule or dependency dirty state;
- patch identifiers and hashes, or an explicit `none`;
- baseline configuration and checkpoint identity/hash;
- ReliableEndo-GS adapter version and contract schema version;
- ReliableEndo-GS code revision and configuration hash;
- relevant environment/backend versions.

An output that lacks the upstream commit cannot be promoted into a Phase I, oracle, router, or report artifact.

## Validation before scientific use

The future integration milestone is accepted only after:

- upstream installation and license requirements are documented;
- a pinned revision is reproducible in the supported environment;
- the reference baseline metric is reproduced within an approved tolerance;
- adapter contract tests cover shapes, masks, conventions, and optional evidence;
- a direct-import audit shows no third-party imports outside the adapter boundary;
- clean and dirty upstream states are represented correctly in provenance;
- CPU-safe contract tests do not claim to validate GPU numerical performance.

Baseline integration precedes uncertainty, geometry, probabilistic Gaussian, action, oracle, or routing implementation.
