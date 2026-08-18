# Plan 00 — Core Contracts and PyTorch Foundation

## Status

PLANNED

## Research stage

Infrastructure

## Recommended executor

Sol. Contract semantics, tensor invariants, dependency policy, and future CUDA separation affect every scientific stage and require architecture-level judgment.

## Estimated implementation risk

MEDIUM. The code is bounded, but an incorrect shape, coordinate, dtype, mutation, or device convention would contaminate all later evidence.

## Objective

Introduce the minimum PyTorch scientific runtime and implement the upstream-independent tensor contracts specified in `docs/contracts.md`, with deterministic CPU validation and no Endo-E2E-GS knowledge.

## Scientific question

Not a hypothesis-testing milestone.

## Why this milestone exists

The baseline adapter, dataset loader, geometry, renderer, and Phase II must exchange the same immutable meanings. Establishing these contracts first prevents third-party tensors or dataset assumptions from becoming implicit project interfaces.

## Prerequisites

- Prompt 03 infrastructure and Prompt 04 data control plane.
- Approved architecture, contract, artifact, execution, and testing specifications.
- Clean CPU validation baseline.

## Inputs

- `docs/contracts.md` field definitions and schema policy.
- Current `pyproject.toml`, CPU environment, runtime manifest, and seed utility.
- Verified supported Python versions.

## Outputs

- A versioned `contracts` package containing `CameraBatch`, `StereoBatch`, `StereoPrediction`, `GaussianField`, `RenderOutput`, and `ReconstructionState`.
- A documented tensor/device/dtype and validation policy.
- PyTorch declared as a justified runtime dependency with CPU CI support and a separately planned CUDA environment.
- Deterministic scientific seeding for Python and PyTorch.
- CPU contract tests and a small run/provenance record; no scientific artifact claim.

## Scope

- Select a PyTorch version range compatible with Python support and CPU CI; record why it was chosen.
- Keep CUDA/toolkit packages out of the CPU environment. Define a future `environments/cuda.yml` only with versions verified during implementation.
- Implement frozen or mutation-controlled contract containers with explicit batch/shape checks, optional-field checks, mask checks, finite-value policy hooks, and schema identifiers.
- Require explicit device/dtype conversion methods; constructors must not move tensors silently.
- Extend seeding to CPU PyTorch and, when available, CUDA without initializing CUDA at import time.
- Document `[B,C,H,W]`, `[B,N,*]`, boolean-mask, floating dtype, device consistency, and contiguous-memory expectations only where consumers require them.

## Non-goals

- No Endo-E2E-GS integration, dataset decoding, renderer, geometry formula, training engine, model head, CUDA kernel, or GPU requirement.
- Do not resolve still-unverified camera axes, pixel centers, disparity sign, or dataset units. Contracts carry explicit metadata or reject missing required convention identifiers.
- Do not create Phase II contracts before their owning milestone.

## Files expected to be created

- `src/reliable_endo_gs/contracts/__init__.py`
- `src/reliable_endo_gs/contracts/common.py`
- `src/reliable_endo_gs/contracts/camera.py`
- `src/reliable_endo_gs/contracts/stereo.py`
- `src/reliable_endo_gs/contracts/gaussian.py`
- `src/reliable_endo_gs/contracts/rendering.py`
- `src/reliable_endo_gs/contracts/reconstruction.py`
- `tests/contracts/test_camera.py`
- `tests/contracts/test_stereo.py`
- `tests/contracts/test_gaussian.py`
- `tests/contracts/test_rendering.py`
- `tests/contracts/test_reconstruction.py`
- `tests/runtime/test_scientific_seed.py`
- `environments/cuda.yml` only if an exact verified environment can be declared; otherwise retain a documented follow-up requirement rather than guessed pins.

## Files expected to be modified

- `pyproject.toml`
- `environments/cpu.yml`
- `src/reliable_endo_gs/runtime/manifest.py`
- `src/reliable_endo_gs/utils/seed.py`
- `docs/development.md`
- `docs/contracts.md` only for implementation-resolved, non-scientific details.

## Interfaces and contracts

Implement the fields and shapes from `docs/contracts.md`. Contract construction validates rank, batch agreement, spatial agreement, mask dtype, optional target shape, covariance tail shape `[3,3]`, and consistent device. `ReconstructionState` remains Phase I-only and contains no router/action label. Validation errors identify the field, expected invariant, and observed value without dumping sensitive tensors.

Public conversion methods return new instances. Schema version is separate from package version. Tests prove the contracts can be constructed and used without importing any third-party package.

## Scientific formulation

No research equation is implemented. Shape and validity checks are infrastructure transformations, not scientific contributions.

## Numerical and coordinate conventions

- Default scientific floating dtype must be explicit in config; do not silently cast float64 calibration to float32.
- Images use `[B,3,H,W]`; disparity/depth use `[B,1,H,W]`; Gaussians use `[B,N,*]`.
- Masks are boolean unless a named soft-weight contract explicitly says otherwise.
- Camera/world, pixel-center, depth, disparity, baseline, and Gaussian-frame conventions remain required metadata with unresolved values clearly marked until Plans 01-02 verify them.
- Non-finite validation may be strict or mask-aware, but its ownership and cost must be documented.

## Configuration changes

Plan a typed scientific runtime section for device, dtype, determinism, and seed. Do not create experiment YAMLs. CPU defaults must work without CUDA; CUDA selection must fail clearly when unsupported.

## Tests

### Unit

- Valid and invalid shape/rank cases for every field.
- Batch/spatial/device/dtype mismatch detection.
- Optional fields, empty valid masks, padded Gaussians, immutability, and conversion-copy behavior.
- Deterministic CPU random streams and seed provenance.

### Contract

- Round-trip representative contract metadata and schema identifiers.
- Prove contracts import without upstream, dataset, CUDA, or renderer access.

### Smoke

- Construct a tiny CPU `StereoBatch`, prediction, field, render output, and reconstruction state.

### Regression

- Stable schema names and error behavior for supported public invariants.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/contracts tests/runtime
pytest -q
python -m compileall src
git diff --check
```

## Experiments

No scientific experiment.

## Metrics

- Contract test pass rate.
- CPU smoke completion.
- Import-time dependency and CUDA-access checks.

## Acceptance criteria

- All six contracts exist independently of Endo-E2E-GS and enforce documented shapes/semantics.
- CPU installation and CI pass with PyTorch and without CUDA.
- No import initializes CUDA or accesses data.
- Deterministic seeding and manifest metadata are tested.
- Existing infrastructure/data tests remain green.

## Failure conditions

- Contract semantics require guessing upstream/dataset conventions.
- CPU dependency resolution or supported Python compatibility cannot be made reproducible.
- Constructors hide device moves, mutation, or lossy casts.

## Pivot / rollback path

Keep unresolved conventions as required explicit metadata and defer concrete values to Plans 01-02. If one PyTorch pin conflicts with supported Python, revise the supported matrix transparently rather than adding ad hoc environment branches. Roll back any premature CUDA pin while retaining CPU contracts.

## Artifact/provenance requirements

Record dependency versions, contract schema versions, config hash, Git revision/dirty state, seed policy, Python/PyTorch versions, and CPU platform in the smoke run. Do not call this a baseline or Phase I artifact.

## Completion checklist

- [ ] PyTorch dependency and CPU/CUDA separation are documented.
- [ ] Tensor/device/dtype policy is explicit.
- [ ] Six Phase I contracts and validations are implemented.
- [ ] Scientific seeding is deterministic and recorded.
- [ ] CPU unit, contract, smoke, and existing tests pass.
- [ ] No upstream or dataset integration was introduced.

## Handoff to next plan

Plan 01 may assume stable, upstream-independent tensor contracts and a CPU-testable PyTorch foundation; it may not assume any camera or baseline convention until it verifies the actual upstream source.
