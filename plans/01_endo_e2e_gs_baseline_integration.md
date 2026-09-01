# Plan 01 — Endo-E2E-GS Baseline Integration

## Status

IN PROGRESS

## Research stage

Baseline

## Recommended executor

Sol. This is a high-risk source, license, checkpoint, CUDA, renderer, and numerical-parity integration whose real API must be discovered rather than inferred.

## Estimated implementation risk

HIGH. Upstream version drift, undocumented preprocessing, custom rasterization, checkpoint compatibility, and coordinate conventions can invalidate all downstream work.

## Objective

Pin and isolate an authorized Endo-E2E-GS revision, expose unmodified inference through ReliableEndo-GS contracts, and prove adapter conversion parity without introducing research behavior.

## Scientific question

Not a hypothesis-testing milestone; it establishes an honest reference implementation.

## Why this milestone exists

Every scientific claim is relative to Endo-E2E-GS. The adapter must reveal verified disparity, Gaussian, camera, and renderer behavior while preventing upstream object names and tensors from leaking into core packages.

## Prerequisites

- Plan 00 accepted.
- Contract schema and CPU tensor policy available.
- `docs/third_party_integration.md` approved.

## Inputs

- Official repository, paper, license, release/checkpoint instructions, and available tags/commits.
- Plan 00 contracts.
- Current environment/dependency policy.

## Outputs

- Documented official origin, license/attribution, exact commit SHA, dirty-state policy, and optional fork/patch identity.
- Pinned submodule or equivalently immutable approved dependency under `third_party/endo_e2e_gs`.
- Baseline adapter that maps upstream inference to ReliableEndo-GS contracts.
- API/convention discovery record and numerical parity regression fixtures.
- Integration provenance manifest; not yet the dataset baseline artifact.

## Scope

1. Inspect official source and checkpoint terms before choosing a revision.
2. Reproduce the upstream-provided inference path unchanged on its supported fixture/example.
3. Choose direct submodule, research-fork submodule, or equivalent pin based on patch need; prefer wrapper-only integration.
4. Map actual access points for stereo disparity, optional disparity iterations, depth/backprojection, GSRegresser attributes, Gaussian parameters, renderer inputs/outputs, and left/right cameras.
5. Normalize only representation details such as shape/order/device at the adapter boundary and measure conversion tolerance.
6. Record unsupported optional evidence as absent; never fabricate RAFT features.
7. Audit repository imports so only `baseline` directly understands upstream internals.

## Non-goals

- No SCARED loader, training modification, uncertainty head, covariance, cross-view loss, repair, oracle, or router.
- Do not patch scientific behavior to make integration easy.
- Do not design a generic renderer abstraction until actual upstream renderer semantics are known.

## Files expected to be created

- `.gitmodules` when submodule strategy is selected.
- `third_party/endo_e2e_gs/` as a pinned external source boundary.
- `src/reliable_endo_gs/baseline/__init__.py`
- `src/reliable_endo_gs/baseline/adapter.py`
- `src/reliable_endo_gs/baseline/config.py`
- `src/reliable_endo_gs/baseline/provenance.py`
- `docs/endo_e2e_gs_api_audit.md`
- `tests/baseline/test_adapter_contract.py`
- `tests/baseline/test_upstream_parity.py`
- `tests/baseline/fixtures/` for small legally redistributable metadata/numerical fixtures only.
- `patches/endo_e2e_gs/` only if a documented minimal patch is unavoidable.

## Files expected to be modified

- `pyproject.toml`
- One verified CUDA environment file, if upstream requires it.
- `docs/development.md`
- `docs/third_party_integration.md` with resolved strategy and provenance fields.
- `.gitignore` for upstream-generated caches/checkpoints.

## Interfaces and contracts

The primary adapter accepts `StereoBatch` plus a validated baseline config and returns normalized `StereoPrediction`, `GaussianField`, and renderer outputs that the verified upstream path actually produces. If upstream stages cannot be separated without behavior change, expose a documented composite result rather than inventing hooks.

The adapter owns normalization, resize reversal, camera conversion, optional evidence availability, checkpoint loading, and upstream provenance. All downstream imports stop at `reliable_endo_gs.baseline`; direct imports from `third_party` elsewhere are contract failures.

## Scientific formulation

No new formula. Any upstream disparity-depth, backprojection, Gaussian, or rendering formula is documented as baseline behavior and is not reimplemented here unless conversion is required. Conversion equations are classified as implementation transformations and tested for invertibility/parity.

## Numerical and coordinate conventions

Verify from source and controlled inputs: reference view, disparity sign/scale, resize behavior, focal/baseline units, pixel center, transform direction, camera axes, quaternion ordering, scale/opacity parameterization, Gaussian frame, renderer covariance path, color range, and validity masks. Record unresolved ambiguities as blockers.

## Configuration changes

Plan `configs/baseline/endo_e2e_gs.yaml` for checkpoint logical identity, supported entry point, resolution, normalization, and adapter behavior. Machine paths stay outside committed config or use explicit external resolution. Do not create experiment configs yet.

## Tests

### Unit

- Shape/order and camera conversion helpers.
- Upstream commit/dirty-state and checkpoint-hash validation.

### Contract

- Actual upstream outputs satisfy Plan 00 contracts.
- Missing optional iterations/features remain absent.
- Direct-import audit enforces the boundary.

### Integration

- Unmodified upstream example inference and adapter inference use identical inputs and compare all mapped outputs within documented conversion tolerance.

### Smoke

- Minimal supported inference completes on required hardware with provenance.

### Regression

- Pin small output summaries/hashes/tolerances for the chosen commit and checkpoint; retain an unmodified upstream path even if a later patch is needed.

## Validation commands

```text
git submodule status
ruff check .
ruff format --check .
mypy src
pytest -q tests/baseline
pytest -q
python -m compileall src
git diff --check
```

## Experiments

Run an integration parity study, not a research comparison: upstream-native versus adapter-normalized outputs on identical legal fixtures and hardware.

## Metrics

- Maximum/mean absolute conversion difference per output.
- Valid-pixel agreement and shape agreement.
- Inference completion, checkpoint hash, upstream revision, and dirty state.

## Acceptance criteria

- License/attribution and checkpoint-use terms are documented and compatible.
- Exact upstream commit is pinned and recorded.
- Unmodified upstream inference passes through ReliableEndo-GS contracts without numerical change beyond a conversion tolerance justified and frozen from observed dtype/layout effects.
- No core package imports upstream internals.
- Baseline regression and existing CPU tests pass in their declared environments.

## Failure conditions

- License or checkpoint terms are incompatible or unclear.
- Official inference cannot be reproduced on its own supported example.
- Adapter conversion changes scientific behavior or conventions remain unresolved.
- Required patch cannot be isolated and tested against an unmodified path.

## Pivot / rollback path

Stop downstream work and retain a documented integration audit. Try a different official pinned revision or a minimal research fork only after explaining the incompatibility. Do not replace Endo-E2E-GS with an unapproved baseline or modify its science silently.

## Artifact/provenance requirements

Record origin/fork URLs, commit SHA, submodule dirty state, patch IDs, license files, checkpoint identity/hash, adapter schema, config hash, environment, hardware, and parity outputs. Immutable identities replace local checkpoint paths.

## Completion checklist

- [ ] Official source, license, and checkpoint terms verified.
- [x] Exact revision and optional patches pinned.
- [x] API and convention audit completed.
- [x] Adapter produces only ReliableEndo-GS contracts.
- [ ] Upstream-versus-adapter parity passes.
- [x] Direct-import and regression tests pass.

## Handoff to next plan

Plan 02 may assume a pinned adapter and verified input/camera requirements. It may not assume any SCARED filesystem layout, calibration interpretation, or rectification status until authoritative data inspection is complete.

## Implementation record (2026-08-19)

- Direct official Git submodule pinned at
  `186fa2b4a2159b28393492f6df1aa444b54391a8`; no patch or fork.
- CPU-safe contract translations, strict baseline config, checkpoint hashing,
  provenance, capability detection, and isolated optional imports implemented.
- Actual API, formulas, entry points, dependencies, license notices, accessible
  evidence, and unresolved conventions recorded in
  `docs/endo_e2e_gs_api_audit.md`.
- Official checkpoint terms/download/hash, legal fixture, CUDA 11.8 environment,
  external rasterizer revision, and end-to-end native parity remain blocked.
- CPU validation: 92 passed and one explicitly skipped optional renderer import;
  Ruff, Ruff format, mypy, compileall, CLI smoke/data help, and diff checks pass.
- Current capability probe: source and Python dependencies available, CUDA
  available, `diff_gaussian_rasterization` unavailable; full native inference
  and checkpoint loading were not completed.
- Server runtime follow-up: RTX 2050 and PyTorch CUDA are present, but no CUDA
  toolkit/MSVC toolchain, `corr_sampler`, authorized checkpoint, or legal
  fixture is available. Native inference therefore stops before disparity and
  rasterization. Exact runtime/provenance evidence is in
  `docs/endo_e2e_gs_runtime_parity_record.md`.
- Status remains **IN PROGRESS**. CPU conversion parity cannot satisfy the
  acceptance criterion requiring unmodified upstream inference.

## Implementation record update (2026-08-27)

- Current hardware/environment state:
  - Hardware: NVIDIA GeForce RTX 5070 Ti x2.
  - Python: 3.10.11.
  - PyTorch: `2.7.1+cu128`.
  - CUDA toolkit: private CUDA 12.8.1 toolkit prepared.
- Rasterizer state:
  - `diff_gaussian_rasterization` prior build and smoke check passed in the prepared environment, but was not rerun in this execution scope under the hard no-GPU rule.
- Correlation sampler compatibility:
  - Provenance resolved: `https://github.com/princeton-vl/RAFT-Stereo` at commit `6068c1a26f84f8132de10f60b2bc0ce61568e085`.
  - Minimal compatibility patch prepared under `patches/raft_stereo/0001-corr-sampler-scalar-type-compatibility.patch` and documented under `patches/raft_stereo/README.md` (`BUILD_API_COMPATIBILITY_ONLY`, updating `volume.type()` to `volume.scalar_type()` in `sampler/sampler_kernel.cu` for PyTorch 2.x ATen dispatch compliance while preserving `forward`/`backward` bindings and underlying kernel math).
  - Patch is statically validated; it has not been compiled or GPU-tested in this scope.
- Capability detection:
  - Refactored `BaselineCapabilities` and `inspect_capabilities` in `src/reliable_endo_gs/baseline/upstream.py` to separately report `upstream_source`, `python_dependencies`, `rasterizer`, `corr_sampler`, `cuda`, `renderer_ready` (`renderer_import_ready`), and `native_inference_ready` (`native_inference_dependency_ready`). Native inference readiness requires `corr_sampler` and is false when absent, while renderer readiness remains independently true when the rasterizer is present.
- Remaining blockers:
  - Official/author-provided checkpoint authorization and download/hash remain blocked.
  - Legal SCARED dataset fixture remains blocked on Plan 02 data inspection.
  - End-to-end native baseline inference and native-versus-adapter parity have not been run.
- Status: **Plan 01 remains IN PROGRESS**; its final checkpoint/parity validation is deferred. **Plan 02 is IN PROGRESS under the explicit early SCARED-C data-pipeline exception** and does not imply Plan 01 acceptance.
