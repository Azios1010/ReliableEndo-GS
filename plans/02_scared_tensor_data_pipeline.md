# Plan 02 — SCARED Tensor Data Pipeline

## Status

PLANNED

## Research stage

Infrastructure / Data

## Recommended executor

Terra. Once the real layout and conventions are verified, this is bounded adapter, indexing, tensor-loading, validation, and manifest work.

## Estimated implementation risk

MEDIUM. The implementation is conventional, but silent calibration, rectification, unit, or sequence-split mistakes would create severe scientific leakage.

## Objective

Extend the external-data control plane into a deterministic, validated SCARED sample/tensor pipeline that yields contract-valid `StereoBatch` objects without moving private data into the repository.

## Scientific question

Not a hypothesis-testing milestone.

## Why this milestone exists

Baseline reproduction needs authoritative stereo pairs, calibration, depth/disparity supervision, masks, and sequence identity. The current adapter intentionally knows none of the real layout; implementation must begin from mounted-data and official-format inspection, not guessed paths.

## Prerequisites

- Plans 00 and 01 accepted.
- Read access to an authorized SCARED mount via `RELIABLE_ENDO_DATA_ROOT`.
- Official dataset documentation/license and baseline input requirements.

## Inputs

- Existing `configs/data/scared.yaml`, resolver, registry, adapter scaffold, and split schema.
- Mounted SCARED release metadata and authoritative documentation.
- Plan 00 tensor contracts and Plan 01 verified camera/input conventions.

## Outputs

- A documented release/layout and calibration audit.
- Deterministic sequence/keyframe enumeration and stable sample IDs.
- Calibration, stereo image, optional depth/disparity, and mask decoding into `StereoBatch`.
- Frozen sequence-level split manifest with schema/hash and no test-set tuning.
- Dataset validation/index CLI and small synthetic or redistributable fixtures.

## Scope

1. Identify the mounted release/version and verify real directory/file layout.
2. Verify left/right association, keyframe semantics, image encoding, calibration format, pose meaning, rectification status, depth/disparity representation, units, invalid sentinels, and masks.
3. Define stable sequence and sample IDs independent of absolute paths.
4. Enumerate samples deterministically and detect missing/duplicate/mismatched members.
5. Decode data with explicit resize/crop and intrinsic-update behavior compatible with the baseline.
6. Produce `CameraBatch` and `StereoBatch` with named masks and convention metadata.
7. Create a grouped train/validation/test protocol at sequence or dataset/keyframe level; hash it before model selection.
8. Extend CLI validation from “directory exists” to detailed layout/index/calibration checks.

## Non-goals

- No random frame split, download logic, augmentation policy, baseline training, uncertainty, SCARED-C mixing, EndoNeRF/C3VD tensors, or test-sequence tuning.
- Do not commit images, depth, calibration containing restricted information, copied subsets, patient identifiers, or absolute server paths.
- Do not invent sequence IDs or layout during planning or implementation.

## Files expected to be created

- `src/reliable_endo_gs/data/index.py`
- `src/reliable_endo_gs/data/tensors.py`
- `src/reliable_endo_gs/data/calibration.py`
- `src/reliable_endo_gs/data/loaders.py`
- `tests/data/test_scared_layout.py`
- `tests/data/test_scared_index.py`
- `tests/data/test_scared_tensors.py`
- `tests/data/test_calibration.py`
- `tests/data/fixtures/scared_synthetic/` containing generated non-medical fixtures.
- `configs/data/splits/scared_<verified_protocol>.json` only after authorized IDs and grouping rules are approved for commit.
- `docs/scared_data_contract.md`.

## Files expected to be modified

- `src/reliable_endo_gs/data/adapters/scared.py`
- `src/reliable_endo_gs/data/base.py`
- `src/reliable_endo_gs/data/schema.py`
- `src/reliable_endo_gs/data/validation.py`
- `src/reliable_endo_gs/cli/main.py`
- `configs/data/scared.yaml`
- `docs/data_architecture.md`
- `docs/server_data_setup.md`
- `.gitignore` if new external cache/index payload patterns arise.

## Interfaces and contracts

The adapter exposes deterministic `enumerate_sequences()` and `inspect_metadata()` using stable logical records. A separate sample index maps IDs to external paths without serializing machine paths as scientific identity. A loader converts selected records to `StereoBatch`; it never chooses a split implicitly.

Calibration parsing produces `CameraBatch` only after convention checks. Resize/crop updates intrinsics through one tested transform. Invalid supervision is represented by named masks, not zero-value inference.

## Scientific formulation

No new research formula. Intrinsic resize/crop adjustment and any depth/disparity decoding are standard implementation transformations owned by `data/calibration.py` or the canonical geometry owner when metric conversion is required. They require analytic round-trip tests and are not novelty claims.

## Numerical and coordinate conventions

Resolve and record pixel center, original/resized coordinates, reference view, rectification, disparity sign/unit, depth type/unit, baseline vector/unit, transform direction, camera axes/handedness, invalid sentinels, color space/range, and interpolation modes. Never derive a convention solely from a plausible image.

## Configuration changes

Extend `configs/data/scared.yaml` only with verified scalar options such as release label or keyframe protocol. Plan `configs/experiment/p0_baseline_reproduction.yaml` later; do not create it here. Split membership lives in an immutable manifest, not inline YAML lists.

## Tests

### Unit

- Filename/metadata parsing using synthetic fixtures.
- Calibration decoding and intrinsic adjustment with analytic cameras.
- Stable ID formation, sorting, duplicate detection, and missing-member failures.

### Contract

- Loaded tensors satisfy `StereoBatch`/`CameraBatch` shapes, masks, devices, dtype, and metadata.
- Dataset imports and adapter construction perform no filesystem access.

### Integration

- Root resolver -> SCARED adapter -> verified index -> split manifest -> tensor loader on a mounted small subset.

### Smoke

- CLI validates and loads a requested sample by ID deterministically without exposing sensitive paths in ordinary logs.

### Regression

- Freeze index counts and metadata hashes for an explicitly named authorized development subset; do not commit its contents.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/data tests/contracts
pytest -q
python -m compileall src
reg data validate --config configs/data/scared.yaml
git diff --check
```

## Experiments

No scientific experiment. Run a data protocol audit on the mounted release and generate descriptive counts only.

## Metrics

- Sequence/sample counts and missing/invalid counts.
- Stereo/calibration/depth availability rates.
- Contract-valid sample rate.
- Deterministic index and split hashes across repeated runs.

## Acceptance criteria

- A small authorized mounted subset deterministically produces valid `StereoBatch` objects.
- Layout, calibration, rectification, units, and masks are documented from authoritative evidence.
- Sequence-level split isolation passes and split hash is stable.
- Detailed validation distinguishes missing, incomplete, invalid, and usable data.
- No private data or machine-specific paths are committed.

## Failure conditions

- Calibration meaning, rectification, units, or stereo association cannot be established.
- Required files are missing or license terms prevent the planned use.
- Split grouping cannot prevent adjacent-frame/sequence leakage.

## Pivot / rollback path

Stop baseline reproduction and retain the layout audit. Narrow to a verified keyframe subset or an approved SCARED/SCARED-C protocol reported separately; never merge corrected and original pose regimes silently. Keep synthetic tests while access issues are resolved.

## Artifact/provenance requirements

Record dataset name/release, portable config hash, split schema/hash, index schema/hash, selection policy, adapter version, contract schema, code revision, and validation summary. Do not claim a whole-dataset content hash unless one is actually computed and defined.

## Completion checklist

- [ ] Real layout and release inspected authoritatively.
- [ ] Calibration, rectification, units, and masks verified.
- [ ] Stable sequence/sample IDs and deterministic index implemented.
- [ ] Frozen grouped split produced without test tuning.
- [ ] Mounted-subset `StereoBatch` integration passes.
- [ ] Privacy and external-storage audit passes.

## Handoff to next plan

Plan 03 may assume a frozen SCARED protocol and deterministic baseline-compatible tensors. It may not change preprocessing or split membership while establishing baseline metrics.
