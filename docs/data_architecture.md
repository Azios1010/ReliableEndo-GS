# External Dataset Architecture

## 1. Why datasets remain external

SCARED, EndoNeRF, and C3VD may be large, licensed, server-mounted, or medically sensitive. ReliableEndo-GS therefore versions only portable identity configuration, adapter code, validation rules, and approved split manifests. Raw data, copied subsets, credentials, and developer-specific mount paths remain outside Git.

Importing `reliable_endo_gs.data` performs no root resolution or filesystem inspection. Filesystem access occurs only after a caller explicitly asks for validation or, in a future task, sample indexing.

## 2. Dataset root resolution

Each dataset YAML contains either a portable relative root such as `SCARED` or an intentional absolute override. Relative roots are resolved against an explicit `global_root` first and otherwise against `RELIABLE_ENDO_DATA_ROOT`. Both global-root forms must be absolute.

Resolution never falls back to the current directory, repository root, home directory, or `data/`. It does not require the resulting path to exist. Existence and directory checks belong to explicit validation.

## 3. Dataset config versus training config

`DatasetConfig` contains dataset name, configured root, optional release/version label, optional split-manifest path, and a restricted scalar options mapping. It does not contain batch size, learning rate, augmentation, model resolution, or scientific preprocessing. Prompt 03's infrastructure smoke configuration remains independent from all dataset configuration.

Dataset-specific options are allowed only when a verified adapter needs a small declarative scalar. Nested arbitrary structures are rejected so `options` cannot become an uncontrolled second configuration system.

## 4. Adapter responsibilities

The non-PyTorch `DatasetAdapter` contract owns dataset identity, explicit root validation, future sequence enumeration, and future non-tensor metadata inspection. Registration is deterministic and explicit for `scared`, `endonerf`, and `c3vd`.

Current adapters distinguish a missing path, a non-directory path, and an existing directory. An existing directory passes basic validation while also reporting that detailed layout validation is not implemented.

## 5. What adapters must not do

Adapters do not access data during import or construction. At this stage they do not decode images or depth, parse calibration, infer camera conventions, create tensors, perform augmentation, rectify stereo pairs, or assume unverified directory layouts. Sequence enumeration and metadata inspection deliberately raise `NotImplementedError` until authoritative layouts are approved.

## 6. Split manifests

Committed split manifests use JSON and an explicit schema version. They identify the dataset and optional dataset version, name the protocol, and assign string sequence IDs to train, validation, and test. IDs must be unique within a split and isolated across splits. Empty manifests are valid for schema development but are not evidence of an approved scientific protocol.

## 7. Split hashing

The split hash is SHA-256 over schema version, dataset, dataset version, and sorted train/validation/test IDs. ID ordering is treated as semantically irrelevant. File location, split name, notes, timestamp, and developer paths are excluded.

Terminology is strict:

- `config_hash` identifies portable dataset configuration.
- `split_hash` identifies scientific membership in train/validation/test.
- `dataset_content_hash` does not exist yet; the repository does not claim to hash mounted dataset contents.

## 8. Validation levels

Structured reports distinguish:

1. `DATASET NOT PRESENT` — the resolved root does not exist.
2. `DATASET PRESENT BUT INVALID` — a path exists but is not a directory.
3. `DATASET PRESENT, BASIC VALIDATION PASSED` — a directory exists.
4. `DETAILED LAYOUT VALIDATION NOT IMPLEMENTED YET` — no verified internal-layout check was performed.

Basic validity must not be presented as proof that images, depth, poses, or calibration are complete.

## 9. Dataset identity and provenance

Dataset name, optional version, portable root declaration, optional split-manifest reference, and restricted options form the dataset configuration identity. A future run manifest may record dataset config and split hashes only after the associated execution path is introduced. It must not fabricate a dataset-content hash.

## 10. Future transition to sample and tensor loading

The intended flow is:

```text
dataset YAML
-> root resolver
-> adapter registry
-> dataset adapter
-> split manifest
-> future sample index
-> future tensor loader
```

This task stops at the split manifest. Sample indexing, decoding, tensor contracts, and scientific preprocessing require later architecture and implementation approval.

## 11. Server workflow

An operator mounts data, sets `RELIABLE_ENDO_DATA_ROOT`, resolves a portable YAML, and explicitly validates the selected dataset. The same committed config can move between servers by changing only the environment variable. An absolute `--data-root` override supports scheduled jobs or alternate scratch mounts without editing committed YAML.

