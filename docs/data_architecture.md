# External Dataset Architecture

## 1. Why datasets remain external

SCARED, SCARED-C, EndoNeRF, and C3VD may be large, licensed, server-mounted,
or medically sensitive. ReliableEndo-GS therefore versions only portable
identity configuration, adapter code, validation rules, and approved split
manifests. Raw data, copied subsets, credentials, and developer-specific mount
paths remain outside Git.

Importing `reliable_endo_gs.data` performs no root resolution or dataset
filesystem inspection. Filesystem access occurs only after a caller explicitly
asks for validation, indexing, or tensor loading.

## 2. Dataset root resolution

Each dataset YAML contains a portable relative root such as `SCARED` or
`scared_c`. Relative roots are resolved against an explicit `global_root`
first and otherwise against `RELIABLE_ENDO_DATA_ROOT`. Both global-root forms
must be absolute.

Resolution never falls back to the current directory, repository root, home
directory, or `data/`. It does not require the resulting path to exist.
Existence and layout checks belong to explicit validation.

## 3. Dataset config versus training config

`DatasetConfig` contains dataset name, configured root, optional release/version
label, optional split-manifest path, and a restricted scalar options mapping.
It does not contain batch size, learning rate, augmentation, model resolution,
or scientific preprocessing. Dataset-specific options record verified protocol
choices such as calibration source and keyframe enumeration policy.

## 4. Adapter responsibilities

Adapters own dataset-specific identity, layout recognition, file association,
logical enumeration, and non-tensor metadata extraction. Generic indexing,
calibration math, tensor conversion, and split validation are separate modules.

`ScaredAdapter` remains the original SCARED scaffold and does not mean
SCARED-C. `ScaredCAdapter` is registered separately under `scared_c` and
recognizes the inspected static-keyframe layout. Its selected protocol is
`scared_c_endoscope_stereo_calibration_v1`; it keeps the single COLMAP intrinsic
file separate from the endoscope stereo `M1`/`M2` regime.

Adapters do not access data during import or construction. They decode data
only from explicitly requested validation/loading methods.

## 5. SCARED-C data flow

```text
portable YAML
-> root resolver
-> ScaredCAdapter
-> deterministic SampleIndex
-> centralized calibration parser
-> lazy PNG/TIFF loader
-> CameraBatch + StereoBatch
```

The SCARED-C adapter enumerates one logical `reference` sample per complete
top-level `dataset_*/keyframe_*` directory. The temporal archive and frame log
are recorded as metadata, but the current development protocol does not
pretend that the top-level pair maps to a particular temporal frame.

The TIFF assets are per-pixel XYZ coordinate maps, not scalar depth. They are
loaded as `gt_depth_xyz` and `gt_right_depth_xyz`; scalar `gt_depth` is absent.
Validity requires finite values in all three channels and is represented by
the named `left_depth_valid` and `right_depth_valid` masks.

## 6. Split manifests and leakage control

Committed split manifests use JSON and an explicit schema version. They identify
the dataset and optional dataset version, name the protocol, and assign string
sequence IDs to train, validation, and test. IDs must be unique within a split
and isolated across splits. The generic grouped validation helpers reject a
sample mapping that places one sequence group in multiple splits.

The one-sequence local SCARED-C development sample has no final train,
validation, or test manifest. A random-frame split is not fabricated.

## 7. Split hashing

The split hash is SHA-256 over schema version, dataset, dataset version, and
sorted train/validation/test IDs. ID ordering is semantically irrelevant. File
location, split name, notes, timestamp, and developer paths are excluded.

Terminology is strict:

- `config_hash` identifies portable dataset configuration.
- `split_hash` identifies scientific membership in train/validation/test.
- `index_hash` identifies a deterministic logical index and its relative member
  metadata; it is not a whole-dataset content hash.
- `dataset_content_hash` is not claimed by this repository.

## 8. Validation levels

Detailed validation reports one of `MISSING`, `INCOMPLETE`, `INVALID`, or
`USABLE`, with counts for sequences and samples, stereo availability,
calibration/depth status, contract validation, and an index hash where one
exists. The CLI redacts resolved external roots in ordinary JSON output.

`USABLE` for SCARED-C means complete indexed static keyframe records decode to
contract-valid tensors under the named development protocol. It does not mean
that pose direction, metric units, camera axes, rectification, or temporal
frame association have been scientifically resolved.

## 9. Dataset identity and provenance

Dataset name, optional version, portable root declaration, optional split
manifest reference, restricted options, protocol, adapter version, contract
schema, index hash, and validation summary form the portable control-plane
identity. Mounted file paths are operational pointers only and are never part
of logical IDs or committed manifests.

## 10. Future transition to sample and tensor loading

The current Plan 02 flow is:

```text
dataset YAML
-> root resolver
-> dataset adapter
-> deterministic sample index
-> optional grouped split manifest
-> selected tensor loader
```

No baseline training, geometry conversion, uncertainty, rendering, repair, or
routing behavior is part of this data layer.

## 11. Server workflow

An operator mounts data, sets a portable external root, and explicitly
validates the selected dataset. The same committed config can move between
servers by changing only `RELIABLE_ENDO_DATA_ROOT` or passing an absolute
`--data-root` override. Ordinary validation output redacts the resolved path;
debugging can inspect the explicit command environment locally.
