# SCARED Multi-Sequence Data Contract & Expansion Specification

## Scope
This document specifies the pipeline boundaries, file layout, and tensor contracts for the ReliableEndo-GS multi-sequence SCARED expansion.

## Controlled Five-Keyframe Split (v3 Revision)
The current canonical split is the frozen five-keyframe v3 benchmark. It retains the v2 train groups and adds `dataset_4/keyframe_4` as a second held-out validation keyframe:

- **TRAIN**:
  - `dataset_1/keyframe_1`
  - `dataset_2/keyframe_1`
  - `dataset_3/keyframe_1` (already processed and validated)
- **VALIDATION**:
  - `dataset_7/keyframe_2`
  - `dataset_4/keyframe_4`
- **TEST**: `[]` (unallocated at this phase)

*(Historical note: the earlier v1 candidate additionally had `dataset_9/keyframe_3` in validation, which was removed in v2 because of zero-valid disparity frames.)*

### Grouping and Isolation Principles
1. **Complete Case-Level Isolation**: The dataset cases in the training split (`dataset_1`, `dataset_2`, `dataset_3`) and validation split (`dataset_7`, `dataset_4`) are strictly disjoint. No patient/case data crosses split boundaries.
2. **Complete Keyframe-Level Isolation**: All frames within each designated keyframe belong exclusively to that split. There is zero keyframe overlap between train and validation.
3. **No Machine-Specific Paths**: All manifest entries use portable relative paths `dataset_X/keyframe_Y` resolved against the external root (e.g. `RELIABLE_ENDO_DATA_ROOT/scared`).

## Expected Keyframe Directory Structure
Each processed keyframe directory (e.g. `<SCARED_ROOT>/dataset_X/keyframe_Y`) must contain the following subdirectories:

```text
dataset_X/keyframe_Y/
`-- data/
    |-- frame_data/          # Per-frame camera calibration & pose JSONs (frame_data000000.json, ...)
    |-- left_finalpass/      # Rectified 8-bit RGB left PNG images
    |-- right_finalpass/     # Rectified 8-bit RGB right PNG images
    |-- disparity/           # Single-channel float32 disparity TIFFs
    |-- newpram_data/        # Rectified intrinsics and extrinsics JSONs
    `-- reprojection_data/   # 4x4 reprojection Q-matrix JSONs
```

### File Formats & Keyframe Metadata
1. **`frame_data/{frame_id}.json`**:
   - Upstream frame log containing `camera-calibration` (`KL`, `KR`, `DL`, `DR`, `R`, `T`) and `camera-pose` (4x4 matrix).
2. **`left_finalpass/{frame_id}.png` & `right_finalpass/{frame_id}.png`**:
   - 8-bit RGB rectified stereo images.
   - Standard resolution: `(1280, 1024)` (width x height).
3. **`disparity/{frame_id}.tiff`**:
   - 32-bit floating point single-channel disparity map `(1024, 1280)`.
   - All values must be finite. Zero indicates unmeasured/invalid disparity.
4. **`newpram_data/{frame_id}.json`**:
   - `intr0`: 3x3 float matrix of rectified left camera intrinsics.
   - `intr1`: 3x3 float matrix of rectified right camera intrinsics.
   - `extr0`: 3x4 float matrix `[R | T]` of left camera extrinsics.
5. **`reprojection_data/{frame_id}.json`**:
   - `reprojection-matrix`: 4x4 float matrix `Q` satisfying standard OpenCV stereo reprojection.

## Stage-1 Tensor Contract
Each decoded item returned by the dataset or DataLoader conforms strictly to the Stage-1 tensor contract:

| Field | Type | Shape | Value Domain | Description |
|---|---|---|---|---|
| `dataset_id` | `str` | scalar | e.g. `"dataset_3"` | Case identifier |
| `keyframe_id` | `str` | scalar | e.g. `"keyframe_1"` | Keyframe identifier |
| `frame_id` | `str` | scalar | e.g. `"frame_data000000"` | Frame file stem |
| `sample_id` | `str` | scalar | `dataset_id/keyframe_id/frame_id` | Canonical provenance string |
| `left` | `torch.Tensor` (float32) | `[3, H, W]` | `[-1.0, 1.0]` | Normalized left RGB image, all finite |
| `right` | `torch.Tensor` (float32) | `[3, H, W]` | `[-1.0, 1.0]` | Normalized right RGB image, all finite |
| `disparity` | `torch.Tensor` (float32) | `[H, W]` | `>= 0.0` | Disparity map, all finite |
| `intr` | `torch.Tensor` (float32) | `[3, 3]` | finite | Left camera intrinsics |
| `right_intr` | `torch.Tensor` (float32) | `[3, 3]` | finite | Right camera intrinsics |
| `extr` | `torch.Tensor` (float32) | `[3, 4]` | finite | Left camera extrinsics |
| `Q` | `torch.Tensor` (float32) | `[4, 4]` | finite | OpenCV stereo reprojection matrix |

## DataLoader Semantics
- Standard training / evaluation configuration: `DataLoader(dataset, batch_size=1, num_workers=0)`.
- Collated output preserves all dictionary keys, batching tensors to `[1, 3, H, W]`, `[1, H, W]`, `[1, 3, 3]`, `[1, 3, 4]`, and `[1, 4, 4]`.
- Strings are collated into length-1 lists: `['dataset_3/keyframe_1/frame_data000000']`.

## Division of Responsibilities
- **ReliableEndo-GS Pipeline (This Work)**:
  - Manifest schema and strict cross-split leakage validation (`scared_manifest.py`).
  - Keyframe dataset and multi-sequence composing dataset (`scared_multi.py`).
  - Strict finite, shape, and value range checks on the Stage-1 tensor contract (`validate_stage1_sample`).
  - Keyframe structural contract validation helper (`scared_contract.py`).
  - Focused unit and regression test suite (`tests/data/test_scared_multi_sequence.py`).
- **Data Acquisition & Extraction**:
  - Raw archive extraction and stereo rectification preprocessing are decoupled and performed externally (by Luna).
