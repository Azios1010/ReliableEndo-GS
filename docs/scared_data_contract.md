# SCARED-C Data Contract

## Scope
This document specifies the pipeline boundaries for the `scared_c` dataset adapter as implemented in Plan 02.

## Protocol and Identity
- **Dataset Name**: `scared_c`
- **Dataset distinction**: SCARED-C is a separately identified corrected/re-estimated development dataset. It is not original SCARED, and this adapter does not reinterpret original SCARED records.
- **Adapter**: `ScaredCAdapter`
- **Configuration**: Uses `configs/data/scared_c.yaml` and enforces the protocol `scared_c_endoscope_stereo_calibration_v1`.
- **Validation**: Enforces checks for layout, calibration files, and depth maps, yielding `MISSING`, `INCOMPLETE`, `INVALID`, or `USABLE` statuses.

The selected protocol uses `endoscope_calibration.yaml` (`M1`, `M2`, and available stereo fields) as its camera source. The separate `intrinsics_colmap.yaml` file is recorded as a distinct available source but is not substituted into this protocol. Per-frame `camera-pose` records are available in the frame archive, but pose direction and coordinate-frame semantics are unresolved, so they are not applied by the loader.

## Geometry and Conventions
- **Axes**: Handedness, axis order, and coordinate-frame semantics are `UNRESOLVED` at the data interface. Numeric signs alone are not used to select a convention.
- **Units**: Treated as `UNRESOLVED`. No automatic millimeter conversion is assumed.
- **Rectification**: The rectification state and any warp relationship between the left/right maps are `UNRESOLVED`; no rectification is performed.
- **Image-to-Frame Mapping**: The top-level images are tracked as reference frames but the exact temporal frame index mapping is `UNRESOLVED`.

## Depth Format
- **3D Maps**: The selected sample provides separate left and right 3-channel (X, Y, Z) point maps stored as 32-bit floats. This local sample evidence is not generalized into an unverified claim about every SCARED-C release, and it is not silently converted to scalar depth.
- **Invalid Masking**: Handled explicitly using finite-value checks against IEEE `NaN`. The invalid pixels are strictly masked.
- **StereoBatch Contract**: Additive left XYZ depth (`gt_depth_xyz`) and right XYZ depth (`gt_right_depth_xyz`) are provided to `StereoBatch`, with corresponding validity masks. The loader uses a camera-local identity transform because the external pose transform is unresolved; metadata records that neither external pose nor stereo extrinsics were applied. This is not a claim about the dataset world frame.

## Indexing
- Sequences are assigned deterministic, sorted logical IDs independent of local server absolute paths.
- Sample logic enforces duplicate, missing, and mismatch detection.
- A split manifest configures sequence partitioning (train/val/test) without test set tuning.
