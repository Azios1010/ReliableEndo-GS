# A00 processed SCARED protocol

The active A00 adapter reads `dataset_N/keyframe_M/data` (or the equivalent
keyframe root) without importing SCARED-C. Each keyframe must carry an
`a00_protocol.json` sidecar. The sidecar is part of the calibration identity;
the adapter will not infer units or transform direction from legacy
`newpram_data` or `frame_data` values.

The required schema is:

```json
{
  "schema_version": 1,
  "image_range": "uint8_0_255",
  "rectified": true,
  "distortion_removed": true,
  "pixel_center": "integer",
  "disparity": {
    "units": "pixels",
    "sign": "u_left_minus_right",
    "reference": "left"
  },
  "extrinsics": {
    "source": "newpram.T_R_from_L",
    "direction": "right_from_left",
    "units": "m"
  },
  "preprocessing": {"images_grid": "processed"}
}
```

`extrinsics.source` names a per-frame JSON field. It may be a homogeneous
`[4,4]`/`[3,4]` matrix, or a translation vector only when
`extrinsics.rotation_source` names the companion `[3,3]` rotation. Supported
translation units are `m`, `cm`, and `mm`; supported directions are
`right_from_left` and `left_from_right`. The resulting A00 camera must be
rectified, equal-focal, scanline-aligned, and have a positive metric baseline.

The adapter decodes PNG RGB to `[0,1]`, applies only an explicitly requested
paired crop/resize, scales both intrinsics and horizontal disparity exactly
once, and keeps disparity in `ScaredSupervision`; `DeploymentStereoBatch`
contains only RGB, camera, and preprocessing/sensor support. A missing sidecar,
missing sample file, unsupported field, or ambiguous calibration raises a
protocol/sample error with the required metadata rather than guessing.
