# A00 processed SCARED protocol

The active A00 adapter reads original SCARED `dataset_1`–`dataset_9` at
`dataset_N/keyframe_M/data` (or the equivalent keyframe root), and rejects
SCARED-C and `dataset_10+`. Dataset_1–7 are final training data; dataset_8–9
are final-test data. Each keyframe must carry an
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

## Original SCARED one-sample preprocessing

`configs/data/a00_scared_sample_v1.yaml` pins the first-party preprocessing
settings. The command
`env PYTHONPATH=<checkout>/src <target-python> -m reliable_endo_gs.cli.main a00 prepare`
reads one video frame and the exact same-ID `frame_data`/`scene_points` members
without extracting them into the source dataset. It writes processed files
only under an external output root.
The v1 transform is `R_rect = R2 @ R @ R1.T` and
`T_rect = R2 @ T`; it checks rectified rotation and y/z translation before
writing `newpram_data.T_R_from_L`. Historical `extr0` is kept separate and is
never used for stereo geometry.

The external calibration evidence manifest consumed by `a00 prepare` must
bind evidence to the exact selected input members and use this shape:

```json
{
  "schema_version": 1,
  "sample_id": "dataset_1/keyframe_1/frame_data000000",
  "calibration": {
    "source_field": "frame_data.camera-calibration.R/T",
    "translation_unit": "mm",
    "transform_direction": "right_from_left",
    "calibration_value_source": {
      "field_path": "R/T",
      "source_title": "Original SCARED calibration value YAML",
      "source_uri": "SCARED:dataset_1/keyframe_1/endoscope_calibration.yaml",
      "source_path": "dataset_1/keyframe_1/endoscope_calibration.yaml",
      "source_sha256": "597878f105e38ff2c2b7973d2c3a6af31293b904ca3b73bbe0a5f239466e6e9e",
      "frame_data_match": {
        "source_field": "frame_data.camera-calibration.R/T",
        "frame_data_member_sha256": "<SHA-256 of selected JSON member bytes>",
        "R_matches": true,
        "T_matches": true
      }
    }
  },
  "direction_resolution": {
    "method": "cross_view_nearest_neighbor",
    "resolution_basis": "empirical_paired_cloud_residuals",
    "selected_direction": "right_from_left",
    "candidate_transform": "R @ left_xyz + T -> right_xyz",
    "direction_control": "R @ right_xyz + T -> left_xyz",
    "scene_points_layout": "top_left_bottom_right_xyz",
    "metric_ground_truth_source_uri": "https://endovissub2019-scared.grand-challenge.org/About/"
  },
  "unit_resolution": {
    "method": "cross_view_nearest_neighbor",
    "resolution_basis": "empirical_cross_view_against_metric_ground_truth",
    "selected_translation_unit": "mm",
    "metric_ground_truth_source_uri": "https://endovissub2019-scared.grand-challenge.org/About/"
  },
  "source_member_sha256": {
    "frame_data": "<SHA-256 of selected JSON member bytes>",
    "scene_points": "<SHA-256 of selected TIFF member bytes>"
  },
  "frame_correspondence": {
    "frame_data_id": "frame_data000000",
    "scene_points_id": "scene_points000000",
    "matched": true,
    "ambiguous": false
  },
  "independent_validation": {
    "method": "cross_view_nearest_neighbor",
    "valid_point_policy": "finite_nonzero_xyz_positive_z",
    "point_counts": {"left": 0, "right": 0},
    "metrics_mm": {
      "translation_as_mm": {"median": 0.0, "p90": 0.0, "p99": 0.0},
      "translation_as_cm": {"median": 0.0, "p90": 0.0, "p99": 0.0},
      "translation_as_m": {"median": 0.0, "p90": 0.0, "p99": 0.0},
      "wrong_direction_reuse_declared_transform": {"median": 0.0, "p90": 0.0, "p99": 0.0}
    }
  },
  "metric_ground_truth": {
    "xyz_units": "mm",
    "depth_units": "mm",
    "source_uri": "https://endovissub2019-scared.grand-challenge.org/About/"
  }
}
```

The example's metric values and frame-member hash are placeholders. The pinned
YAML SHA-256 is only the numeric R/T source; the processor checks those matrices
exactly against the selected `frame_data` JSON. It does not use the YAML's
serialization format to infer field semantics or camera order. It evaluates
`R @ left_xyz + T` against the right cloud and the opposite-use control
`R @ right_xyz + T` against the left cloud, using the declared stacked
scene-points layout and exact member hashes. It accepts `right_from_left` only
if the recomputed candidate residual beats the opposite-use control by the
configured margins. The `mm` conclusion is separately empirical: official
SCARED metric GT is the scale reference, and the measured nearest-neighbor
residuals must pass the configured mm limits and beat absolute cm and m
hypotheses. No unit or direction is attributed to the YAML.

The three translation hypotheses interpret raw numeric T absolutely as mm, cm,
and m against the exact two-view XYZ points. The config pins the calibration
YAML hash as the numeric-value source only. Neither producer semantics nor
camera order is claimed from its OpenCV serialization.

The producer records full code/config/evidence hashes, input member hashes,
output file hashes, OpenCV version and rectification settings in
`a00_protocol.json` and `a00_provenance.json`. It compares Q-derived depth with
A00's `fx * B / (d - (cx_L - cx_R))` and with the same-pixel metric scene-point
depth; configured maximum absolute errors are 0.01 mm and 0.05 mm respectively.
The selected v1 grid uses no crop or resize and requires a full source-support
mask after rectification, matching the A00 loader's all-supported input.

`a00 preflight` checks paths and output hashes, rehashes the recorded external
calibration-evidence JSON, validates its trusted source and member identities,
and parses the complete `A00Protocol` schema. It does this without decoding
images, loading GT, or calling the A00 adapter. It prints a runnable command
using the active Python executable and this checkout's `src` path. `a00 smoke`
is the separate deployment-only step:
it loads one `DeploymentStereoBatch` with supervision disabled and does not run
teacher inference, A01, A04, rendering, or training.

Future external provenance records the actual Python executable, module, argv,
working directory, `PYTHONPATH`, and resolved raw-data, output, config, and
evidence paths. The config pins the selected calibration YAML as a numeric
value source; direction and unit conclusions remain dependent on measured
cross-view and metric-GT acceptance. This remediation does not run real
preparation, preflight, or smoke.

## Batch frames without rectified GT support

`a00 prepare-dataset` retains synchronized RGB/calibration samples even when
valid raw XYZ projects outside the rectified stereo overlap, when no valid
left-camera XYZ exists, or when every projection rounds outside the image.
The disparity file contains only the declared invalid value zero. The adapter
returns an all-false GT validity mask while deployment sensor support remains
unchanged. GT never enters `DeploymentStereoBatch`.

Per-frame provenance reports `metric_validation.status: unavailable`,
`valid_pixel_count: 0`, a reason and projection counts. Error statistics are
`null`; neither Q/metric agreement nor a zero error is claimed. A
`sample_without_gt` progress event identifies every such sample, and batch and
keyframe summaries count `samples_without_gt`. Metric evaluation must exclude
these targets from its valid denominator and report their count. Samples stay
in their existing split. Geometry errors, calibration drift, ambiguous IDs and
failed tolerances remain fatal. Single-sample `a00 prepare` remains strict and
requires nonempty GT for readiness evidence.

Completed positive-GT outputs from the exact processor pair preceding this
remediation may be resumed through the explicit before/after SHA-256 registry
in `data/scared_a00_identity.py`. Producer identities remain unchanged; resume
checks config, role, source stats, per-frame metrics and every recorded output
hash. Metadata preflight reports `reviewed_positive_gt_compatibility` when it
uses this rule. Unknown code identities or legacy empty targets are rejected.

The real failure was reproduced at
`dataset_2/keyframe_4/frame_data001730`: 2,759 valid left-camera XYZ points
projected to rectified left `u=1300.903..1384.144` on a 1,280-pixel-wide grid.
No point remained in left-image support. Raw-camera reprojection differed from
the scene-point raster indices by at most 0.687 pixels, and KL/KR/DL/DR/R/T
matched frames 0, 1729 and 1731 exactly. The frame JSON SHA-256 was
`50513f51f30893c38ab88e4ced5dc853ce2f4a10bcaa1a4ca6165f45cd1de9f5`;
the scene-point member SHA-256 was
`f235a1dc3a49a7fc085ba0c3484e0757ca20c87593e87b76c56bb5f15a3c3e8c`.
Frame 1731 also had no rectified left support. Frame 1729 retained four GT
pixels; its numeric outputs and validation statistics were exactly equal
before and after this remediation. These checks validate the failure handling
on those frames; they do not claim completion of full preprocessing or training.

## Reciprocal calibration correspondence (2026-10-02)

The failed batch reached `dataset_6/keyframe_1` and exhausted 637 calibration
candidates. On `frame_data000000`, the original one-way query over all valid
left points had median/p90/p99 residuals of 0.081236/0.290307/1.388310 mm.
The configured p99 ceiling is 1 mm. Unequal sampling and surfaces without a
reciprocal counterpart make all valid points an unsuitable correspondence
domain for this calibration gate.

New calibration evidence uses exact reciprocal nearest neighbours under the
declared transform. A left/right pair is retained only when each point is the
other's nearest neighbour. There is no residual cutoff: large residuals on
retained pairs still enter the median/p90/p99 and can fail the gate. At least
three pairs and 25% of each raw valid cloud are required. This is a new explicit
correspondence policy; it does not relax the original absolute-error or control
separation thresholds.

The three alternative-unit/direction controls query those same retained clouds
and are free to select their own nearest neighbours. They are not forced to
use the declared transform's pair assignment. Evidence records
`valid_point_policy: finite_nonzero_xyz_positive_z_mutual_nearest_neighbor`,
the raw counts, mutual-pair count and support requirements. Replay checks both
the counts and residuals. Historical `finite_nonzero_xyz_positive_z` evidence
continues to reproduce using the original algorithm.

Offline checks used the first corresponding archive members of these samples:

| Sample | Mutual pairs | Declared p99 (mm) |
| --- | ---: | ---: |
| dataset_6/keyframe_1/frame_data000000 | 436112 | 0.184831 |
| dataset_7/keyframe_1/frame_data000000 | 383620 | 0.169410 |
| dataset_8/keyframe_1/frame_data000000 | 483968 | 0.162406 |
| dataset_8/keyframe_3/frame_data000000 | 484705 | 0.126216 |
| dataset_9/keyframe_1/frame_data000000 | 460171 | 0.114444 |

All retained domains exceeded the minimum support fraction. The mm candidate
passed the unchanged absolute-error and cm/m/wrong-direction controls; each
sample's frame R/T exactly matched its sibling calibration YAML. On these
samples, maximum Q-vs-A00 depth error was 0.000133 mm and maximum Q-vs-metric-GT
error was 0.000245 mm, below 0.01/0.05 mm. Rectification checks used dummy RGB
and real calibration/scene points; these are offline geometry checks, not real
RGB deployment smoke or full-sequence validation. Final-test data was used
only for protocol verification, not teacher fitting or threshold selection.
Detailed diagnostics and source-member hashes are outside the repository at
`/tmp/reliable-endo-calibration-fix/verified-samples.json` and
`verified-dataset7.json` on the audited runner.

The exact reviewed processor pair is in
`CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR` in `data/scared_a00_identity.py`.
Numeric rectification/disparity/GT/mask construction, archive streaming and
resume file verification are unchanged. The compatibility rule accepts the
preceding empty-GT producer's positive and explicitly unavailable zero-target
outputs, while retaining the older positive-only rule. It preserves all
original producer hashes and rejects unknown code identities.

Read-only metadata preflight passed on the existing positive sample
`dataset_1/keyframe_1/frame_data000000` and empty-GT sample
`dataset_2/keyframe_4/frame_data001730`, with their original provenance unchanged.
Synthetic regression tests cover partial surfaces, failed scale/direction,
large matching residuals, insufficient correspondence support, evidence
tampering, empty-GT retention and automatic batch resume. Full preprocessing,
real A00 deployment smoke, teacher inference and training were not run in this
remediation. Dataset_6–9 outputs still require the external batch resume.
