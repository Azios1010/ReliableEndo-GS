# SCARED-C Data Inspection Report

## 1. Layout
The dataset structure for a sample keyframe is observed at `data/scared_c/dataset_1/keyframe_1`. It contains:
- Stereo image pairs: `Left_Image.png`, `Right_Image.png`
- Depth/Coordinate maps: `left_depth_map.tiff`, `right_depth_map.tiff`
- Point cloud: `point_cloud.obj`
- Calibration data: `endoscope_calibration.yaml`, `intrinsics_colmap.yaml`
- Sequence metadata: `frame_log.json`
- A `data/` subdirectory containing compressed archives (`rgb_frames.tar.gz`, `scene_points.tar.gz`) and an `rgb.mp4` video.

## 2. Sequence, Keyframe, and Frame Meaning
- **Sequence/Keyframe:** The directory hierarchy (`dataset_1/keyframe_1`) organizes data into sequences (datasets) and keyframes. 
- **Frame Meaning:** The `frame_log.json` defines this keyframe (`key: "1_1"`) as a sequence of 197 frames. The `data/` subdirectory contains the full temporal video/frames. The top-level images (`Left_Image.png`, `Right_Image.png`) are a static reference, but their exact index mapping to the 197 frames is **UNRESOLVED**.

## 3. Left/Right Association
- Left and right contexts are distinguished by file names (`Left_Image`, `left_depth_map` vs `Right_Image`, `right_depth_map`). 
- In `endoscope_calibration.yaml`, `M1` and `D1` correspond to the Left camera, while `M2` and `D2` correspond to the Right camera.

## 4. Image Dimensions and Encoding
- Both `Left_Image.png` and `Right_Image.png` are `1280 x 1024` pixels.
- Encoding is 8-bit RGBA (Truecolor with Alpha, PNG color type 6).

## 5. Depth Dtype, Representation, Units, and Invalid Values
- **Dtype:** 32-bit little-endian IEEE float (`float32`).
- **Representation:** The TIFF files are not scalar 1D depth maps. They are 3-channel 3D coordinate maps (X, Y, Z) per pixel.
- **Invalid Values:** Missing or invalid geometry is explicitly represented as IEEE `NaN`.
- **Units:** The baseline is `~4.14` and typical Z depths are `~59`. Numeric plausibility is not evidence of a physical unit; the exact unit is absent from the inspected metadata and is strictly **UNRESOLVED**.

## 6. Independent Right Depth
- The left and right depth maps are separate files with different non-finite-pixel counts (`left_depth_map.tiff`: 283,454; `right_depth_map.tiff`: 256,702). This supports maintaining separate validity masks; whether either map is independently reconstructed or warped cannot be proven from these files alone.

## 7. Calibration Contents and Intrinsic Semantics
- **`endoscope_calibration.yaml`**: Contains original 3x3 camera matrices (`M1`, `M2`) and 1x5 distortion coefficients (`D1`, `D2`).
- **`intrinsics_colmap.yaml`**: Contains a 3x3 camera matrix (`K`) and image dimensions. It does not include distortion coefficients.

## 8. Endoscope vs. COLMAP-Refined Calibration
- The endoscope-calibration matrices (`M1`, `M2`) contain different focal lengths (e.g., Left $f_x \approx 1134.7$).
- The COLMAP refined matrix (`K`) shares the exact same principal point as `M1` ($c_x = 586.08, c_y = 512.47$) but modifies the focal lengths ($f_x \approx 1065.9$). This distinguishes the source files’ numeric matrices; whether the COLMAP model is undistorted, and which view it applies to, remains **UNRESOLVED** from the inspected files.

## 9. Stereo Extrinsics and Baseline
- The `endoscope_calibration.yaml` stores a rotation matrix `R` (near identity) and a translation vector `T = [-4.143, -0.023, -0.0019]`. 
- The calibration file contains `T = [-4.14339018, -0.0238197, -0.00190685]`. Its transform direction, units, and interpretation as a metric baseline are **UNRESOLVED** from the inspected files alone; the selected development loader does not apply it.

## 10. Pose Representation, Direction, and Coordinate Frame
- Parsing the first valid top-left pixel (u=0, v=0) of the left coordinate map yields `(X = -33.79, Y = -29.44, Z = 58.61)`. 
- The first inspected finite triple has positive third component and negative first/second components at the top-left pixel. Those values are consistent with an OpenCV-like frame, but numeric sign patterns alone do not prove axis orientation or handedness; camera axes remain **UNRESOLVED**.

## 11. Rectification
- The exact rectification state is **UNRESOLVED**. Projecting the 3D TIFF coordinates back to pixel coordinates does not linearly map perfectly to `(u, v)` using either the raw `M1` or the refined `K` matrix. Thus, the exact mathematical transformation that associates the image pixels to the provided 3D coordinates cannot be proven from the files alone.

## 12. Selected development protocol
- **Protocol:** `scared_c_endoscope_stereo_calibration_v1`.
- **Calibration/intrinsics:** the endoscope file's `M1`/`M2` pair and available stereo fields are recorded; the separate COLMAP `K` is not substituted.
- **Pose:** per-frame `camera-pose` values are recorded as available metadata but are not applied because direction and coordinate frame are unresolved.
- **Depth:** both top-level XYZ TIFF maps are loaded independently with finite-value masks; units, axes, rectification, and exact top-level temporal mapping remain unresolved.
