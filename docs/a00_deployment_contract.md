# A00 deployment foundation

Status: IMPLEMENTED CPU foundation; native/GPU acceptance remains pending.
A01 teacher/cache, A03 Gaussian priors and A04 reference code build on this
contract. This document records the A00 boundary and does not claim native or
scientific qualification.

## Inputs and geometry

`contracts/deployment.py` exposes `DeploymentStereoBatch` and
`RectifiedStereoCamera`. Deployment RGB is floating [B,3,H,W] in [0,1].
Calibration is FP32/FP64, metre translations, integer pixel centres, x-right,
y-down, z-forward. Input must already be distortion-corrected and rectified;
the adapter validates the narrow horizontal stereo geometry, not image content.

`deployment_from_stereo` allowlists RGB and calibrated cameras from the existing
container. It copies no GT, arbitrary masks, sequence metadata or future signals.
Default support is all true; callers with rectification padding must explicitly
supply sensor support. A boolean tensor cannot prove its own provenance.
Tensors are borrowed, not immutable snapshots; callers must not mutate live inputs.

`T_R_from_L = inverse(world_from_right) @ world_from_left`. It has identity
rotation and negative x translation for a right camera at positive baseline B.
Both focal lengths and vertical principal points must match. Horizontal principal
points may differ. Nonrectified, skewed, reversed-baseline or unequal-focal inputs
are rejected rather than silently approximated.

`geometry/rectified.py` owns raw-to-effective disparity correction once:

```text
d = u_L - u_R, in current rectified input-grid pixels
d_eff = d - (cx_L - cx_R)
D = fx * B / d_eff
mu = D * inverse(K_L) * [u, v, 1]
```

Depth/backprojection call the existing canonical metric implementations. There
is no depth normalization, batch reduction or baseline-model dependency.
Invalid/unsupported geometry is masked NaN; the renderer must receive a valid
field, not raw invalid centres. Mixed-precision disparity is promoted to the
calibration precision with gradients preserved.

Paired crop/resize helpers use `align_corners=False` pixel-centre semantics:
`u' = sx*(u-crop_x+0.5)-0.5`. Raw disparity values become
`sx*(d-left_crop_x+right_crop_x)`. Helpers transform calibration and already
resampled disparity values; image resampling, crop bounds and validity-aware
interpolation remain the preprocessing caller's responsibility. Historical
calibration helpers and their coordinate conventions are not changed.

## Rendering boundary

`rendering/field.py` defines `FieldRenderRequest` / `FieldRenderer`, accepting
first-party `GaussianField` and explicit rectified cameras. Fields use left-camera
metric centres, normalized wxyz rotations, positive standard-deviation scales,
RGB and alpha in [0,1]. Optional covariance fields are rejected in this initial
R/S-only path rather than guessed or marginalized.

`ExistingRendererBridge` sanitizes invalid slots without mutating the source and
adapts to existing first-party renderers. Legacy representation enums are confined
to this compatibility bridge; its deterministic tag does not instantiate or
import a baseline model. Existing rendering package imports still load historical
protocol definitions; this is not a claim that all legacy package coupling is gone.
CPU reference rendering uses a fixed screen-space kernel: it checks tensor/camera
flow, NOT native anisotropic splatting, scale gradients or production parity.
Native rendering still requires validated native camera settings and backend
precision support. Native depth output is not promised.

## Runtime protocol

`configs/reconstruction/runtime.yaml` is loaded by the dedicated strict
`config/deployment.py` loader, not by the historical experiment-config loader.
It is configuration only, not a training or benchmark runner. Reference settings
are RTX 5070 Ti 16 GB, batch 1, FP16 (BF16 selectable), explicit FP32 geometry,
synchronized timing, eager mode, 20 warmups and 1000 repetitions. All are recorded
settings, not measurements or evidence that hardware is available. Hardware is
configurable; another device must be reported as a separate comparison protocol.
Resolution and deadline are explicit in the current configuration and are
checked by `require_benchmark_ready()`. They are protocol settings, not
measurements. Renderer precision islands must be established by native
validation.

Future full-frame timing starts before preprocessing/transfers and ends after the
required output render is complete. Report component timing separately; do not
sum overlapping timings. Record backend/driver/software, config hash, compilation
and warmup policy, raw per-frame timings, peak allocated/reserved VRAM and all
resident buffers. Mean/median/p95/p99, FPS and deadline misses need a measured
runner/aggregation protocol; the existing profiler's p90 alone is insufficient.
Teacher runtime is offline accounting, not a deployment KPI.

## Validation and remaining gate

Focused CPU tests cover fronto-parallel/slanted planes, principal-point offset,
right projection, crop/resize, invalid disparity/support, mixed precision
promotion, gradients, batch isolation, GT exclusion, field validation, reference
rendering and fresh-process blocking of baseline/lib/third_party imports.

Recorded validation: 109 tests passed across the new A00 suite and existing
`tests/contracts`, `tests/geometry`, `tests/rendering`, `tests/profiling`.
Scoped Ruff checks passed; scoped mypy passed for all four new source modules.
The initial run hit Windows permissions on existing pytest temp/cache folders;
the successful runs used unique temporary roots and disabled pytest's cache.
No GPU/native test, training or research experiment was run.

Native camera/rasterizer parity, GPU FP16/BF16 support, final resolution/deadline,
performance measurements and dataset-specific rectification/support provenance
remain unresolved. Sampling policy and student-output contracts belong to their
subsequent bounded implementation tasks; no sampling/prior model was added here.

A01, A03 and A04 execution remain subject to their own external teacher,
dataset and native-renderer qualification gates. CPU reference rendering alone
does not establish GPU/native acceptance. No research gate or efficiency claim
has passed.
