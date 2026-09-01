# Scientific contract specification

## Scope and status

Plan 00 implements the Phase I structural contracts `CameraBatch`, `StereoBatch`,
`StereoPrediction`, `GaussianField`, `RenderOutput`, and `ReconstructionState`
under `reliable_endo_gs.contracts`. Region, action, cost, and router contracts
remain **PLANNED**. No tensor loader, model, geometry formula, renderer, or
training package is implied by the implemented containers.

Contracts isolate scientific meaning from the pinned baseline, renderer
implementation, training framework, and routing policy. The implemented Phase I
types are frozen, slotted dataclasses with lightweight structural validation.

## Versioning and compatibility

Each persisted or exchanged contract carries a schema name and schema version. Compatible additive fields may increment a minor version. A breaking change to shapes, coordinate conventions, units, validity meaning, covariance meaning, or feature semantics requires a schema version bump and an explicit migration or rejection path.

Implemented classes expose `SCHEMA_NAME` and `SCHEMA_VERSION = "1.0"` as class
metadata. Serialization and migration are deferred until an artifact milestone
requires them.

The ReliableEndo-GS package version is not a substitute for contract schema versions. Consumers validate supported schemas before use. Missing provenance or incompatible semantics are errors, not warnings followed by best-effort guessing.

Notation used below:

- `B`: batch size;
- `H`, `W`: image height and width;
- `N`: number of Gaussians, potentially padded per batch;
- `K`: number of baseline disparity iterations or evidence levels;
- `R`: number of regions.

Floating-point dtype and device must be preserved or converted explicitly at component boundaries. Shape alone never establishes a coordinate convention.

Implemented structural validation reads tensor metadata only. It does not scan
tensor values, synchronize accelerators, cast dtypes, or move devices. Coupled
tensors share a device. Numeric tensors use a floating dtype and masks use
`torch.bool`; no particular floating precision is forced.

## CameraBatch

`CameraBatch` describes calibrated cameras associated with a batch.

| Field | Required | Shape | Semantics |
| --- | --- | --- | --- |
| `intrinsics` | Yes | `[B, 3, 3]` | Camera intrinsic matrices under an explicitly named pixel convention. |
| `world_from_camera` | Yes | `[B, 4, 4]` | Homogeneous transforms from camera coordinates to world coordinates. |

The pixel-center convention, image-resize relationship, handedness, axis
orientation, transform multiplication convention, and units are deliberately
**TBD until verified against the pinned baseline and datasets**. Once chosen,
they must be explicit in schema metadata and tests. Camera identity currently
comes from `StereoBatch` sample/sequence association; an explicit camera-ID field
is deferred until a concrete producer requires it.

## StereoBatch

`StereoBatch` is the baseline-neutral input contract.

| Field | Required | Shape | Semantics |
| --- | --- | --- | --- |
| `left` | Yes | `[B, 3, H, W]` | Left/reference images with explicit range and normalization metadata. |
| `right` | Yes | `[B, 3, H, W]` | Right/source images aligned by sample identity, not necessarily rectified unless declared. |
| `left_camera` | Yes | `CameraBatch` | Left camera calibration. |
| `right_camera` | Yes | `CameraBatch` | Right camera calibration. |
| `sample_ids` | Yes | `[B]` logical values | Stable dataset sample identifiers. |
| `sequence_ids` | Yes | `[B]` logical values | Sequence or patient grouping identifiers used for split integrity. |
| `gt_disparity` | Optional | `[B, 1, H, W]` | Ground-truth disparity under the declared disparity convention. |
| `gt_depth` | Optional | `[B, 1, H, W]` | Ground-truth metric depth under the declared camera convention. |
| `gt_depth_xyz` | Optional | `[B, 3, H, W]` | Ground-truth metric 3D point coordinates. |
| `gt_right_depth_xyz` | Optional | `[B, 3, H, W]` | Ground-truth right-camera 3D point coordinates when independently provided. |
| `masks` | Optional | named collection of `[B, 1, H, W]` | Explicit input/supervision masks; each mask has a declared meaning rather than one inferred from numeric values. |
| `metadata` | Optional | structured mapping | Additional sample or sequence metadata frozen at batch creation. |

Optional targets are absent when unavailable, rather than fabricated with zeros. Dataset adapters establish IDs and metadata; future tensor loaders perform decoding and normalization explicitly.

## StereoPrediction

`StereoPrediction` is the normalized output of the baseline adapter.

| Field | Required | Shape | Semantics |
| --- | --- | --- | --- |
| `disparity` | Yes | `[B, 1, H, W]` | Final reference-view disparity. |
| `valid_mask` | Yes | `[B, 1, H, W]` | Pixels valid for downstream geometry. |
| `disparity_iterations` | Optional | `[B, K, H, W]` or an explicit sequence of `[B, 1, H, W]` | Baseline-exposed iterative estimates used as evidence when available. |
| `sigma_d` | Optional | `[B, 1, H, W]` | Calibrated disparity uncertainty with a declared parameterization and units. |
| `diagnostics` | Optional | structured mapping | Adapter-normalized evidence with named, versioned semantics. |

The contract does not require RAFT-specific correlation volumes, hidden states, feature pyramids, or module classes. Such internals remain private to the baseline adapter. If evidence is unavailable from a pinned upstream version, the field is absent and consumers follow a declared fallback.

`sigma_d` must state whether it is a standard deviation, variance, log-scale, interval-derived value, or other representation. Downstream geometry accepts only the canonical calibrated form.

## GaussianField

`GaussianField` represents a batch of scene Gaussians.

| Field | Required | Shape | Semantics |
| --- | --- | --- | --- |
| `means3d` | Yes | `[B, N, 3]` | Gaussian centers in a declared coordinate frame. |
| `colors` | Yes | `[B, N, C]` | Color or appearance coefficients with explicit basis and range. |
| `rotations` | Yes | `[B, N, 4]` or versioned equivalent | Normalized orientation representation with declared ordering. |
| `scales` | Yes | `[B, N, 3]` | Strictly positive support scales under a declared parameterization. |
| `opacities` | Yes | `[B, N, 1]` | Opacity values or logits with the parameterization stated. |
| `valid_mask` | Yes | `[B, N]` | Marks real, usable Gaussians independently of padding. |
| `cov_surface` | Optional | `[B, N, 3, 3]` | Spatial support covariance derived from Gaussian rotation and scale. |
| `cov_center` | Optional | `[B, N, 3, 3]` | Uncertainty of estimated center location propagated from disparity uncertainty. |
| `cov_effective` | Optional | `[B, N, 3, 3]` | Consumer-specific effective covariance produced by the owned marginalization rule. |

The three covariance fields are not aliases. `cov_surface` describes the represented surface element; `cov_center` describes epistemic or measurement uncertainty about its location; `cov_effective` is a derived quantity with formula version and stabilization metadata. A renderer must declare which covariance it consumes.

Invalid entries must remain masked. Nonpositive scales, invalid rotations, non-finite values, or covariance matrices that violate the configured numerical policy are contract violations or explicitly reported filtered cases.

Plan 07 keeps the covariance meanings explicit: `cov_surface` is intrinsic
Gaussian support, `cov_center` is the Plan 06 center-position uncertainty, and
`cov_effective = cov_surface + cov_center` is the per-primitive covariance
requested by the probabilistic representation variant. The determinant-ratio
opacity correction applies only to actual opacity values, not logits. A
renderer request names `surface`, `effective`, or `none`; the local request
validator does not claim production/native renderer parity.

## RenderOutput

`RenderOutput` is backend-neutral output for one requested view.

| Field | Required | Shape | Semantics |
| --- | --- | --- | --- |
| `image` | Yes | `[B, C, H, W]` | Rendered image with declared color space and range. |
| `depth` | Optional | `[B, 1, H, W]` | Rendered depth under the requested camera convention when the backend exposes it. |
| `visibility` | Optional | batch-leading pixel- or Gaussian-level tensor | Explicit visibility or contribution evidence; exact representation is deferred to a concrete renderer. |

Production GPU and deterministic CPU reference renderers must satisfy the same public contract for the supported subset. Numerical tolerances may be backend-specific and declared in tests.

## ReconstructionState

`ReconstructionState` is the terminal public output of Phase I and the sole reconstruction input to Phase II.

| Field | Required | Semantics |
| --- | --- | --- |
| `batch` | Yes | The `StereoBatch`. |
| `stereo` | Yes | The normalized `StereoPrediction`. |
| `gaussians` | Yes | The current `GaussianField`. |
| `render_left` | Yes | Reference-view `RenderOutput`. |
| `render_right` | Optional | Cross-view `RenderOutput` when available. |
| `diagnostics` | Yes | Named Phase I evidence, calibration summaries, and validity counts. |
| `provenance` | Yes | Artifact IDs, schema versions, configuration hash, code revision, split/sample identity, and backend identities. |

This state contains no router, routing decision, action label, budget allocation, or oracle utility. A repair action may return a new post-action state, but it must not mutate the input state in place.

## RegionState

`RegionState` describes Phase II regions and raw evidence derived from a frozen reconstruction state.

Required conceptual categories include:

- region identity and membership geometry;
- validity and visibility coverage;
- stereo/disparity evidence;
- uncertainty and covariance evidence;
- reconstruction or cross-view residual evidence;
- Gaussian density, opacity, or support evidence;
- provenance connecting the region to the source reconstruction state.

The exact raw tensor/vector schema is deliberately not frozen before Phase I evidence is validated. Raw evidence and encoded router features are separate objects with separate schema versions. `RegionState` must not contain an oracle-selected action as an input feature unless an explicitly leakage-safe analysis contract says it is a target.

## RepairAction

`RepairAction` is an independently callable action interface. Its future semantic interface is:

```text
stable_identity
apply(state, regions) -> ActionResult
estimate_cost(state, regions) -> CostEstimate
```

Required stable identities include `STOP`, `STEREO_REPAIR`, and `GAUSSIAN_REPAIR`. STOP returns an explicit no-change result and still participates in utility comparison. Stereo repair changes only declared stereo-derived fields before rebuilding dependent geometry/rendering. Gaussian repair changes only declared Gaussian-stage fields and rerenders affected outputs.

`estimate_cost` predicts or retrieves cost without executing hidden repair mechanics. Its state or action configuration carries any declared hardware/profile context. The `profiling` component owns measured runtime; estimates are calibrated against those measurements. The same `apply` implementation is used for oracle data generation, action training evaluation, routed runtime, final evaluation, and profiling.

## CostEstimate

`CostEstimate` is a versioned prediction of resource usage, such as latency, memory, or normalized compute units, for a named action and hardware/profile context. It includes estimator identity, units, scope, and uncertainty when available. It is never silently substituted for measured cost in final reporting.

## ActionResult

| Field | Required | Semantics |
| --- | --- | --- |
| `action` | Yes | Stable action identity and implementation version. |
| `affected_regions` | Yes | Region identities the action was authorized to modify. |
| `post_state` or `post_output` | Yes | New immutable state/output after action execution. |
| `changed_fields` | Yes | Explicit paths or semantic categories that changed. |
| `measured_cost` | Yes for oracle/evaluation | Profiled cost with units, hardware context, warm-up policy, and measurement method. |
| `diagnostics` | Yes | Success/failure, validity, and action-specific evidence. |
| `provenance` | Yes | Input artifact/state identity, configuration, code revision, seed, and backend identity. |

An action failure is represented explicitly. It must not fall back to another action without recording a separate policy decision.

## Router and RoutingDecision

The future router interface is:

```text
predict(features_or_state, budget) -> RoutingDecision
```

The accepted feature form is versioned; a raw `RegionState` may be encoded by a separate feature encoder. `RoutingDecision` contains selected action identities, selected regions, predicted costs/utilities as applicable, budget accounting, and router provenance.

The router does not call the baseline, update disparity, modify Gaussians, render, or measure oracle utility. A runtime coordinator validates the decision and invokes the chosen `RepairAction`.

## Coordinate and geometry policy

The Plan 06 analytic development path records an explicit, provisional
convention in `GeometryConvention` and `configs/geometry/stereo_geometry.yaml`:
positive left-reference disparity in resized pixels, integer pixel centres,
axial depth `D = f_x B / d`, camera-frame rays `K^-1 [u,v,1]^T`, metres, and
left-camera means/covariances with `x` right, `y` down, and `z` forward. This
development convention is covered by analytic fixtures and is not a claim
that the real SCARED-C resize, rectification, baseline, units, or camera-frame
semantics have been verified. Those values remain pending the baseline/data
convention audit before real-data validation.

Before geometry implementation, the following values must be resolved and stored as contract metadata:

- whether integer pixel coordinates refer to centers or corners;
- the definition of horizontal disparity and its sign;
- which image is the disparity reference;
- whether disparity is measured in original or resized pixels;
- stereo baseline vector direction and metric unit;
- camera axes and handedness;
- whether transforms are `world_from_camera` or `camera_from_world` and how vectors multiply them;
- whether depth is axial camera depth, range, inverse depth, or another quantity;
- the frame of Gaussian means and all covariance matrices.

The owner modules in `docs/architecture.md` perform conversions. Geometry tests use analytic cameras, known baselines, and hand-computable points before dataset examples are accepted.

## Mutation, masking, and diagnostics

Scientific contract containers are frozen at component boundaries and copy
metadata mappings into read-only views. They intentionally reference tensor
objects without cloning storage, so callers must not mutate those tensors in
place. Functions return new contract values and report changed fields. Validity
masks are explicit and propagate through derived quantities. Empty masks and
zero-valid-element batches are supported as explicit outcomes or rejected with
a clear typed error; they never produce a plausible-looking aggregate from an
accidental denominator clamp.

NaN, infinity, singular covariance, and out-of-range parameters are detected at their owning boundary. Any stabilization rule records its configured threshold and affected count in diagnostics and provenance.
