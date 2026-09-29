# A04 reference reconstruction

The A04 reference path is an independent, quality-first composition:

```text
injected frozen teacher
  -> A01 aligned disparity/validity/support
  -> A00 metric rectified geometry
  -> A03 GeometryPrior and deterministic sampling
  -> bounded reference attribute residual head
  -> GaussianField
  -> injected FieldRenderer
```

`reliable_endo_gs.reconstruction.ReferenceReconstructionModel` owns this
forward boundary. It accepts a live injected teacher or an A01
`AlignedTeacherTarget`, and an injected renderer implementing the independent
`FieldRenderer` contract. FoundationStereo and DEFOM-Stereo therefore use the
same path through A01's neutral `TeacherPrediction`/alignment contracts; no
candidate package or checkpoint is imported by A04.

The final residual projection is zero-initialized, so the initial field is the
A03 priors-only field. Composition uses local-frame right-multiplied
axis-angle rotation, bounded log-scale residuals and bounded alpha-logit
residuals. Invalid primitive masks are propagated separately from opacity;
invalid slots are finite placeholders and are omitted by the renderer.
Numerically sensitive composition is promoted to FP32 for FP16/BF16 inputs.

`ReferenceProvenance` records teacher identity, target/cache/alignment metadata,
calibration identity, A01 data role, supervision versus assessment views,
renderer identity, and resolved configuration hash. A04's default role policy
is the A01 SCARED training/development contract over `dataset_1` through
`dataset_7`; SCARED-C is excluded. Real data and teacher caches remain
external through `RELIABLE_ENDO_DATA_ROOT`.

`save_reference_checkpoint` stores only the independent reference head and
provenance/config identities. Teacher weights and renderer state are explicitly
external. `ReferenceReconstructionModel.profile` delegates to the measured
runtime profiler and accepts synchronization and peak-memory hooks; no latency
optimization or final quality claim is implied by the CPU contract tests.

Injected or cached targets must carry the A00 alignment schema, input-grid
disparity units, disparity sign and calibration metadata. The A04 boundary
detaches those target tensors, requires a resolved teacher identity, and binds
checkpoint loading to the injected runtime teacher identity. Assessment must
include a view held out from supervision; the default trains/observes the left
appearance path and renders both left and right views.

This is engineering implementation only. Teacher qualification, native
renderer parity, measured GPU timing/memory, reconstruction quality, and the
Stage 1 reference gate remain unverified research evidence.
