# Controlled Geometry–Attribute Decoupling Ablation for GaD

Date: 2026-11-12. Status: **DESIGN ONLY — pending approval for implementation and execution**.

This document is the only file created for this task. No training code, configuration, manuscript, checkpoint or experiment output is changed. No training, evaluation, inference, GPU probe or benchmark was executed. Source citations refer to the inspected working tree at HEAD `9da799be696fcf18a2545906badcdf94559d2d31`; the tree contains pre-existing modifications, so HEAD alone does not identify all inspected bytes. A later implementation must archive the approved source snapshot and resolved configurations.

## 1. Executive summary

**Recommended primary experiment:** compare freshly trained fixed-centre attribute refinement (B) with feed-forward joint centre–attribute refinement (C), starting from the **same frozen clean10 student and initial Gaussian head**. The initial field (A) is their common reference. Neither branch retrains the student/head or learns source colours.

Use the existing 11-input, 32/32-hidden, seven-output attribute refiner. In a new opt-in wrapper, add a zero-initialized `Linear(32,3)` centre head on the same hidden features. C predicts bounded XYZ offsets in the left camera coordinate system, in metres. B bypasses this head and retains the exact existing computation. Attribute architecture, initial weights, source correspondence, field count, loss, training order and budget remain matched. The centre head adds **99 active parameters**, giving a proposed count of **1,770**, versus **1,671** in B; these are architecture-derived counts to verify in implementation, not measured performance results. [S3, S4]

Predeclare C's primary maximum displacement as **5 mm**, no displacement penalty, with a camera-plane safety bound. This is a proposed engineering setting, not an established optimal or clinically meaningful threshold. Validation-only 1/10 mm and learning-rate checks protect against a misleadingly weak C. Do not choose settings from datasets 8–9.

Train B and C for **13,150 updates each**, on the existing `train_1236` roster, for **three paired refinement seeds**. Reuse A once. Keep the existing trained main refiner as a separately labelled historical anchor, not as the only B comparator. Report per-frame paired differences and keyframe-cluster sensitivity, plus variation across seeds.

Measure four distinct properties:

1. **Coordinate preservation:** whether a Gaussian remains at its original stereo centre.
2. **Centre accuracy:** whether its coordinates agree with GT at its retained source-pixel identity.
3. **Rendered geometry:** how alpha-composited apparent depth changes.
4. **Rendering fidelity:** PSNR, windowed SSIM and LPIPS in both views.

The current production renderer does **not** export depth or per-pixel accumulated opacity. Its visibility output is Gaussian radii, and the deployed predictor drops source indices. Therefore direct centre metrics and depth diagnostics require explicit additions. A proposed auxiliary RGB moment render can estimate **alpha-composited centre depth**, subject to validation; it must not be described as an existing native depth output or a certified continuous surface. [S6–S8]

Expected planning budget: approximately **5–8 GPU-hours for six primary training runs**, plus **2–6 GPU-hours** for full quality/geometry evaluation, pilots and timing allowances. Reserve **10–16 GPU-hours** on one RTX 5070 Ti-class device, excluding implementation effort and optional extra full-budget controls. These are estimates with assumptions in Section 10, not new measurements.

## 2. Current implementation findings

### 2.1 Confirmed representation and training boundaries

| Capability | Confirmed implementation/artifact | Consequence for this study |
| --- | --- | --- |
| Student disparity is trainable during initial training | `reconstruction/student.py:297–305,327–350`; `training/student_learning.py:544–613` | Initial training is already joint stereo/head learning; this ablation concerns the subsequent stage only. |
| Calibrated lifting | `geometry/rectified.py:22–53` subtracts left/right principal-point offset once, then calls canonical depth/backprojection | Reuse calibration and metre-valued centres; no new depth convention. |
| Pixel correspondence | `gaussians/sampling.py:66–83`; `gaussians/priors.py:374–405,407–431,500–515` | Each valid Gaussian has an immutable source index and source pixel. |
| Source colours share the same sampling | `gaussians/priors.py:458–467` | Retain colour assignment even when C moves a centre. |
| Attribute composition preserves prior centres | `gaussians/composition.py:284–306` | Existing seven outputs are rotation, scale and opacity, not XYZ corrections. |
| Refiner architecture | `gaussians/refinement.py:22–46,92–117,134–165` | Zero output initialization, shared pointwise MLP, bounded attributes; no centre output exists. |
| Refinement uses the already composed field as its prior/context | `actions/refinement.py:84–103` | Start both B/C from the same base field, not from raw geometric priors alone. |
| Base freeze and optimizer isolation | `training/field_refinement.py:105–117,197–228,246–259` | Student/head run under no-grad; only refiner learns; frozen state hashes are checked. |
| Two-view objective | `training/interventions.py:31–63` | Mean left/right Charbonnier over fixed deployment masks; right support is not GT validity. |
| Existing identity correction preserves gradients | `actions/repair.py:145–174` | Reuse exact-zero identity handling; do not return a detached field at zero offsets. |
| Rendering preserves coordinate autograd and metre semantics | `rendering/production.py:489–526` | Native ABI applies uniform metre-to-native scaling once; offsets must be applied before this boundary. |
| Main RGB metric implementation | `evaluation/multi_model.py:81–129`; `evaluation/rendering.py:40–66,112–149` | Full-image PSNR, Gaussian-window SSIM, pinned LPIPS; not the older global SSIM helper. |

All relative source paths in this document are under `src/reliable_endo_gs/`, unless otherwise marked. See Section 15 for links and artifact roots.

### 2.2 Verified main artifact specifications

- Student: `/mnt/d/3DGS_runs/reliableendo-gs/clean10-20261103/kd/FINAL.pt`; recorded SHA-256 `db7413e8d456177ee55eb7e3cb75580b5cfd40c443403359f8173fb21a6131b6`. `kd/attempt-00/result.json` records 10 epochs, 131,500 optimizer updates and `schedule_complete:true`.
- Main refiner: `/mnt/d/3DGS_runs/reliableendo-gs/clean10-migration-20261106T143532+0700/refiner/refiner.pt`; recorded SHA-256 `aa6736393b15cad40d59dc834ca18e7f2a19817979d972bd3a541610e34fc9a6`.
- Its `spec.json` records 11 channels: RGB, centre Z, base normal XYZ, base scales XYZ and opacity. Hidden widths 32/32, no colour residual, rotation bound 0.7 rad, log-scale bound 1.5, opacity-logit bound 4.0; min/max scale `1e-6/0.5` metres and alpha epsilon `1e-4`.
- `summary.json` records 1,671 trainable parameters, 13,150 updates, `frozen_unchanged:true`, 2,909.964 training seconds and 3,183.500 wall seconds.
- The original launcher constructs these bounds by multiplying base bounds by `bounds_scale:2`. The loader also derives bounds from the base model: `evaluation/ours.py:128–154`. **Do not multiply the already resolved 0.7/1.5/4.0 values by two again.**
- Migration protocol lines 8–23 and launcher `scripts/train_refiner.py:48–79` in that external artifact specify datasets 1/2/3/6, a seeded permutation of 13,150 samples, LR 0.001, clipping 1.0, final checkpoint and no early stopping. Validation is dataset 7 (56 frames); final test is datasets 8–9 (5,907 RGB frames, 5,807 geometry-valid frames). [A1–A3]

Historical saved main means verify EPE 2.102800869 px and mean per-frame depth RMSE 5.596403262 mm, identical between base and refined. Right PSNR gains are approximately +0.89/+1.22 dB on datasets 8/9. These are context, **not expected outcomes of newly trained B/C**. The 21.237152 ms saved timing covers resident-input stereo + field + both renders on RTX 5070 Ti, FP32 Ours with CUDA graph, excluding I/O/decode/H2D. [A3, A4]

### 2.3 Historical evidence and its limits

Original native artifacts confirm a joint128→1152 development continuation with depth MAE 13.238349→16.095515 mm, RMSE 39.843979→54.940627 mm, and right PSNR 19.385441→19.513953 dB. Joint459 versus frozen459 gave MAE 10.436251 versus 9.272147 mm, but RMSE 15.645196 versus 15.696837 mm. These are compact student/repair/selector studies on 140 development images with known backbone ancestry overlap. They do not isolate independent centre offsets. [A5]

The three-epoch controls also matter: KD+photometric EPE 3.446/3.575/3.623 px versus KD-only 3.648/3.752/3.695 px across three seeds; photometric-only had EPE 51.653 px and predicted mean disparity 0.25 px in raw records. Thus joint photometric training is not uniformly harmful. These controls evaluate disparity geometry only. [A6]

The preceding audit is supporting context: [joint optimisation audit](docs/gad_joint_optimisation_audit_2026-11-12.md). Important mechanisms above were checked directly against source and saved specifications. No direct matched fixed-centre versus independently movable-centre refinement result was recovered.

### 2.4 Reusable versus missing capabilities

**Reuse:** stereo/data/calibration contracts; canonical lifting; source sampling; attribute bounds/composition; frozen-base training pattern; two-view loss; LPIPS weights and RGB evaluators; checkpoint hashes; existing paired and cluster statistics; static-grid inference helpers.

**Missing/proposed:** a persistent feed-forward centre correction head; a shared train/inference correction adapter; checkpoint schema binding offsets/bounds; correspondence-aware centre metrics; source-index access in deployed inference; an explicit depth-compositing diagnostic; C-capable graph capture and parity checks. Existing centre Chamfer is nearest-neighbour cloud comparison, not source-correspondence error. `evaluate_geometry` accepts optional rendered depth, but production output supplies none. [S6–S9]

## 3. Precise research hypotheses and estimands

**H1 (constructive):** B can improve two-view rendering relative to A while retaining exactly the stereo-derived centres. Coordinate preservation is a structural invariant; rendering gain is empirical. Retaining centres implies equal centre-to-GT accuracy on identical correspondence/support, not that those centres were accurate initially.

**H2 (comparative, non-directional):** relaxing the fixed-centre constraint in C changes the attainable rendering/centre-accuracy trade-off under matched resources and training budget. C may improve, degrade or leave either metric unchanged. A null difference is valid evidence; no finite study can establish universal equivalence.

**H3 (surface diagnostic):** changing attributes can alter alpha-composited apparent depth even with fixed centres. The experiment tests this separately; neither H1 nor unchanged student EPE establishes surface preservation.

Primary paired response vector, C minus B, per seed and per frame:

- Right full-image PSNR (higher is better).
- Source-correspondence 3D centre MAE in mm (lower is better).

Co-primary interpretation is a trade-off vector, not an arbitrary weighted sum. Left RGB, SSIM/LPIPS, centre depth MAE/RMSE, movement percentiles, coverage and composited-depth diagnostics are secondary. Report every direction and confidence interval. A claim of improved rendering must acknowledge contradictory SSIM/LPIPS or left-view effects.

Treat **refinement training seed** as the replication unit for optimization variation and **keyframe** as the spatial/temporal sampling cluster. All seeds share one frozen student: conclusions are conditional on that base artifact, not across independently trained stereo models. Thousands of correlated frames are not thousands of independent training replicates.

## 4. Proposed experimental configurations

| Arm | Student / initial head | Centres | Attributes | Source colours | Refinement parameters |
| --- | --- | --- | --- | --- | ---: |
| A: Initial field | Frozen | Base calibrated centres | Initial composed values | Fixed | 0 |
| B: GaD | Frozen | Exactly base centres | Scale, rotation, opacity | Fixed | 1,671 |
| C: Centre + attributes | Frozen | Base + feed-forward XYZ offset | Same scale, rotation, opacity architecture | Fixed | 1,770 proposed |

**Primary comparison: newly trained B versus newly trained C.** Both begin at A with zero outputs; do not warm-start C from a converged B while B receives no equal extra training.

Use a shared 11→32→32 GELU trunk and the unchanged 32→7 attribute output. Add an independently named 32→3 output module for C. This is an additive wrapper/new class, not a change to the old seven-output checkpoint layout. Deep-copy the same initialized trunk/attribute output for paired B/C, then attach the zero centre output; avoid RNG consumption changing their shared weights.

The existing trained B is evaluated only as `B_historical` if needed to bridge to main-paper numbers. It is not one of three newly trained seeds. Exact reproduction of seed20261006 can be checked against that historical artifact, but parity failures must be reported and resolved structurally, not hidden by substituting historical scores.

No action selector, stereo repair, region selection, teacher inference, GT loss, opacity pruning, densification, grid override or post-render edge guard participates. One correction is applied to the complete base field and used for both cameras. [S4, S5]

## 5. Trainable-centre parameterization

### 5.1 Alternatives considered

| Parameterization | Definition, coordinates and units | Initialization / constraints | Capacity and assumptions | Interpretation |
| --- | --- | --- | --- | --- |
| **Left-camera XYZ (primary)** | `mu' = mu + delta`, delta is a 3-vector in metres, left camera frame | Zero final head; norm-bounded offset; positive-depth/near-plane safeguard | Three outputs, +99 parameters on shared trunk; no assumed correction direction | Most general local centre movement; includes lateral and depth errors; preserves feed-forward inference. |
| Viewing-ray | `mu' = mu + t r`, unit ray `r = mu/||mu||`, t in metres | Zero scalar head; bound t, keep positive depth | One output, +33 parameters; left pixel ray is retained | Tests range/depth correction; cannot discover useful lateral movement. Equivalent to depth change only after accounting for ray Z. |
| Surface-normal | `mu' = mu + t n`, fixed unit normal from the base geometric prior, t in metres | Zero scalar; bounded signed t; explicit policy for invalid/fallback normals | +33 parameters; assumes useful motion is normal to the estimated surface | Sensitive to noisy normals/discontinuities; using learned refiner rotation to update n would introduce extra coupling and is excluded. |
| Disparity-space | Predict a scalar correction in current-grid pixels and call canonical calibrated lifting at the same source pixel | Zero scalar, bound effective disparity away from zero; explicit pixel bound or derived metre bound | +33 parameters; keeps left-ray anchor and nonlinear depth sensitivity | Useful stereo-shaped geometry control, but changes field construction, may change validity/priors if naively recomputed; corrected field disparity must be distinguished from frozen student output. |

All variants generate **one common 3D field** rendered by the calibrated left/right cameras; none uses per-view offsets or test-time scene optimization. Pixel-space offsets require calibration to convert to metric length; metre-based offsets require no change to scale units. Ray/normal/disparity variants are optional follow-ups, not replacements selected after seeing final-test performance. Their different output dimension and representational restriction must be disclosed.

### 5.2 Primary XYZ definition

For base valid slot i, frozen context x_i and hidden feature h_i:

`v_i = W_mu h_i + b_mu`, shape `[B,N,3]`, dimensionless.

`delta_i = R_i * v_i / sqrt(1 + sum(v_i^2))`, shape `[B,N,3]`, metres.

`mu'_i = mu_i + delta_i` for C; `mu'_i = mu_i` for B.

This smooth radial bound is differentiable at zero and ensures `||delta_i|| < R_i` without a zero-norm square-root derivative. Initialize W_mu and b_mu exactly to zero. The proposed primary global cap is R=0.005 m. This formula is a **new canonical transformation to implement later**, not code that currently exists.

**Camera-plane safeguard:** compute once from the detached base centres. For both cameras, let Z_iv be centre depth in metres and let near/far be the renderer's actual recorded metric planes. Set R_i to the minimum of R and half each positive near/far margin. A unit-vector depth projection changes by at most the Euclidean offset norm, so this bound keeps an originally interior centre inside both depth ranges. Slots already outside either camera's admissible depth range get R_i=0; retain them and their original validity rather than silently filtering them from centre metrics. Log the count of such slots and safety-limited radii. Do not impose an image-frustum bound: lateral movement into/out of image support is a possible effect to measure.

Use the camera extrinsics for right-camera depth, not a hand-coded assumption that both Z coordinates match. Apply offsets in first-party metres before native unit scaling. Do not use min–max depth normalization, per-frame fitted scales or evaluation alignment. [S1, S6]

**Primary regularization:** none (`lambda_move=0`). The hard bound already states the feasible geometry change. A penalty that favours zero movement could manufacture the result. If the pilot reveals unstable unconstrained behavior, retain it and add a separately labelled regularized sensitivity: `Lmove = mean_valid(||delta/R_i||^2)` on R_i>0 with proposed lambda0.001. It does not replace the unregularized primary after observing test results.

### 5.3 Composition order and model inputs

1. Run frozen base under no-grad; obtain field, source indices, base prior and cameras.
2. Build the **same frozen base context** used by B from the already composed field, including its base quaternion-derived normal. Select exactly the main 11 channels. [S3, S4]
3. Compute shared hidden features once; predict bounded seven attribute residuals and C's offset vector.
4. Compose attributes against the base field using the existing canonical composer; preserve exact zero-residual identity and derivatives.
5. Replace only centres with `base.means3d + delta`; retain colours, source indices, slot order and validity. Covariance remains the one determined by the refined rotation/scales; translation does not require covariance recomputation.
6. Render the resulting field through both cameras. Continue to expose original student disparity separately for stereo metrics.

Do not rebuild priors/normals/scales from moved centres or feed moved coordinates back into the refiner. That would change more than centre freedom. Invalid padded slots have exactly zero offset and retain their base values. Full-frame composition must not be undone by a helper that assumes Gaussian repair cannot change geometry. Review consumers of `ActionResult`, `changed_fields`, zero-identity logic and `RefinedInference` before introducing the wrapper. [S4, S5]

### 5.4 Why a new head rather than `requires_grad` on centres

Current centres are transient outputs recomputed for each stereo pair. Turning on tensor gradients neither creates a persistent optimizer-owned rule nor produces corrections for unseen pairs. A shared feed-forward centre head learns across training pairs and predicts offsets once at inference.

A ten-output replacement layer could be mathematically equivalent, but would break the old seven-output state shapes and complicate migration. A separate named output on shared hidden features preserves the old attribute weights and checkpoint loader. A fully independent 11→32→32→3 MLP is a useful sensitivity (+1,539 parameters), but is not the minimal primary comparison: it nearly doubles refinement capacity and gives an independent feature extractor. No additional RGB/student features are granted only to C.

## 6. Gradient flow and optimizer ownership

| Owner | A | B | C |
| --- | --- | --- | --- |
| Stereo student and initial attribute head | eval/no-grad | eval/no-grad | eval/no-grad |
| Base tensors/context/source colours/calibration | detached | detached | detached |
| Refiner trunk and seven-output attribute head | absent | trainable | trainable |
| Centre output head | absent | absent or frozen unused zero | trainable |
| Rasterizer | evaluation only | differentiable RGB path | differentiable RGB path including means3D |
| Metric code/GT | offline only | offline only | offline only |

Use Adam with two named, disjoint parameter groups in C: attributes/trunk and centre output. Both initially have LR0.001; betas(0.9,0.999), eps1e-8, weight_decay0, matching PyTorch Adam defaults used by the current trainer. No scheduler, batch size1, global gradient norm clip1.0. Log pre/post clipping norms and group update norms. No optimizer owns base parameters or scene-specific centres. [S5]

The current loss is `0.5*(L_charbonnier_left + L_charbonnier_right)`, epsilon0.001, over unchanged deployment masks. Preserve current black/background settings from the base construction spec and record the resolved value. No KD, GT-depth loss, stereo warp, centre correspondence loss or extra photometric weights in the primary refinement stage. GT is loaded by evaluation only. [S5, S10]

Centre gradients enter shared hidden features in C; this is an intended consequence of relaxing the geometry constraint. Attribute architecture/initialization remain equal, but final attribute weights need not match. Frozen-base hashes, colours, index arrays, validity and disparity must remain invariant.

At zero-initialized output layers, the trunk can legitimately have zero gradient on the very first step. Require nonzero finite output-head gradients on a nondegenerate fixture and nonzero trunk updates after output weights have moved; do not misdiagnose this known initialization behavior as broken optimization. Zero-offset identity code must retain the derivative to W_mu, not short-circuit by returning base. [S3, S4]

## 7. Controlled variables and fairness requirements

### 7.1 Frozen control sheet

| Variable | Predeclared setting |
| --- | --- |
| Student/head checkpoint | Same clean10 weights/hash in every arm/seed; historical Stage-A checkpoint used only for construction identity, never as the final scientific weights |
| Input/calibration | Existing SCARED rectified tensors512×640, identical intrinsics/extrinsics/baseline/support and resampling provenance |
| Base field | Same grid stride2/cap from approved base spec, approximately81,920 slots at this resolution; verify exact resolved sampling rather than infer from count |
| Context | Main 11 selected channels and frozen base normal/scales/opacity; no feature additions to C |
| Attribute refiner |11→32→32→7, GELU, same resolved bounds, no colour outputs |
| Initial state | Identical shared parameter tensors and A field per seed; zero attribute and centre outputs |
| Loss/masks | Same full-frame two-view Charbonnier; no GT-driven/learned visibility mask or right-edge guard |
| Training |13,150 updates, batch1, same ordered IDs per paired seed, no early stopping |
| Optimizer/precision | Adam settings in Section6, FP32, same environment/backend/native revision |
| Output policy | One correction pass, no test-time fitting, pruning, densification or centre-dependent resampling |
| Compute scheduling | Same GPU UUID where possible; randomized/interleaved B/C run order by seed; no timing while other GPU workloads overlap |

Freeze the actual manifests and their hashes. Do not replace the immutable final test split or silently exclude difficult frames. The historical test set has already been inspected in this project: call this a **new predeclared follow-up on an existing benchmark**, not an untouched confirmatory test.

### 7.2 Parameter capacity versus geometry constraints

**Primary:** identical attribute architecture plus 99 extra C parameters. This estimates the effect of enabling an offset mapping with modest extra capacity. It does not isolate geometry freedom from active parameter count perfectly; additional geometric degrees of freedom are themselves the intervention.

An implementation can store the same 1,770-parameter supermodel in both arms, with B's99 centre parameters frozen and bypassed. Report **stored**, **trainable** and **executed** counts separately. A frozen/dummy centre head does not constitute active capacity matching.

**Recommended secondary, if approved:** an exactly parameter-count-matched fixed-centre control B_cap with the same trunk, seven original outputs, plus a zero-initialized32→3 auxiliary scale branch (99 parameters). Add `tanh(aux_scale)` to the three raw log-scale channels before the existing scale bound. The extra nonlinearity supplies a real additional attribute mapping while retaining fixed centres; total trainable count1,770. It must be separately labelled, since equal counts do not equal function classes, gradient conditioning or expressivity. Compare C to both original B and B_cap; do not replace B with B_cap or claim perfect capacity equivalence.

Do not shrink C's attribute trunk to match B, remove rotation/opacity from C, choose a tiny movement cap from test results, or train C with fewer updates. Optional independent centre-MLP/ray controls answer different questions and require their own counts/budgets.

### 7.3 Hyperparameter fairness

Default primary C: R5mm, no penalty, centre LR0.001. During development only, check R1/5/10mm and centre LR0.0003/0.001/0.003 with a bounded search plan in Section9. Fix the primary default before seeing new test outputs; label alternatives as sensitivity runs.

If C is numerically sound but performs poorly, do not call that an implementation failure. Before a strong superiority claim, allow the predeclared validation-only sensitivity and, if needed, a separately labelled equal-budget LR check for B. Retain every attempted configuration, failure and validation score. A rescue setting can be evaluated alongside the original primary after sealing; it cannot silently replace the primary.

## 8. Geometry, rendering and efficiency metrics

### 8.1 Frozen stereo geometry

Reuse `evaluate_geometry` for disparity EPE,1px accuracy, disparity-derived depth MAE/RMSE/AbsRel and geometry coverage. Principal-point correction and depth/backprojection stay canonical. Values should be exactly equal for A/B/C under the same base inference path. [S1, S8]

Any mismatch triggers an integrity investigation before interpretation. Never label centre-corrected field depth as the student's disparity-derived metric. Unchanged EPE says nothing about C's moved centres.

### 8.2 Centre-coordinate preservation and displacement

Let source slot i keep source pixel p_i; base centre mu_i and refined centre mu_i^k are in the left camera frame. Define displacement `s_i=1000*||mu_i^k-mu_i||_2` in mm.

For each frame, compute mean, median, p90,p95,p99 and max s_i over **all base-valid slots**, independent of predicted opacity or visibility. Also report signed X/Y/Z displacement summaries, fraction with `||delta|| > 0.95 R_i`, and fraction safety-limited/frozen by R_i=0. Store counts. A/B must have exactly zero displacement, not merely a small mean. Padding is excluded.

Aggregate frame means as the main dataset score. For percentiles, publish the mean of per-frame percentiles with that label; optionally publish centre-pooled percentiles separately, with counts, never interchange them.

### 8.3 Correspondence-aware centre accuracy

**Primary centre metric:** retain p_i from the frozen base sampler, gather calibrated GT 3D position X_GT(p_i), then compute:

- `centre_corr_mae_mm = mean_i 1000*||mu_i^k - X_GT(p_i)||_2`.
- `centre_corr_rmse_mm = sqrt(mean_i (1000*||mu_i^k-X_GT(p_i)||_2)^2)`.
- `centre_z_mae_mm = mean_i 1000*abs(mu_i,z^k-Z_GT(p_i))`.
- Centre Z RMSE, signed Z bias and relative Z error `abs(z-Z_GT)/Z_GT`.

The support is fixed by base-valid slots, in-range immutable source indices, sensor support and valid finite positive GT lifted with the same camera convention. Build and hash this support once per frame and use it across A/B/C/seeds. Never re-associate a moved centre to a more favourable pixel or nearest GT point for this primary metric. A centre moving outside the image remains in the comparison at its original source identity.

Measure displacement separately on slots without GT; do not fill missing GT with teacher disparity. No predicted opacity threshold can remove a centre from primary accuracy. A nonfinite moved centre is a failure, not a removable outlier. A frame with no GT-supported slots yields null plus zero count; preserve it for RGB/displacement results. Audit index bounds with per-image HW, not a flattened multi-batch shortcut. [S2, S8]

This metric tests **retention/correction of source-anchored tissue correspondence**. A large tangential relocation may improve a visible surface while breaking that anchor. Therefore add existing symmetric unsquared Chamfer as a secondary geometric diagnostic, with its cap1024/chunk256 and sampling definition disclosed. Chamfer cannot replace source-correspondence error because it permits nearest-neighbour reassignment. [S8]

Occlusions: the primary correspondence metric concerns the original left-view source, so right-view occlusion is not a reason to discard it. Optionally stratify by a frozen base-derived left/right visibility estimate, normal fallback and discontinuity flags; these subsets are exploratory and never replace all-supported-slot results. No GT visibility enters inference or training.

### 8.4 Rendered geometry: current blocker and proposed diagnostic

**Confirmed blocker:** production rasterization returns `(color,radii)` and wraps `RenderOutput(image,visibility)` without depth. `RenderOutput.depth` is only optional. `evaluate_geometry` explicitly returns null depth metrics when none is exported; current visibility is per-Gaussian radius, not per-pixel opacity. Thus existing depth-evaluation support alone does not establish a usable depth image. [S6, S8]

**Preferred minimal diagnostic, proposed:** an auxiliary evaluation-only RGB moment render with the identical centres/covariance/opacities/cameras and **black background**. For each centre, compute optical-axis Z_i in the evaluated camera in metres and set diagnostic RGB to `(Z_i/z_ref, (Z_i/z_ref)^2, 1)`, with z_ref equal to a fixed recorded metric far plane. All retained depths must be inside the supported positive near/far range. Never min–max normalize per frame or clip unsupported depths to make them fit.

If the pinned backend's compositing weights are `w_i=T_i alpha_i`, the diagnostic channels provide:

`M1=sum_i w_i Z_i/z_ref`, `M2=sum_i w_i (Z_i/z_ref)^2`, `A=sum_i w_i`.

For pixels with A≥tau, define `D=z_ref*M1/A` and `V=max(0,z_ref^2*M2/A-D^2)`. These are **alpha-composited centre-depth expectation and depth dispersion**, not the first Gaussian surface intersection or metric mesh surface. Record small negative variance roundoff before clamping; substantial negatives fail validation. One auxiliary render produces all three moments. Never mutate source colours in the actual field/checkpoint; construct a detached diagnostic copy only. Exclude this pass from production latency.

This reuse is conditional: inspect the pinned native source to verify that RGB values do not change sorting, alpha accumulation, culling or termination, and validate the result numerically on fixtures. `production.py:388–400` confirms precomputed-colour input, but does not itself prove that the native accumulation has these semantics. If that source or parity validation is unavailable, **rendered depth stays N/A**, with an explicit reason. Do not relabel stereo depth as rendered depth. An actual native depth/alpha export would be a larger separately approved fallback, not an unplanned rasterizer rewrite.

Proposed primary tau0.5; report tau0.1/0.9 sensitivities without selecting the best test threshold. Compare D against left-view GT depth on unchanged GT/sensor support and each branch's coverage mask. Report:

- Per-frame composited centre-depth MAE/RMSE/AbsRel and valid-GT-pixel coverage.
- The same errors on the common A/B/C high-opacity intersection, with intersection coverage and the excluded GT fraction.
- Coverage-change rates and high-opacity depth dispersion; accuracy gains accompanied by coverage loss cannot be called an unqualified improvement.
- B-minus-A composited-depth differences even though centre displacement is zero.

Do not silently gate the primary depth error by low dispersion: that would discard hard mixtures. Any unimodal/low-variance subset is secondary with full counts. Right-view depth may be rendered diagnostically, but no metric right-GT error is reported unless an independently verified right-depth reference exists. Warping predicted or left GT depth does not create an independent dense right GT.

Scale/rotation change projected covariance/support and opacity changes alpha/transmittance, so w_i and D may change with fixed centres. This is the exact reason centre preservation and apparent surface preservation must be separated.

**Fallback if moment depth fails:** report correspondence centre geometry, secondary Chamfer and projected-centre/occlusion illustrations only. Label rendered-depth cells N/A and retain the limitation; do not claim surface preservation.

### 8.5 RGB quality and aggregation

Use the current main full-frame protocol in `evaluation/multi_model.py:119–128`: PSNR with data_range1, Gaussian11×11 SSIM sigma1.5/zero padding5, and pinned offline LPIPS-AlexNet v0.1 weights. Do not use `rendering.py::ssim`, the global development metric. No new photometric crop or opacity mask for primary RGB. [S7]

For each frame and view compute a scalar; dataset mean is the arithmetic average of available frame scalars. Pooled main score averages contributing frames across datasets, not an equal mean of dataset means and not a PSNR derived from pooled MSE. Depth RMSE is computed within each frame and then averaged. Save per-metric availability/counts. The generic `evaluation/aggregation.py:37–55` is valid-count-weighted and **must not be used blindly** to recreate the main unweighted frame protocol. [S7, A3]

For new centre metrics also use per-frame summaries then arithmetic mean, with centre counts reported separately. Keep unpaired absolute means and strict common-frame paired differences distinct. A nonfinite model prediction is a model failure, not a reason to shrink a branch's test roster silently. Exact-image PSNR infinity must be explicitly handled/logged, not silently excluded as finite-only.

### 8.6 Efficiency

Report stored/trainable/executed parameter counts, module precision, one-pass refiner-only latency (frozen base field resident), field-correction-plus-two-renders latency, full stereo+field+correction+two-renders latency, and peak allocated/reserved inference memory. List whether source-index export and diagnostic copies are enabled; they must be disabled or identically accounted for during production timing.

Matched RTX5070Ti GPU UUID,512×640, batch1,FP32, same CUDA/Torch/rasterizer/source snapshot. Reuse the saved timing frame roster from dataset7,20 warmups and200 measured iterations per invocation, three invocations with alternating A/B/C order. Use CUDA events with synchronization at defined boundaries, record graph-capture and compilation time separately, and measure resident input scope without I/O/decode/H2D. No concurrent training/evaluation/timing on the measured device. Report medians and sample distributions, not FLOPs as latency. [A4, S11]

Preferred path: all A/B/C use the same parallel GroupNorm/static grid/CUDA graph settings after parity gates. If C cannot capture yet, primary measured comparison uses common eager execution for **all** arms; provide historical deployed21.2ms only as separately scoped context. Do not compare B graph latency against C eager latency as a scientific efficiency effect.

## 9. Training and validation protocol with decision points

### Phase 1 — Implementation feasibility review (later, no experiment)

1. Archive hashes of inspected/approved source, environment, clean10/base-construction artifacts, main refiner spec, LPIPS pins and train/validation/test manifests. Verify all base model keys are replaced by clean10, not a partial import. [S9, A1]
2. Add a new opt-in wrapper/head and offset composer; preserve old `FieldRefiner` public API and seven-output checkpoints. Add a dedicated training/inference adapter using the same correction implementation, no private evaluation duplicate.
3. Extend evaluation output with immutable source-index/valid-slot metadata, using the exact deployed static-grid sampling table. Current `StudentPredictor` returns source_index=None; recomputing an eager base independently can change the field and is not an acceptable silent workaround. [S2, S9]
4. Add correspondence metrics and optional moment diagnostic under new names; do not change historical `stereo_depth_*` definitions or pretend cloud Chamfer is correspondence MAE.
5. Identify graph-capture constraints and legacy zero-identity/merge consumers. Test common eager path first; graph support is an opt-in layer after semantic parity.

**Gate:** old checkpoint load/forward contracts remain intact; all new paths are opt-in; no source/calibration/formula semantics change outside this ablation. Otherwise revise the implementation plan before GPU experiments.

### Phase 2 — Small development pilot

No final-test images/results are used to choose architecture, bounds or LR. Use fixed synthetic fixtures and a sealed32-frame training subset (eight entries from each train dataset, selected deterministically from the sorted manifest); evaluate all56 dataset7 validation frames. Log their IDs/hashes later when approved, without copying medical data into Git.

**Implementation checks, objective pass/fail:**

- At initialization A=B=C valid-slot centres/attributes/colours and source indices exactly match; zero outputs render identically under one execution path.
- Base weight/buffer hashes, original disparity, source colours and sampling indices remain unchanged after updates; frozen parameters have no gradients/Adam state.
- At a nondegenerate synthetic fixture, both attribute and centre output gradients are finite and nonzero; finite differences of means3D versus image/loss agree within a predeclared numerical tolerance (relative1e-2 away from sorting boundaries). Validate both cameras, offset units and camera transforms. Missing native centre gradients is a blocker, not a negative research result.
- All valid centres/quaternions/scales/opacities finite; scale bounds and unit-norm quaternion tolerance1e-5 hold; covariance symmetric positive definite within the scale-floor tolerance. Identity derivative tests cover zero attributes with nonzero centre offsets and the reverse.
- Correspondence fixtures: identity error0; known translated centre has expected mm error; pixel-index permutation is detected; no-opacity pruning changes metric support; an all-missing-GT frame returns null plus count0.
- Moment-depth feasibility: one/two-depth, overlap/transparent-background fixtures match a reference compositing calculation within max(0.01mm,relative1e-4); test native unit-scale invariance, left/right extrinsics, alpha thresholds and normalized moment range. Colour-independent weights must be verified before accepting the diagnostic.
- Deployed/eager parity, if attempted: valid centres max abs difference≤1e-6m, RGB max difference≤2e-4 and per-frame PSNR difference≤0.02dB on the development fixture roster. These are proposed engineering tolerances, not scientific accuracy thresholds. Failures lead to one common validated path, not wider tolerances chosen from test outputs.

**Optimization pilot:** six matched short runs of512 updates: B_default and C with (Rmm,centre LR)=(1,.001),(5,.001),(10,.001),(5,.0003),(5,.003). Fixed seed20261006; identical attribute initialization and ordered512 samples. Trainable head/trunk updates and centre norms/saturation are logged every25 updates. All configurations retain the same attribute LR/clip.

**Pilot decision rules:**

- Nonfinite loss/tensors, base-state drift, failed unit/correspondence tests, or missing centre gradients: implementation gate fails; preserve outputs and halt the affected lane.
- R5 primary saturation>20% of movable slots at the last100 updates or repeated norm clipping is a warning that bounds/conditioning affect interpretation. It does not invalidate a stable scientific result. Require reporting the R1/R10 sensitivity before a strong claim about C.
- No rendering improvement, near-zero learned offsets, or worsened geometry are **not** pilot engineering failures. Report them. They do not authorize dropping C.
- If LR sensitivity reveals a stable C setting with lower validation photometric loss than default, nominate it as `C_dev` using the lowest full-validation two-view Charbonnier, ties within1e-6 resolved by LR nearest.001 then smaller R. The original C5 remains primary; shortlist at most one C_dev. Before test, C_dev requires its own full-budget three seeds or is labelled single-seed exploratory. GT centre accuracy is a diagnostic, not a hidden selection constraint favouring zero motion.
- No test-dependent threshold, centre bound, mask or stopping choice. Seal decisions and all pilot scores before Phase3. An empirical failure in this short pilot is not evidence of final comparative inferiority.

### Phase 3 — Full matched comparison

Use refinement seeds20261006,20261007,20261008. For each seed, create fresh attribute weights once; clone them into B/C. Seed controls the new MLP initialization and a single permutation of all13,150 sorted training IDs; B/C share exactly that permutation. Across seeds allow distinct permutations, record both initialization and sample-order seeds. This estimates refinement variation; it does not retrain clean10.

Each run: one full epoch13,150 updates, Adam settings Section6, final checkpoint at update13,150, no selection of best validation checkpoint. Validation after training can check integrity and report quality; it cannot change the predefined checkpoint. Save full Adam/RNG/cursor/source/config/order identity at update0 and every1,000 updates plus final; resume only with identical identity and full state. An interruption is disclosed; a divergent run is not silently restarted from a healthier checkpoint. [S5]

Freeze all B/C checkpoint/spec hashes and metric/aggregation protocols before generating their first new final-test row. Evaluate A once and each of the six final models on all5,907 test frames; geometry metrics keep their actual GT-supported frame counts. Retain per-frame results, mask hashes, centre counts and failure records. Do not mix every20th/296-frame diagnostics with full-test means.

Pair scheduling example to predeclare: seed1 B→C,seed2 C→B,seed3 B→C, on the same device; initial order is randomized/recorded before running. If two GPUs are used, cross arm/device assignments across seeds and record UUIDs; do not put every B on one GPU and every C on another. Scientific timing runs separately on an idle fixed UUID.

**Historical B handling:** evaluate unchanged `B_historical` under the same metric path only if needed, with its exact seed/order/budget provenance. It supplements rather than replaces newly matched B. If seed1 should reproduce it but does not, inspect source snapshots/normalization/path/bounds/permutation before interpreting the difference. No new student training is authorized by this design.

### Phase 4 — Analysis and decision reporting

For each seed: common-frame B−A,C−A,C−B deltas; per-dataset means, per-keyframe means, improvement fractions and paired scatter of centre error change versus PSNR change. Keep all seeds, including failures. Report seed mean/sample SD of dataset-level effects; three seeds are limited evidence, not a reliable estimate of rare-instability probability.

Use5,000 paired keyframe-cluster bootstrap resamples, analysis seed20261112, retaining all paired frame values within sampled clusters and original frame weighting. Pooling across datasets should sample keyframes within each dataset stratum. Publish per-dataset intervals (four keyframes each) and pooled sensitivity (eight total), explicitly acknowledging the small cluster count. Existing `comparison_stats.py` can supply per-seed functions but needs a stratified/hierarchical wrapper for the pooled analysis. [S12]

Also show frame-bootstrap intervals as sensitivity, not the sole evidence. For joint seed/data uncertainty, resample paired refinement seeds and dataset-stratified keyframe clusters, retaining branch pairing; do not treat seed×frame rows as independent. Publish per-seed results rather than relying on a narrow aggregate CI. No multiple-claim significance fishing: the co-primary centre-error/PSNR vector is predeclared, secondary findings are exploratory. No equivalence claim without a separately justified margin and power analysis.

## 10. Expected computational requirements and dependencies

### 10.1 Planning estimate, not measured new runtime

Historical full refiner training:2,909.964 compute-loop seconds /13,150 updates≈0.221s/update; end-to-end3,183.500s≈0.884GPU-hours per run. [A2]

| Work | Count / assumption | Proposed allowance |
| --- | --- | --- |
| Feasibility fixtures + six512-update pilots |3,072 updates plus checks; historical speed is only a starting point |0.5–1.5GPU-hours |
| Main B training |3×13,150 updates |2.7–3.5GPU-hours |
| Main C training |3×13,150; assume1.0–1.5× B wall time due centre gradients/culling effects |2.7–4.5GPU-hours |
| Main evaluation |A+six trained models=41,349 frame predictions; two-view LPIPS and geometry/moment passes add cost |1.5–4GPU-hours |
| Matched timing and controlled retries |Idle-device campaign, no interrupted timing reuse |0.5–1GPU-hour |

Reserve10–16GPU-hours on one5070Ti-class GPU; a smaller7–10hour run is plausible but not promised. Scheduling/I/O/implementation work can exceed GPU time. Six primary training runs are78,900 updates. Existing21.2ms full inference only provides an approximate compute lower bound for predictions; it does not include LPIPS, GT loading, centre metrics or auxiliary moment renders and must not be used as a measured evaluation throughput.

Optional three-seed B_cap adds approximately2.7–3.5training GPU-hours plus evaluation. A full-budget C_dev three-seed confirmation adds roughly3–5training GPU-hours. One additional independent centre-MLP or parameterization family requires a separate budget. No student retraining is included.

Memory target: fit on one16GiB-class device, batch1; actual peak must be measured after implementation. Historical refiner migration used a0.35 per-process training allocation cap, which is not a measurement of needed memory. C backward can change rasterization storage and footprint; do not promise equal memory. Storage planning: allow20–50GB for snapshots/checkpoints/scalars and fixed qualitative arrays; avoid saving all full-resolution float32 renders (hundreds of GB across runs). Final sizing is a Phase1 inventory, not permission to delete old artifacts.

### 10.2 Dependencies

- Existing configured SCARED data outside Git, immutable train_1236/val7/test manifests and role/calibration provenance.
- The clean10 student, base construction/config artifact and historical full refiner for reference only.
- Pinned native RGB rasterizer with differentiable means3D; compatible Torch/CUDA/environment; offline LPIPS weights/hashes.
- New opt-in centre head/composer, training/inference adapter, source-correspondence export, metric implementation and checkpoint identity support.
- Optional moment diagnostic only after native-source review and fixture gate. No new CUDA extension is presumed necessary.
- Idle GPU scheduling and an external new output root. Each attempt must create a new directory; no overwrite of successful outputs.

No installation, data preparation, hash rewriting or runtime availability certification was performed during this design task.

## 11. Proposed CVPR table and qualitative figure

### Effect of Geometry–Attribute Decoupling

Proposed subsection text before results exist:

> We isolate the refinement-stage geometry constraint by freezing the same distilled stereo student and initial Gaussian head. Starting from identical fields, we train attribute-only and centre–attribute refiners with matched initialization, samples, objective and update budget. The latter additionally predicts bounded metric XYZ offsets. We evaluate rendering and source-correspondence centre geometry separately; unchanged stereo disparity is an integrity check rather than evidence of unchanged Gaussian geometry.

**Main-paper table template — no new numerical outcomes are supplied:**

| Refinement | Centre move mean/p95 (mm) | Corr. centre3D MAE (mm)↓ | Composited-depth MAE (mm)↓ / coverage↑ | PSNR_R↑ | SSIM_R↑ | LPIPS_R↓ | Trainable params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A: Initial field |0 /0 by construction | TBD | TBD, or N/A if diagnostic fails | TBD | TBD | TBD |0 |
| B: GaD fixed centres |0 /0 by construction | Same as A on fixed support | TBD, or N/A | TBD | TBD | TBD |1,671 |
| C: Centres + attributes | TBD | TBD | TBD, or N/A | TBD | TBD | TBD |1,770 proposed |

Use mean±SD across three refinement seeds for B/C quality; A is one deterministic base. Do not present A as three independent runs. Put C−B paired effects/cluster intervals and per-dataset results in an adjacent small panel or Supplementary. Geometry may contribute fewer frames than RGB: state both counts. The depth-column heading must explicitly say **composited centre depth** if using moments. If the table is too wide, keep the core centre-error/PSNR/parameters table in the main paper and move depth coverage, left-view metrics, all percentiles and sensitivity runs to Supplementary.

This comparison belongs in the **main paper** if GaD is a central methodological contribution. Bounds/LR/capacity controls, all seeds, failures and full metric definitions belong in Supplementary. If direct centre metrics are blocked, do not fill the main table with stereo EPE and claim it tests centre movement.

**One proposed qualitative panel:** same stereo frame, crop, colour scale and camera for A/B/C; show observed right RGB, rendered right RGB, absolute RGB error, source-anchored centre displacement map (mm), and signed centre-to-GT Z error. Add a fixed camera-space close-up of centre arrows/GT points. If validated, include composited-depth difference B−A and C−A with opacity/coverage beside it. The B panel should visibly have zero centre motion even if depth compositing changes; C can show geometry improvements or drift, whichever occurs.

Select the display frame without inspecting C's final results: predeclare a valid representative frame ID from the test roster or choose the median A baseline right-PSNR frame by a sealed rule, and disclose prior benchmark exposure. Use the same frame across all seeds or a predeclared seed1 display; no cherry-picked failure/success. No figure generation is part of this task.

## 12. Interpretation of possible outcomes

| Observed controlled outcome | Responsible interpretation / paper consequence |
| --- | --- |
| C improves rendering, worsens correspondence centre error | Supports a geometry–rendering trade-off for this model/bound/budget. B provides exact coordinate control; include depth/coverage and both views before a broader claim. |
| C improves both centre geometry and rendering | Weakens any comparative claim that fixed centres are preferable. Narrow GaD to a guaranteed-preservation option; report benefits of learned geometry correction and discuss whether the methodological contribution needs repositioning. |
| C worse in both | First audit gradient flow, bounds, saturation, LR, initialization and capacity controls. If fair and robust across seeds/sensitivities, claim an empirical advantage for B in the tested regime, not an inherent law. |
| Similar RGB, C centres worse | B offers coordinate retention without sacrificing observed RGB in this study. “Similar” is descriptive unless equivalence margins/power were approved. |
| Similar RGB, C centres better | Preservation may be unnecessarily restrictive; report improved accuracy rather than treating movement as intrinsically harmful. |
| Similar centre accuracy and RGB | No detected comparative methodological advantage at this budget. B may have simpler control/less overhead only if counts/runtime support it; no equivalence claim. |
| B centres fixed but composited depth changes | Centre preservation holds; rendered-surface preservation does not follow. Rewrite any surface-preservation wording precisely. |
| C barely moves / saturates heavily | Could reflect optimum, feature insufficiency, optimization or bound limitation. Report output gradients, displacement distribution and predeclared sensitivities; do not treat this alone as proof that movable centres are harmful/useless. |
| C unstable, B stable | Record failure frequencies and diagnostics. Preserve all failed attempts; distinguish numerical implementation defect from a reproducible property of the approved recipe. |
| C improves a cloud metric but worsens correspondence error | State geometric reassignment versus source-anchor accuracy explicitly; neither metric alone settles tissue correspondence or surface accuracy. |

The comparison does not prove causality of historical stereo-joint failures, universal superiority over joint optimization, clinical measurement safety, or preservation of an entire continuous rendered surface.

## 13. Remaining implementation risks

1. **Correspondence missing in deployed outputs.** Rebuilding geometry through a different eager normalization path can invalidate pairing. Export exact static-grid metadata or use a single common validated path. [S9]
2. **Identity fast path can suppress centre gradients.** Keep attribute identity handling separate from centre addition; test zero attribute/nonzero centre cases. [S4]
3. **No pixel alpha/depth export.** Auxiliary colour moments depend on unverified native accumulation; keep N/A fallback until gate passes. [S6]
4. **Main bound double-scaling.** Use resolved config once and archive it. [S9, A2]
5. **Sampling/opacity selection bias.** Fix source slots/support from the base; do not remove moved or dimmed Gaussians from centre evaluation.
6. **Colour/correspondence assumption.** Moving a Gaussian while keeping its source colour assumes that colour can be transported locally. This is intentionally controlled, not evidence of true material-point identity.
7. **Restricted context.** Main11-channel input excludes absolute XY and neighbours; C may be limited for lateral correction. Do not compensate by granting hidden image features only to C; independent/ray variants are sensitivities.
8. **Nonstationary projection/culling/sorting.** Native derivatives are piecewise, centre shifts can change visibility and numerical gradients. Test away from sort boundaries; report final footprint/coverage rather than forcing identical visible splats.
9. **Ground-truth semantics.** Reuse main GT/masks; flag doubtful sentinels/calibration rather than silently filtering. GT-supported sampled centres can have a different count/distribution from full-image stereo pixels.
10. **Limited independent replication.** Three refinement seeds and eight keyframes do not establish broad generalization; student artifact is shared and test results were historically seen.
11. **Budget/regularization asymmetry.** A fixed5mm bound can constrain C; sensitivity runs and separately scoped rescue variants are necessary before strong negative conclusions.
12. **Compatibility/provenance.** Working tree differs from frozen snapshots; do not claim reproduction using HEAD only or overwrite existing checkpoint schemas.

## 14. Concrete implementation plan for later approval

All paths below are **proposed changes only**. No files other than this design were created or modified in this task.

| Step | Minimal proposed source work | Acceptance evidence before progressing |
| --- | --- | --- |
|1. Freeze inputs | New external experiment spec/protocol, manifests/hashes, source snapshot and environment identity; portable paths | Clean10/config/LPIPS/data/split/order identities complete; existing outputs untouched |
|2. Implement centre wrapper | Add an opt-in class in `gaussians/refinement.py` or a focused sibling; preserve old FieldRefiner signatures/state shapes. One canonical metre-offset function with explicit `[B,N,3]` contracts | Counts1671/1770; paired shared weights; old checkpoint round-trip; zero/bounded-offset CPU tests |
|3. Compose once | Add opt-in centre-aware composition entry in `actions/refinement.py`, calling existing attribute composer, then adding offsets | Same implementation used for training/inference/evaluation; colours/index/validity invariants; no extra render for training diagnostics by default |
|4. Train adapter | Extend `training/field_refinement.py` through an explicit optional correction interface or focused sibling; base remains frozen, named optimizer groups and complete resume identity | CPU analytic gradient/ownership/resume tests; original trainer default unchanged |
|5. Evaluation | New correspondence metric owner under `evaluation/`; opt-in source-index export in `actions/refinement.py`/`evaluation/ours.py` without altering old headline outputs | Fixed-support synthetic metric tests; exact identity centre error; new metric names/counts |
|6. Optional depth diagnostic | Evaluation-only RGB moment helper with same native rasterization; no production colour mutation | Native-source review and fixture gate; otherwise explicit N/A |
|7. Deployment/profiling | Opt-in C-compatible capture and three-arm common timing harness; thin new `scripts/` entry point only | Eager/deployed parity, read-only evaluation hashes, exact timing scope |
|8. Pilot | Run only after explicit execution approval, with Phase2 sealed roster/settings | All integrity gates; every pilot result and failure retained; sensitivity decisions sealed |
|9. Matched study | Three B/C seed pairs, fixed update budget; freeze all final checkpoints before test | Complete counts, per-frame paired records, hashes and scientific failure reporting |
|10. Analysis/export | Generate immutable scalar tables/plots externally; review results before any manuscript edit | All predeclared outcomes, limitations and comparison scopes shown; no invented metrics |

Search consumers before touching established interfaces. Keep formulas in canonical owners, scripts thin, paths configurable, public APIs typed and shapes documented. No architecture/routing packages, new scientific stage or baseline modifications are required. A full implementation review must include tests for unexpected mutation of source colours, masks, base tensors and scientific model state during evaluation.

**Decisions to approve before implementation/execution:**

1. Primary shared-trunk XYZ head,5mm cap, unregularized RGB objective and the three-seed B/C scope.
2. Whether to include B_cap (recommended if budget permits) and full-budget C_dev/sensitivity confirmations.
3. Proposed engineering tolerances, moment-depth convention and opacity thresholds; acceptance of N/A if the diagnostic cannot be validated.
4. Reserved GPU time/output storage and test-evaluation scope; whether an idle deployment path or common eager path is used.
5. Source snapshot/compatibility approach and one immutable protocol amendment before execution. Approval of this document alone does not authorize GPU jobs or code changes.

**Stop condition for this task:** deliver this Markdown design, then wait for explicit authorization. No experiment is running on behalf of this task.

## 15. Evidence ledger

### Source references

Line numbers identify the inspected working tree. Ranges are explanatory references; links open a starting line.

| ID | Exact repository source | Supporting statement |
| --- | --- | --- |
| S1 | [geometry/rectified.py](src/reliable_endo_gs/geometry/rectified.py#L22), lines22–53 | Principal-point correction once; metric depth/backprojection and gradients |
| S2 | [sampling.py](src/reliable_endo_gs/gaussians/sampling.py#L66),66–83,217–295; [priors.py](src/reliable_endo_gs/gaussians/priors.py#L374),374–431,458–467,500–515 | Source correspondence, static-grid slots, sampled colours |
| S3 | [refinement.py](src/reliable_endo_gs/gaussians/refinement.py#L22),22–46,92–165; [residual_head.py](src/reliable_endo_gs/gaussians/residual_head.py#L140),140–180 | Main channel selection, MLP outputs/zero initialization, frozen context construction |
| S4 | [actions/refinement.py](src/reliable_endo_gs/actions/refinement.py#L64),64–103,141–170; [repair.py](src/reliable_endo_gs/actions/repair.py#L129),129–174; [composition.py](src/reliable_endo_gs/gaussians/composition.py#L284),284–306 | Attribute composition, centres retained, exact-zero derivative semantics |
| S5 | [field_refinement.py](src/reliable_endo_gs/training/field_refinement.py#L42),42–58,105–140,197–259; [interventions.py](src/reliable_endo_gs/training/interventions.py#L31),31–63 | Frozen model, Adam ownership, loss/backward, state invariance/resume |
| S6 | [production.py](src/reliable_endo_gs/rendering/production.py#L360),360–402,489–539; [rendering contract](src/reliable_endo_gs/contracts/rendering.py#L19),19–33 | Precomputed colours, metre conversion, `(color,radii)` only, optional depth |
| S7 | [multi_model.py](src/reliable_endo_gs/evaluation/multi_model.py#L81),81–129; [rendering metrics](src/reliable_endo_gs/evaluation/rendering.py#L112),40–66,112–149; [perceptual.py](src/reliable_endo_gs/evaluation/perceptual.py#L69),69–100; [aggregation.py](src/reliable_endo_gs/evaluation/aggregation.py#L37),37–55 | Main full-image RGB metrics; pinned LPIPS; generic pixel-count weighting differs |
| S8 | [benchmark_geometry.py](src/reliable_endo_gs/evaluation/benchmark_geometry.py#L110),110–176,193–247 and25–107 | Stereo metrics, nearest-neighbour centre Chamfer, explicit unavailable rendered-depth records |
| S9 | [ours.py](src/reliable_endo_gs/evaluation/ours.py#L103),103–154,157–184; [student.py](src/reliable_endo_gs/reconstruction/student.py#L297),297–305,327–350 | Strict clean weights import, bound loading, deployed predictor drops indices, initial student gradient path |
| S10 | [cross_view.py](src/reliable_endo_gs/rendering/cross_view.py#L258),258–268,313–350 | Charbonnier epsilon and reduction |
| S11 | [comparison_runner.py](src/reliable_endo_gs/evaluation/comparison_runner.py#L141),141–179,220–250; [idle_latency.py](src/reliable_endo_gs/profiling/idle_latency.py#L142),142–176 | Dataset/evaluator/provenance pattern; current timing helper is variant-specific and needs a new A/B/C roster |
| S12 | [comparison_stats.py](src/reliable_endo_gs/evaluation/comparison_stats.py#L22),22–75 | Paired frame/keyframe-cluster bootstrap, mean frame weighting |

### Saved artifact references (external, read only)

**A1 — clean student:** `/mnt/d/3DGS_runs/reliableendo-gs/clean10-20261103/kd/attempt-00/result.json` (`cursor`, `model_initialization`, `schedule_complete`); sibling `FINAL.pt` identity recorded above. No checkpoint was deserialized or rehashed in this design task.

**A2 — actual main refiner:** root `/mnt/d/3DGS_runs/reliableendo-gs/clean10-migration-20261106T143532+0700/`:

- `refiner/spec.json`, `refiner/summary.json`, `refiner/train.jsonl`, `refiner/refiner.pt`.
- `plan/PROTOCOL.txt:8–26`: architecture, sample order, budget, validation/test scope, precision/environment constraints.
- `plan/settings.json`: snapshot/source script/environment identities.
- `scripts/train_refiner.py:48–79`: strict clean weight import, sampling order and seed, resolved bounds, trainer call. This historical operational launcher contains machine-specific paths; a new implementation must use configuration/environment paths, not copy them into core code.

**A3 — main stored scores:** `/mnt/d/3DGS_runs/reliableendo-gs/clean1-completion-20261107T000149+0700/results/means.json`, keys `ours_base/ours_refined`, per-metric `dataset_8/dataset_9/pooled` scalar/count pairs. Same-root `comparison_tables.md` and repository `docs/PAPER_CODE_RESULTS_CHECK_2026-11-11.md:25–34` document current frame-mean aggregation and centre/surface limitations.

**A4 — matched historical timing:** `/mnt/d/3DGS_runs/reliableendo-gs/matched-optimized-latency-20261107T004544+0700/results/result.json`, keys `scope`,`precision`,`timing_frame_ids`,`warmup`,`repetitions`,`ours_ms`,`ours_median_ms`; same-root `raw/ours_run1.json`…`ours_run3.json` identify GPU/precision/raw measurements.

**A5 — original native joint pilots:** under `/home/bkai/.local/share/reliableendo-gs/runs/`:

- `joint-larger-gpu1-4vgk9wi_/parent128-val/result.json` and `joint1152-val/result.json`: `summary.base/selector.metrics`; `completion_report.md:3–37` gives update budget/limits and checkpoint SHA.
- `joint-versus-frozen-gpu0-11fuurbl/completion-report.json`: `evaluation_scope`,`results.joint459/frozen459`, selected and base metrics; `joint459-val/result.json`,`frozen459-val/result.json` retain primary native results.
- `joint-geometry-remediation-qayuedy8/ancestry-overlap-audit.json`; repository `docs/joint_geometry_selector_audit_2026-10-13.md:18–36,115–145` reports objective and ancestry issues. No later comparative result in this design is inferred from these pilots.

**A6 — supervision controls:** `/home/bkai/.local/share/reliableendo-gs/reports/insight-ablations-20261101/results/REPORT_C.txt:1–14`, `REPORT_D.txt:1–11`, `REPORT_G.txt`; raw `results/job3/reconly_ep3.json` (`rows[].pred_disp_mean/epe/depth_mae_mm`) and KD-only/seed JSONs. Predeclared `plan/PROTOCOL_C.txt:6–24` explains paired initialization and geometry-only evaluator. They do not report rendering quality for the collapsed model.

**A7 — design workflow:** [experimental-design skill](.agents/skills/experimental-design/SKILL.md), applied for paired replication, blocking of GPU/run-order nuisance effects and separation of frame clustering from independent training seeds. No skill scripts or dependency installations were run.

### Design-task completion record

Confirmed by inspection: existing implementation, recorded artifacts, loss/optimizer boundaries and metric blockers. Proposed only: centre head, bounds/regularization, new metric export, depth diagnostic, pilots, training, timing and paper presentation. Not tested in this task: native gradient correctness, memory fit, depth moments, parity or training outcomes. Those are explicit feasibility gates for a later authorized session.
