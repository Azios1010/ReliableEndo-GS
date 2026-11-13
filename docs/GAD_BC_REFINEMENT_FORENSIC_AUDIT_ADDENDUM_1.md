# GaD B/C Forensic Audit — Addendum 1: half-pixel renderer mismatch

Date: 2026-11-13. The original audit `docs/GAD_BC_REFINEMENT_FORENSIC_AUDIT.md` is unchanged and remains the base document. This addendum supersedes its findings **F-31** (pixel-centre convention, was UNVERIFIED), **E4** (§9, was UNVERIFIED) and **C-03** (§8, was UNVERIFIED/REQUIRES EXPERIMENT), and corrects three statements made in the intermediate probe report (§6 below).

Artifacts (all outside tracked source, nothing existing was modified):
- `outputs/gad_pixel_convention_probe_20261113T215032/` — Phase 1 single-Gaussian probe (`REPORT.md`, rows, scripts).
- `outputs/gad_halfpixel_validation_A_20261113T220548/` — arm A original vs corrected, validation dataset_7 only (`gate_and_integrity.json`, `rows_*.jsonl`, `ANALYSIS.txt`, `analysis.json`, `sweep.json`, `region_stats.json`, `extended_sweep/`).

Labels: **OBSERVED** = measured in the artifacts above. **HYPOTHESIS** = not tested. **UNVERIFIED** / **REQUIRES EXPERIMENT** as in the original audit.

---

## 1. Confirmed mismatch, root cause and verified correction (supersedes F-31, E4, C-03)

**OBSERVED (CONFIRMED).** The production renderer places a Gaussian at `(u_K − 0.5, v_K − 0.5)` in the repository's integer-pixel-centre convention: Δu = −0.5007 ± 0.0032, Δv = −0.5005 ± 0.0020 px over 336 valid single-Gaussian renders, within 0.019 px of −0.5 in every case; constant across view (left/right), camera (real dataset_7 and two synthetic), depth (0.03/0.06/0.12 m), footprint (σ 2.0/3.5 px), image position and integer/fractional sources. Repo `backproject_depth` equals `D·K⁻¹[u,v,1]ᵀ` to 5.4e-9 m, so the discrepancy is on the rasterization side.

**Root cause (source lines).**

| Component | Convention | Location |
|---|---|---|
| Data, resize, backprojection, metrics | pixel `i` centred at `u = i` | `src/reliable_endo_gs/geometry/conventions.py:21`; `geometry/backprojection.py:88`; `geometry/rectified.py:56-76`; `…/dataset_7/keyframe_1/data/a00_protocol.json "pixel_center": "integer"` |
| Native camera builder | frustum from `cx`, `cy` without the half-pixel term, giving `x_ndc = (2/W)(fx·X/Z + cx) − 1` | `src/reliable_endo_gs/rendering/production.py:141-144,150-151` (function at `:82`; single call site `:719`) |
| Native NDC→pixel | `pix = ((ndc+1)·S − 1)/2` (= `u_K − 0.5`) | `/mnt/d/an_3dgs/Endo-E2E-GS/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h:41-44`, `forward.cu:233` |
| Native pixel loop | pixel `j` evaluated at coordinate `j` (`d = xy − pixf`) | `forward.cu:282,333` |

The pinned native revision is `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d`; `production.py` has been unchanged since the `a00_camera_v2` renderer entered the repository (`426d2d9`, 2026-10-05) and is byte-identical in the clean10 source snapshot.

**Verified correction (not applied).** Adding +0.5 to `cx` and `cy` of `K_L`/`K_R` only when the native frustum is built gives Δu = −0.0005, Δv = −0.0004 px (max 0.024) on the same fixture. Stereo disparity, backprojection and Gaussian centres are not involved. The offset is equal in both views (Δ_L − Δ_R = +0.0017 px), so disparity-derived geometry is unaffected.

---

## 2. Controlled evaluation: arm A, original vs corrected renderer

Design: one frozen forward pass per frame (existing snapshot code, GPU `8a3f4d31…`, same environment variables, canonical data root and sealed 56 validation ids as the existing evaluation). The same Gaussian field is rendered (a) by the model's own renderer (A-original) and (b) by a renderer subclass that adds +0.5 to `cx, cy` only inside `_native_camera`, never injected into the model (A-corrected). Metrics use the existing `decoupling_frame_metrics` (same PSNR/SSIM/LPIPS implementations). Dataset_7 only; no test data.

**Gates, all OBSERVED (`gate_and_integrity.json`):**

| Gate | Result |
|---|---|
| A-original vs the existing `evaluation/validation/A/rows.jsonl` | 1,736/1,736 metric values bitwise equal (max abs diff 0.0); 896/896 diagnostic fields equal |
| Re-render through the model's renderer with the stripped-covariance request equals the model's own base renders | 56/56 frames, both views (`torch.equal`) |
| Second forward pass bitwise identical (first 3 frames) | yes (disparity, field, both renders) |
| Field tensors hashed before and after all renders | unchanged on all frames |
| `state_digest(model)` before = after | `bf161a43…a571`, unchanged |
| Stereo, depth, centre, Chamfer metrics and `source_index`/`centre_support`/`frozen_disparity`/`base_centres`/`calibration` hashes, A-original vs A-corrected | identical on all 56 frames |

So A-corrected differs from A-original only by the camera shift.

**Paired results (corrected − original, 56 frames; CI is a keyframe-cluster bootstrap with only 4 clusters, indicative):**

| Metric | Original | Corrected | Mean diff | Median | Improved / degraded | CI95 |
|---|---|---|---|---|---|---|
| left PSNR dB | 25.4702 | 25.7108 | **+0.2406** | +0.2112 | 56 / 0 | [+0.204, +0.332] |
| right PSNR dB | 21.6653 | 21.8428 | **+0.1775** | +0.1672 | 56 / 0 | [+0.156, +0.213] |
| left SSIM | 0.7743 | 0.7751 | +0.0008 | +0.0006 | 29 / 27 | [−0.0002, +0.0029] |
| right SSIM | 0.7178 | 0.7222 | +0.0044 | +0.0038 | 50 / 6 | [+0.0036, +0.0059] |
| left LPIPS (lower is better) | 0.2845 | 0.2897 | **+0.0053 (worse)** | +0.0039 | 15 / 41 | [+0.0015, +0.0079] |
| right LPIPS | 0.3339 | 0.3363 | +0.0025 (worse) | +0.0004 | 26 / 30 | [−0.0003, +0.0043] |

Per-keyframe PSNR change (left/right): kf1 (8 frames) +0.40/+0.25; kf2 (15) +0.20/+0.16; kf3 (7) +0.26/+0.15; kf4 (26) +0.21/+0.17 dB, the same sign in all four.

**Error by image region (pooled MSE ratio corrected/original; share of the total squared-error change):**

| Region | Left ratio (share) | Right ratio (share) |
|---|---|---|
| all | 0.947 | 0.964 |
| interior (≥16 px from border) | 0.945 (93.2%) | 0.959 (83.2%) |
| 16-px border band | 0.963 (6.8%) | 0.977 (16.8%) |
| interior flat (gradient ≤ median) | 0.961 (18.2%) | 0.969 (22.4%) |
| interior mid gradient (median–p90) | 0.941 (43.7%) | 0.955 (37.6%) |
| interior edge (gradient > p90, 10% of interior pixels) | 0.938 (31.4%) | 0.950 (23.2%) |
| right-view uncovered strip (`right_edge_strip_mask`) | | 0.983 (26.6%) |
| right-view covered | | 0.941 (73.4%) |
| inside / outside sensor support | all pixels inside | all pixels inside |

OBSERVED: the gain is spread over the interior, slightly larger in high-gradient pixels (ratio 0.938 vs 0.961 flat; edge pixels carry 31% of the change from 10% of the interior), and smallest in the part of the right view that the left view cannot observe. It is not confined to the image border.

**Supplementary render-only shift sweep (same forward pass; shift added to `cx, cy`; 0 = existing renderer; means over 56 frames):**

| Diagonal shift (px) | −0.50 | −0.25 | 0 | +0.25 | **+0.50** | +0.75 | **+1.00** | +1.25 | +1.50 | +2.00 | +2.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| left PSNR dB | 24.946 | 25.219 | 25.470 | 25.616 | 25.711 | 25.856 | **25.947** | 25.900 | 25.797 | 25.666 | 25.276 |
| left SSIM | 0.7410 | 0.7580 | 0.7743 | 0.7760 | 0.7751 | 0.7845 | **0.7914** | 0.7827 | 0.7716 | 0.7691 | 0.7404 |
| right PSNR dB | 21.394 | 21.534 | 21.665 | 21.761 | 21.843 | 21.935 | 22.010 | 22.040 | 22.050 | **22.079** | 21.999 |
| right SSIM | 0.6905 | 0.7041 | 0.7178 | 0.7206 | 0.7222 | 0.7303 | **0.7373** | 0.7324 | 0.7262 | 0.7265 | 0.7064 |

x-only / y-only at +1.00: left PSNR 25.637 / 25.778 (diagonal +1.00: 25.947). Per-frame best left diagonal shift: +1.00 in 32/56 frames, +1.25 in 8, +2.00 in 9, +0.50 in 2.

OBSERVED: A's left-view PSNR and SSIM peak at a diagonal shift of about **+1.0 px**, i.e. about 0.5 px beyond the +0.5 correction. The best left PSNR is +0.48 dB over the original (+0.24 dB at +0.5). The right view keeps improving to about +2.0 px in PSNR (its optimum is broader and also depends on the student's disparity error of 3.54 px EPE on this set), so only the left view gives a clean registration optimum.

---

## 3. Scientific interpretation (observed vs hypothesis)

**Q1 — Does the corrected projection improve reconstruction quality?** OBSERVED: partly. PSNR improves on 56/56 frames in both views (+0.24 / +0.18 dB), right SSIM improves (+0.0044), left SSIM is unchanged, and **LPIPS is slightly worse** (left +0.0053, 41/56 frames). The correction therefore improves pixel fidelity modestly and does not improve the perceptual metric. The metrics disagree; no single "improves quality" statement is supported. HYPOTHESIS (H-phase, §3 Q2): part of this PSNR/LPIPS disagreement at exactly +0.5 px may be a sampling-phase effect rather than registration; it cannot be separated with the two conditions and the coarse 0.25 px sweep.

**Q2 — Does it reveal compensation already learned by the frozen Stage 1?**
- Provenance, now verified: clean10 was trained with a photometric reconstruction loss through this renderer (`student_kd_control.yaml`: `reconstruction_weight 1.0`, `kd_weight 1.0`, `joint_epochs 0`; loss `reconstruction_weight·rgb + kd_weight·kd` with `rgb = reconstruction_loss(state, base)` over both views, `training/student_learning.py:602` and `training/student_runtime.py:343` in the clean10 snapshot; whole model trainable; `production.py` byte-identical). So **Stage 1 could have adapted**; whether it did is UNVERIFIED.
- **Verdict: undetermined.** Two conditions cannot answer this, and the sweep cannot either: reading "compensation" off the optimum position would require knowing where the optimum would be without any compensation, and A's observed left optimum (about +1.0 px, not +0.5) shows that this reference point is itself unexplained. Until the extra 0.5 px is explained, the sweep speaks neither for nor against compensation learned by Stage 1.
- A left optimum about 0.5 px beyond the convention fix is **OBSERVED but unexplained**. Two untested hypotheses:
  - **H-lattice:** the stride-2 lattice anchors each Gaussian at the top-left pixel of a 2×2 block whose centre is at +0.5 px in both axes (`gaussians/sampling.py:236-253`, offset `(0,0)`, stride 2, `common.yaml:107-115`), which together with the +0.5 convention fix predicts an optimum near +1.0. A stride-1 field would predict about +0.5.
  - **H-phase:** sub-pixel sampling phase. At shifts 0 and +1.0 the Gaussian centres sit at half-pixel positions, at +0.5 on integer pixel centres, so the sweep can contain a period-1 ripple on top of any registration trend. Consistent with this, left SSIM dips at +0.5 (0.7760 at +0.25, 0.7751 at +0.5, 0.7845 at +0.75) while PSNR keeps rising, and LPIPS is worse at +0.5. If this holds, the "+1.0 optimum" is not a clean 0.5 px residual offset.
  Other causes are not excluded.

**Q3 — Does it change the interpretation of the historical A/B/C comparison?**
- OBSERVED: for the frozen A field, the half-pixel correction is worth +0.24 dB (left) and the whole sweep at most +0.48 dB, against historical differences of A→B +3.02 dB, B→C +2.89 dB, A→C +5.91 dB (left PSNR, validation). A pure global translation therefore moves A's PSNR by about six times less than the B→C difference (+0.48 dB at the best swept shift vs +2.89 dB).
- This does **not** measure how B or C respond to a shift. B and C were trained under the offset renderer; evaluating them under a corrected renderer would be biased (C's learned lateral displacement could become an overshoot). No portion of C's improvement is attributed to the renderer mismatch in this addendum, and none can be without a controlled measurement on B and C themselves (§5).
- HYPOTHESIS (generated, not tested): C's constant centre-head bias, converted to the image plane, is about (+1.0, +0.9) px at 54 mm depth (`b·R` = (0.092, 0.081, −0.135) mm; `Δu = f·Δx/Z`, f = 592.06 px), numerically close to A's optimal shift of about (+1.0, +1.0) px. Caveats: crude depth (54 mm), the state-dependent `W·h` term is of comparable size (§9 E1 of the original audit), a constant in mm is a depth-dependent shift in pixels, and A's optimum is not necessarily B's or C's. It is a reason to run the controlled test of §5, not a result.

---

## 4. Implications for Stage 1 and the B/C experiments

- **Stage 1 (clean10 student and head):** trained through the offset renderer with a photometric term (verified, above). Evaluation-time geometry metrics (disparity, depth, centres, GT-referenced metrics) never use the renderer and are unaffected at evaluation. **During training, however, a bias in the learned disparity from the offset is possible:** the right-view photometric term compared a Gaussian rendered at `u_L − d − 0.5` against a target at `u_L − d_true`, which could push the learned disparity by up to about 0.5 px horizontally (with the KD term pulling against it), whereas a vertical misregistration cannot be absorbed by disparity. No artifact settles this (the signed `centre_z_bias` changes sign between datasets and is not diagnostic). This is UNVERIFIED. Rendering-related adaptation of the head is likewise UNVERIFIED.
- **A→B and A→C contrasts:** unchanged in interpretation for pure attribute refinement, but their absolute values and any registration component of the gain remain uncharacterised.
- **B/C:** both trained and evaluated under the same −0.5 px renderer, so the B/C pairing is internally matched (original audit F-01 to F-09 stand). The mismatch is a shared-pipeline confounder that only C, with lateral freedom, can partly absorb. Its size for C is unknown.
- **Scope of the mismatch:** every render through `native_camera_batch_from_rectified` / `ProductionFieldRenderer` (single call site `production.py:719`; used by our evaluation, student training, the repository's matched Endo baseline path, phase 2 and downstream CLIs). Renders that call the native rasterizer with their own camera construction (`baseline/upstream.py`, `baseline/posed_gaussians_native.py`, `evaluation/external_geometry.py`) and external methods (MVSplat, DepthSplat, upstream Endo-E2E-GS) were **not** checked. Whether any comparison is tilted, and in which direction, is unknown. This addendum proposes no paper change.
- **Geometry claims:** unchanged. GT-referenced geometry never used the renderer, and remains flat between A, B and C.

---

## 5. Is a clean retraining experiment justified, and what comes next?

Evidence for and against, all from this addendum:
- The mismatch is real, but its measured effect on A is small (+0.24 dB left, mixed perceptual metrics) and the larger residual (+0.5 px beyond the fix) has no demonstrated cause. Retraining B, C (and especially Stage 1) under a corrected renderer today would fix a convention without knowing whether the convention is the dominant confound.
- A clean B/C attribution **does** eventually require training under a corrected renderer, because the existing checkpoints cannot be re-read under a different renderer without bias. That is a planned, not an immediate, step.

**Verdict: retraining is not yet justified. The recommended next experiment is eval-only and directly controlled; it requires your approval before anything is run:**

> On validation dataset_7 only, with the existing frozen checkpoints and the original renderer, evaluate **A, B, C and C-zero-offset** (C with XYZ offsets set to zero immediately before rendering, attributes unchanged) each under a render-only global shift sweep (shift added to `cx, cy` at the native frustum, as here) over {0, +0.5, +1.0, +1.5} px **plus a finer step of about 0.1 px between +0.5 and +1.25**, reporting PSNR, SSIM **and LPIPS** at every shift, so that a period-1 sampling-phase ripple (H-phase) can be separated from a monotonic registration trend. Also record the actual per-slot projected displacement of C in pixels, split into lateral (x, y) and depth-related components.

How to read it (decided in advance, so the result is not fitted afterwards):
1. If the best shift of B is near +1.0 but C's best shift is near 0, and C-zero's best shift moves towards +1.0, then part of C's gain is absorbed global registration. The size of that part is measured as `PSNR(B at its best shift) − PSNR(B at 0)` compared with `PSNR(C) − PSNR(B)`.
2. If C-zero keeps most of C's gain, the gain is carried by attribute co-adaptation rather than direct movement.
3. If all four share the same best shift and C-zero loses most of the gain, the gain is direct, shift-independent centre movement.
This design needs no training and supplies exactly the "direct controlled measurement" that the attribution of C's gain requires. Because the earlier C-zero evaluation was gated on the pixel-convention result, nothing is run until you approve it. Only after it, and only if the registration component is material, is a **paired B/C retraining under the corrected renderer** (Stage 1 frozen, at least two seeds, validation first) justified. Retraining Stage 1 itself would be a separate, much larger decision.

---

## 6. Corrections to earlier statements

1. The intermediate probe report (`outputs/gad_pixel_convention_probe_20261113T215032/REPORT.md` §6) gave an image-vs-itself shift proxy and arithmetic suggesting up to about +1 dB of C−B might be a compensation. That arithmetic was not a bound, and the proxy overestimated: for A the measured effect is +0.24 dB left where the proxy implied roughly +0.4 dB. The "+1 dB" figure should not be used.
2. The statement "the clean10 student may have been trained through this renderer" is now **verified** (§3 Q2).
3. The scope of affected renders is limited to the camera builder named in §4; claims about other pipelines remain unverified.

---

## 7. Integrity

No training, no checkpoint or production-source modification, no test datasets, no writes to the experiment run root (a `find` for files newer than the first probe over both the run root and the clean10 directory returned nothing), tracked files unchanged (the six pre-existing modified files are unmodified by this work). The only repository files created are this addendum and the two `outputs/` directories. GPU work, all on GPU `8a3f4d31…` and all short: the single-Gaussian probe, its half-pixel-camera repeat, the image-shift proxy (56 validation images, no model), the A original/corrected evaluation with the sweep, and the extended sweep (the last two each ran the frozen A forward pass once per sealed validation id).
