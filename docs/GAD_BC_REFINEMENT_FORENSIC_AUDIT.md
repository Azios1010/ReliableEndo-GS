# GaD B/C Refinement — Forensic Audit (read-only)

Date: 2026-11-13. Repository HEAD `5e8fb2330b35753aacf85c8f0a581b75a4de0100` plus uncommitted/untracked source (the experiment ran from a hashed source snapshot, see §3). Scope: the single-seed A/B/C study run **R** = `/mnt/d/3DGS_runs/reliableendo-gs/gad_decoupling_single_seed/s20261006-20261112T082123Z/`.

**Nothing was trained, evaluated or benchmarked for this audit, no GPU/CUDA API was touched, and no existing source, config, checkpoint or result was modified.** Only this file was created. Section 12 lists exactly what was read, hashed or deserialised.

Classification of every finding:

- **CONFIRMED**: supported by an executed artifact (log, checkpoint, rows, hash) or by a code path that provably ran. A config value alone never counts.
- **CONTRADICTED**: an artifact or code path disagrees with the stated intention or protocol.
- **UNVERIFIED**: no artifact or code evidence either way.
- **REQUIRES EXPERIMENT**: could only be settled by a run that this audit is forbidden to perform.

Arm definitions (`docs/GAD_DECOUPLING_SINGLE_SEED_PREPARATION.md:59-63`): **A** = initial clean10 Gaussian field, no refiner. **B** = fresh `11→32→32→7` attribute MLP, centres fixed. **C** = the same MLP plus `Linear(32→3)` XYZ head on the shared hidden features, bounded to 5 mm.

---

## 1. Executive summary

1. **The B/C pair is well matched on everything this audit could inspect, except the centre head itself and operational placement.** The same frozen digest, identical shared initial tensors, identical ordered sample list, identical optimiser settings and update count, per-frame bitwise-identical base/support/calibration hashes, and bitwise-identical stereo metrics across arms (F-01 to F-09, F-30).
2. **The rendering gain of C over B is not an evaluation artifact specific to C.** It appears in the training loss from the first 1,000 updates (§9, E3), measured before each update on a frame not yet used for training, so it does not depend on the evaluation pipeline, data root or GPU. (It does not by itself show generalisation: neighbouring video frames are near-duplicates.) The gain also appears on 56/56 validation frames (left) and in the official test summary. A defect shared by both arms' render path is **not** excluded (C-03, §8).
3. **"Mean displacement ≈ 0.19 mm" is not small in image space.** With native fx ≈ 1184 px at 1280 px (≈ 592 px at 640: the CUDA resize accepts only an exact 2× reduction, `data/scared_a00.py:439-462`) and typical depth ≈ 54–66 mm, 1 mm lateral ≈ 9–11 px. So 0.19 mm lateral ≈ 1.7–2.1 px, about one stride-2 lattice spacing; p99 ≈ 0.54–0.62 mm ≈ 5–6 px; max 3.7 mm ≈ 40 px. The apparent "tiny movement, large gain" paradox mostly disappears (§9, E1).
4. **That does not establish direct-movement causality.** C is not "B plus 99 parameters". The shared trunk also receives the centre loss gradient. At update 1 the total gradient norm is 0.219 (C) vs 0.0103 (B), and the final trunk weights diverge by about the same magnitude as each arm's own movement (§5, §9 E2). The left-view gain (+2.89 dB val, positive on 56/56 frames) exceeds the right-view gain, which a depth-only correction cannot explain (§9 E1).
5. **No geometric benefit is demonstrated.** GT-referenced centre metrics are unchanged within error, and the Chamfer metric is blind to sub-millimetre effects by construction (§7). Do not write that centre freedom improves geometry. The supported statement is that, in this single-seed run, bounded centre offsets coincided with higher PSNR/SSIM and unchanged GT-referenced centre accuracy; the cause is not isolated.
6. **One source-level hypothesis cannot be dismissed and could affect all arms.** The data/backprojection stack declares `integer_pixel_centers` while the native camera builder passes `cx` straight into a Graphdeco frustum. The native rasterizer source is not vendored and no test found asserts a rendered pixel position through the real backend, so a possible half-pixel offset is UNVERIFIED and REQUIRES EXPERIMENT (§9 E4). If real, C may be partly compensating a pipeline offset that B cannot. The only directional evidence available (sign of the centre-bias loss gradient and of the learned bias) is *consistent with, not confirming*, such an offset. This is why the cheapest probe (§10 P1) is ranked first.
7. **Reliability: provenance and matching are high; the C>B rendering effect is probably real for these two checkpoints but is single-seed and nondeterministic; the mechanism is unidentified; the geometry claim is unsupported.** Highest-value next step: the pixel-convention probe (cheapest, and it conditions how every contrast is read), then the eval-only C-zero-offset plus displacement decomposition (§10).

Authorised protocol deviations (user-approved, none is a B/C confound per se, all recorded): concurrent GPU workloads, C trained on GPU0 while B on GPU1, and a different evaluation data root (§6, F-29/F-30).

---

## 2. PASS / FAIL / UNVERIFIED register

| ID | Invariant | Verdict | Class | Key evidence (details in later sections) |
|---|---|---|---|---|
| F-01 | Same frozen clean10 student + initial head in B and C | PASS | CONFIRMED | `frozen_digest.model = bf161a43eb480d0ccbdfd4dd4e1dd33bef4ceb93702e3458160fc52af5aba571` in both `summary.json` and both `step-00013150.pt` identities, `frozen_unchanged: true` (`training/field_refinement.py:292-302`). Input byte hashes recomputed (§3). The digest itself was **not** recomputed (needs the model); only its recorded before/after equality is confirmed |
| F-02 | Same Gaussian construction/sampling | PASS | CONFIRMED | `source_index`, `centre_support`, `frozen_disparity`, `base_centres`, `calibration` SHA-256 equal across arms on every frame (validation 56×3 and test 5,907×3, checked per frame by me and enforced by `decoupling_summary.py:65-74`, which completed for both splits). `loss_stop_reference` equal in 527/527 logged training rows, max abs diff 0.0 |
| F-03 | Same refiner architecture and shared initial tensors | PASS | CONFIRMED | `initial.pt` shared keys `torch.equal`; `shared_initial_sha256 = 9b7d0f49…a32` in both; 1,671 vs 1,770 parameters; C adds only `centre_output.{weight,bias}` (zeros) |
| F-04 | Same sample order | PASS | CONFIRMED | `ordered_samples_sha256 = e0bcc6ce…7aa5` in both; the 527 logged step→sample pairs equal; both `calibration.jsonl` are byte-identical (13,150 unique ids, equal to `ordered_ids.json`) |
| F-05 | Same seed | PASS (narrow) | CONFIRMED | Seed only fixes the init (`training/decoupling_run.py:70-72`). Training is **not bit-reproducible**: B's update-1 `grad_norm` for the identical init and frame is 0.010319380 (main run, GPU1), 0.010319376 (readiness, GPU1, `…/gad_bc_gpu1_shared_20261113T142444/readiness_steps.jsonl`) and 0.010319388 (readiness, GPU0, `…/gad_bc_parallel_20261113T144531/readiness_steps.jsonl`), i.e. relative differences of ~4e-7 on the same device and ~8e-7 across devices. C's update-1 norm (0.21879319846630096) is identical in all three runs |
| F-06 | Same loss, masks, optimiser, clip, update count | PASS | CONFIRMED | `training_config` equal (`lr 0.001, grad_clip 1.0, steps 13150`); optimiser groups at update 13150: `lr 0.001, betas (0.9,0.999), eps 1e-8, wd 0`; every Adam `step = 13150`; max logged `grad_norm` < 1 so clipping was inactive in all logged rows |
| F-07 | Same source colours | PASS | CONFIRMED (eval) / CONFIRMED by code (train) | `evaluation/decoupling.py:26-27` raises on any colour change; every one of 5,907+56 rows was written. Training: `compose_gaussian_field` returns clamped `prior.rgb` (`gaussians/composition.py:293-297`), no colour residual; per-step colour equality is not logged |
| F-08 | Same camera/rasterizer settings | PASS (partial) | CONFIRMED + UNVERIFIED | `_model()` raises unless `(znear, zfar, native_units_per_metre) == (0.01, 100, 20)`, 512×640, `context_scale_m == 1` (`decoupling_run.py:186-194`), and it runs for every train/eval launch. TF32/cuDNN flags and the native backend binary revision at run time are not recorded: UNVERIFIED |
| F-09 | Training completion | PASS | CONFIRMED | `completed_updates 13150`, `summary.json steps 13150`, 16 files in each `checkpoints/`, `latest.json → step-00013150.pt`; B 1,931.6 s, C 1,821.9 s |
| F-10 | Offsets in left-camera metres, one conversion at the native boundary | PASS | CONFIRMED (code) | `geometry/centre_offsets.py:1,66-82`; `rendering/production.py:512-516` multiplies means/scales by 20 once. Readiness probe: max displacement 0.0838 mm ≤ 5.001 mm |
| F-11 | Bounding formula, near/far safety | PASS | CONFIRMED | `δ = R·v/√(1+‖v‖²)`, `R = clamp(0.5·min margin to near/far in both cameras, 0, 5 mm)`; `‖δ‖ < R ≤ margin/2`. `radius_saturation_fraction = 0.0` in every logged training row and every evaluated frame, so the bound never bound |
| F-12 | Zero initialisation and identity | PASS | CONFIRMED | `centre_output` zeros in `initial.pt`; step-1 `loss == loss_stop_reference == 0.04032675921916962` in both arms; readiness asserts `torch.equal` left/right images and all field tensors on one frame per arm |
| F-13 | Invalid/padded slot handling | UNVERIFIED | UNVERIFIED | Code zeroes radius and raw for invalid slots (`centre_offsets.py:49,63`, `centre_refinement.py:46-47`), but **all 81,920 slots were valid in every evaluated frame** (`base_valid_centres` min = max = 81,920 for A, C, val and test). The path never ran in evaluation. Training validity is not logged |
| F-14 | Offset applied once, after composition, to all valid slots, same field for both views | PASS | CONFIRMED | `actions/refinement.py:97-114,175-177`; `actions/repair.py:411-429` |
| F-15 | Source index / colours / slot order unchanged | PASS | CONFIRMED | colour equality asserted per row; `source_index_sha256` equal across arms; `centre_geometry.py:30-31` raises if validity changes |
| F-16 | No Python-side sampling/filtering/pruning change from moving centres | PASS | CONFIRMED | `valid_indices` come from the unchanged base `valid_mask` (`rendering/production.py:492-493`); same 81,920 primitives sent in all arms |
| F-17 | No native in-rasterizer visibility change from moving centres | UNVERIFIED | UNVERIFIED | Native culling is internal; `visibility` radii are returned but never exported in rows |
| F-18 | Student and initial head frozen; no gradient reaches base predictor | PASS | CONFIRMED | `model.eval().requires_grad_(False)` and base forward under `no_grad` (`training/field_refinement.py:109,220-226`); frozen digest unchanged |
| F-19 | B trains rotation/scale/opacity only; C the same plus XYZ | PASS | CONFIRMED (actual optimiser state) | B: one group, 6 tensors; C: groups `attributes_and_shared_trunk` (6) and `centre_output` (2) in `step-00013150.pt` |
| F-20 | C's centre gradient propagates through the rasterizer | PASS (narrow) | CONFIRMED (narrow) | Native finite-difference on the centre-bias Y axis, one train frame, zero offset: analytic −0.189621 vs FD −0.190089 (0.25% rel.) in `…/gad_bc_parallel_20261113T144531/readiness.json`. Z axis, head weights and trunk are not FD-checked |
| F-21 | No GT depth/disparity in the objective | PASS | CONFIRMED | `include_supervision=False` (`decoupling_run.py:264`); loss uses images only (`training/interventions.py:31-63`). Note: the left mask is `RepairState.support`, which includes **predicted** teacher validity, never GT (`actions/repair.py:53-57`) |
| F-22 | Centre head not detached; zero offset does not suppress gradient | PASS | CONFIRMED | Update-1 `grad_norm` 0.2188 (C) vs 0.0103 (B); final head weights nonzero (Frobenius 0.376); `bounded_centre_offsets` has derivative `R` at 0 |
| F-23 | Feature normalisation, bounds, composition identical | PASS | CONFIRMED | no normalisation layer in `FieldRefiner`; identical `refiner_config`; both arms call the same `refine_gaussian_field`; C only adds the centre path |
| F-24 | Train/inference code parity | PASS | CONFIRMED | training and `EagerDecouplingPredictor` both call `refine_gaussian_field` with defaults (`reconstruction/decoupling.py:53-58`); FP32; no capture (`actions/refinement.py:376-377` rejects centre capture) |
| F-25 | Identical metric definitions, rosters, ids, GT support across arms | PASS | CONFIRMED | `decoupling_summary.py:56-74`; my per-frame checks (§7) |
| F-26 | Per-frame aggregation and missing values | PASS | CONFIRMED | `decoupling_summary.py:91-100`; `None` counts are identical across arms (GT-driven) |
| F-27 | No opacity-based filtering of centre metrics | PASS | CONFIRMED | `centre_geometry.py:36-49`, `benchmark_geometry.py:198-203` filter by base validity and GT support only |
| F-28 | Rendered-depth metrics unavailable by design, not by defect | PASS | CONFIRMED | `evaluation/decoupling.py:40` passes `rendered_depth=None`; renderer returns only image+visibility (`production.py:539`); `evaluation.yaml:69-71` marks it BLOCKED |
| F-29 | Sealed protocol: single GPU, exclusive, sequential, one device for all arms | FAIL | **CONTRADICTED** (user-authorised) | `common.yaml:97 concurrent_gpu_workloads: false` vs actual: B shared GPU1 with a MatRoute job (`SHARED_GPU_AUTHORIZATION.json`), C trained concurrently on GPU0; evaluation A,B on GPU1, C on GPU0 |
| F-30 | Cross-device effect on the frozen **forward** outputs | PASS (forward) / not bitwise (backward) | CONFIRMED | `frozen_disparity` and `base_centres` hashes bitwise equal across devices; `loss_stop_reference` identical on 527 rows. The refiner **backward** is not bit-identical across devices (F-05), so training B on GPU1 and C on GPU0 adds a small numerical trajectory difference on top of the centre head (C-01) |
| F-31 | Pixel-centre convention consistent between data/backprojection and native renderer | UNVERIFIED | **REQUIRES EXPERIMENT** | §9 E4 |
| F-32 | Evaluation does not mutate state; test freeze | PASS | CONFIRMED | `status.json state_unchanged: true` (all); `test_freeze.json finals` equal recomputed `sha256sum` of both `final.pt` |
| F-33 | Chamfer as a test of a 0.19 mm effect | FAIL (insensitive) | CONFIRMED limitation | §7 |
| F-34 | Training data root equals evaluation data root | FAIL (differs) | CONTRADICTED vs. prep doc; benign so far | train: `…/linux-data`; eval: `SCARED_A00` (`authorization.json`). 90 sampled files byte-identical (§8 C-07), not exhaustive |

---

## 3. Exact B/C architecture and checkpoint provenance

Run identity (`R/identity.json`): `protocol 6cbcb229de7b5ef6360d511c436af449a0a4a8faf270b8c4967a2686a6cd3e7d`, `source 6ffe5bb1f7de69223d4efbdaaa0b29bd4fdb035485f1650c8ed8c8bdc407fffe`, `shared_initial 9b7d0f493b928f934b01879d2fea7ae791edab41df0b5199a2c39290a48ca032`.

| Item | B (`fixed_centres/`) | C (`trainable_centres/`) |
|---|---|---|
| `initial.pt` SHA-256 | `10ec34a2…3ac4` | `38194d88…c9ad` (differs only by arm id + `centre_output`) |
| `final.pt` SHA-256 (recomputed) | `c4001555912c719a356db7f5f13d93a3ec44d7d133ce1af7352db5f5c94e19b0` | `3ac4cf280b0441a7f1cf112e61a9ea987a19eb8ceb252f5f8869f2705a94e464` |
| `step-00013150.pt` SHA-256 (`latest.json`) | `0f923813…ab25` | `fba2cf30…1060f` |
| Parameter tensors / count | `body.{0,2}.{weight,bias}`, `output.{weight,bias}` = 1,671 | the same + `centre_output.{weight (3,32), bias (3)}` = 1,770 |
| dtype / finite | float32 / all finite | float32 / all finite |
| Training device | GPU `8a3f4d31…` (GPU1), shared with another job at start | GPU `b9607686…` (GPU0) via adapter `…/gad_bc_parallel_20261113T144531/C_gpu0.py` (sha `ffc5d034…0020`) |
| Time window / duration | ~14:20–14:56, 1,931.6 s | ~14:39–15:13, 1,821.9 s |
| Peak allocated / reserved | 575,159,296 / 639,631,360 B | 582,010,880 / 639,631,360 B |
| Frozen digest before = after | `bf161a43…a571` | `bf161a43…a571` |

Upstream inputs (recomputed now with `sha256sum`, equal to recorded values): clean10 `/mnt/d/3DGS_runs/reliableendo-gs/clean10-20261103/kd/FINAL.pt` = `db7413e8d456177ee55eb7e3cb75580b5cfd40c443403359f8173fb21a6131b6`; construction `…/phase2-frozen-actions-20261028T114155Z/stage-a-attempt-00/final.pt` = `ef7fa35f49e1ce21bd5994dedcb93b638750f98ca21afa68f37d1feb24fb4ba4`. Run-time re-verification of these plus manifests, roles and LPIPS weights is enforced at every train and eval launch (`config/decoupling.py:184-241`, called via `_verified_run`, `decoupling_run.py:140-151`).

Source integrity: all 329 files in `R/source_hashes.json` have identical hashes in `R/source_snapshot/` and in the current working tree. Evaluation ran with `PYTHONPATH` and `cwd` on the snapshot (`supervise_evaluation.py`, `one()`), and `_verified_run` rejects any identity drift. The working tree is nevertheless dirty relative to HEAD (`environment.json git_diff`, 31,493 chars, excludes untracked files), so the snapshot, not HEAD, is the reproducible artifact.

Configuration actually used vs intended: the only raw-file differences between `R/config_snapshot/` (training) and the evaluation `configs/` are execution gates (`common.yaml: training/inference/evaluation`, `evaluation.yaml` and `initial_field.yaml: execution_authorized`); the sealed scientific fingerprint excludes them (`config/decoupling.py:33-38,129-135`). Intended-vs-actual items:

| Setting | Intended (YAML) | Actual (artifact) |
|---|---|---|
| Seed / order | 20261006, sorted ids + `Random.shuffle` | order hash equal (`e0bcc6ce…`); RNG states saved in checkpoints |
| GPU | one UUID `8a3f…` | B `8a3f…`; C `b960…` (`summary.json` carries both declared and actual) |
| Concurrency | none | concurrent (user-authorised) |
| Alpha epsilon | `common.yaml:180 alpha_logit_epsilon 1.0e-4` | `composition_alpha_epsilon 1e-07` in both step-checkpoint identities (`actions/refinement.py:99` overwrites the bound with the construction value). Same for both arms |
| Precision | FP32 | `execution_mode eager_fp32`, float32 tensors |
| Evaluation data root | `${RELIABLE_ENDO_DATA_ROOT}` | `SCARED_A00` (override) |

---

## 4. Forward-pass reconstruction

Verified in `src/reliable_endo_gs/` (snapshot-identical):

1. Frozen student → disparity → `reconstruct_rectified` (principal-point offset corrected once, `geometry/rectified.py:22-53`) → left-camera metre centres on the fixed stride-2 lattice (`common.yaml:107-115`, 81,920 = 256×320 slots, integer pixel coordinates, `gaussians/sampling.py:227-253`) → base `GaussianField` (all under `no_grad`, `training/field_refinement.py:220-226`).
2. `context = context_from_geometry_prior(...)` from the **detached composed base field**; the refiner sees 11 of 13 channels (rgb, `mu_z`, normal, `S0/scale`, `alpha0`; absolute x/y excluded) (`actions/refinement.py:97`).
3. `FieldRefiner.forward_with_features`: `hidden = body(features)` (two 32-wide GELU layers); `raw = output(hidden)`; tanh-bounded rotation (0.7 rad), log-scale (1.5), alpha-logit (4.0) (`gaussians/refinement.py:146-156`).
4. `CentreFieldRefiner.forward` reuses the same `hidden`: `raw_centre = centre_output(hidden)`, zeroed for invalid slots (`gaussians/centre_refinement.py:41-48`).
5. Attributes composed once by `compose_gaussian_field`; exact-zero identity preserves derivatives (`actions/refinement.py:99-102`; `actions/repair.py:145-175`). `compose_gaussian_field` returns `means = prior.mu = base.means3d` (`composition.py:298`).
6. Centre path (C only): `radius = centre_safety_radii(field.means3d, core, camera, cfg)` from the detached base centres using both cameras' Z (`centre_offsets.py:49-63`); `delta = bounded_centre_offsets(raw, radius)` (`:66-82`); `changed = replace(changed, means3d=field.means3d + delta)` (`actions/refinement.py:107-109`). This is the **only** place centres change, and it runs once.
7. `merge_field(base, changed, core)` with the full-frame core (all valid slots) (`actions/repair.py:87-91,129-142`); `_render_outcome` renders the **same** field through the left and right cameras (`actions/repair.py:422-428`).
8. Native boundary: only valid slots are sent; means and scales are multiplied by `native_units_per_metre = 20` once (`rendering/production.py:489-526`); camera translations and clip planes use the same factor (`:121-123`).
9. Loss: mean of left and right masked Charbonnier (`epsilon 1e-3`, `rendering/cross_view.py:257-264`) with `RepairState.support` (sensor ∧ predicted-validity) on the left and sensor support on the right (`training/interventions.py:46-63`). **The right-view target is the observed right image, so centre freedom is supervised photometrically in both views.**

Backward path in C: loss → both renders → rasterizer gradient on `means3d` → `delta` → `bounded_centre_offsets` → `centre_output` → `hidden` → trunk. The base context is detached, so nothing flows into the student, head, priors or camera.

---

## 5. Gradient-flow and optimiser audit

**Actual optimiser state** (from `step-00013150.pt`, CPU deserialisation, `weights_only=True`):

- B: one param group `attributes_and_shared_trunk`, 6 tensors, Adam state for each, `step = 13150`.
- C: groups `attributes_and_shared_trunk` (6 tensors) and `centre_output` (2 tensors, `lr 0.001`), `step = 13150` on all 8.
- `trainable_state == final.pt state_dict` in both arms.

**Logged gradients** (`train.jsonl`, 527 rows each; only the total norm is logged, no per-group norm):

| Quantity | B | C |
|---|---|---|
| `grad_norm`, update 1 | 0.01032 | **0.21879** (≈21×) |
| `grad_norm`, median over logged rows | 0.0184 | 0.0933 |
| max logged `grad_norm` > 1 (clip active) | never | never |
| mean displacement, updates ≤25 / 25–500 / 500–2000 / 2000–5000 / 5000–9000 / 9000–13150 | 0 | 0.066 / 0.237 / 0.229 / 0.236 / 0.201 / 0.184 mm |
| `radius_saturation_fraction` | n/a | 0.0 in every row |
| `centre_safety_limited_count`, max | n/a | 7,522 of 81,920 |

Findings:

- **CONFIRMED:** the centre head is not detached and receives a large, non-suppressed gradient from update 1 (F-22). A native finite difference agrees with the analytic gradient on one coordinate at zero (F-20). That check does not cover Z, head weights or trunk gradients (**UNVERIFIED** for those).
- **CONFIRMED:** the shared trunk is trained by a *sum* of the attribute and centre objectives in C but by the attribute objective alone in B. At update 1 trunk gradients are exactly zero in both arms (zero-initialised output layers), and C's total norm is dominated by the centre head (≈21× B's whole gradient). Whether the centre path dominates the **trunk** gradient later is **UNVERIFIED** (per-group norms not logged).
- **CONFIRMED:** Adam normalises per parameter, so a large centre-path gradient matters for the trunk only through its relative share in each parameter's moment estimates. This cannot be resolved from the logged totals.
- **CONFIRMED:** trunk and output weights end far apart (L2 norms of the final B−C difference: `body.0.weight` 27.7, `body.2.weight` 10.7, `output.weight` 8.17, `output.bias` 1.09). Each arm moved from the shared init by comparable amounts (e.g. `body.0.weight` 28.0 for B, 34.9 for C; initial norm 3.14). `output.bias` moved 0.470 (B) vs 1.477 (C). The two trajectories are different solutions, not B with a small add-on. This is also what any nondeterministic re-run could produce, so by itself it does not attribute the difference to the centre head (C-01).
- **CONFIRMED:** no GT depth/disparity and no teacher loss enter training (F-21); the head trains only `refiner.parameters()` (`training/field_refinement.py:118,121-127`).

---

## 6. Training/inference parity

| Aspect | B vs C | Train vs eval | Evidence |
|---|---|---|---|
| Feature selection (11 channels, no normalisation layer) | identical | identical | `common.yaml:139-162`; identical `refiner_config` |
| Attribute bounds, composition, alpha epsilon | identical | identical | `actions/refinement.py:99-102`; `1e-07` in both identities |
| Gaussian field composition | identical, then C adds `+delta` | same function | `refined_gaussian_field` |
| Rasterizer constants, background `[0,0,0]`, units | identical | identical | `_model()` asserts; camera builder |
| Precision | FP32 | FP32 | `model.float()` (`decoupling_run.py:195`), float32 state dicts |
| Preprocessing (512×640, `include_supervision=False`) | identical | identical | `decoupling_run.py:261-268,369-376` |
| Checkpoint loading | strict, identity-checked | strict | `_correction` (`:209-237`) |
| Image masks | identical | **differ by design**: training uses masked Charbonnier; evaluation PSNR/SSIM/LPIPS are full-image | `interventions.py:46-63`; `evaluation.yaml:98-118` |
| Base render path | A uses the model's own renders; B/C use `_render_outcome` | the same function in train and eval | zero-residual identity verified on 1 frame per arm (readiness); not on all frames |
| `sample_ids`/`data_role` passed to `model()` | train passes them, eval does not | validation-only (role check), no compute effect | `reconstruction/model.py:627-644` |

**Does C bypass or duplicate an attribute-correction step?** No. C calls the same single `compose_gaussian_field` and then adds the centre delta once (`actions/refinement.py:99-109`). The attribute residuals are computed from the unmoved context, so moving centres does not feed back into context, sampling or priors (`common.yaml:105`).

**Differences other than the centre head** (all CONFIRMED, all user-authorised or benign so far):

1. GPU placement and concurrency (F-29), covered by F-30.
2. Evaluation data root differs from the training data root (F-34).
3. The Python-level evaluation of B at test time ran on GPU1 after another process, C on GPU0. Different GPU, same model (RTX 5070 Ti).
4. The deployed fast path (`RefinedInference.capture`) explicitly rejects centre refiners (`actions/refinement.py:376-377`). Results apply only to common eager FP32.

---

## 7. Evaluation and metric audit

Metric chain: `training/decoupling_run.py:369-381` → `evaluation/decoupling.py:15-87`.

| Metric | Computation | Notes |
|---|---|---|
| left/right PSNR | `evaluation/rendering.py:40-66`, full image, no mask, data range 1; MSE=0 maps to a recorded `positive_infinity` | no infinities occurred in any row |
| left/right SSIM | `windowed_ssim`, 11×11, σ 1.5, zero padding, full image | same for all arms |
| left/right LPIPS | `LPIPSAlexNet` with hashed weights | same for all arms |
| centre displacement mean/median/p90/p95/p99/max | `centre_geometry.py:50,60-78`, over all base-valid slots in mm, per frame; summaries average per-frame percentiles | support is independent of opacity |
| centre correspondence MAE/RMSE, Z MAE/RMSE/bias/AbsRel | `centre_geometry.py:41-49,79-95`; error = `refined.means3d − GT point at the fixed original source pixel`, mm | fixed support (base-valid ∧ GT-supported); `None` if no support |
| centre Chamfer | `benchmark_geometry.py:193-208`, `cloud_metrics` with `max_points=1024` | predicted cloud filtered by base-valid ∧ GT-supported ∧ finite; equispaced 1,024-point subsample per side |
| stereo EPE/depth | `evaluate_geometry` from frozen disparity | bitwise identical across arms on every frame (checked) |
| unit conversion | metres → mm (`×1000`) at metric owners only (`centre_geometry.py:50-51`; `benchmark_geometry.py:165-166`) | no other conversion |

**Frame counts and identifiers (CONFIRMED).** Validation 56/56/56 identical sets; test 5,907 (dataset_8 3,152, dataset_9 2,755) in all arms; `status.json complete: true, state_unchanged: true` for all six arm×split runs. The official `evaluation/{validation,test}/summary.json` exist; the summarizer refuses to run on any hash mismatch (`decoupling_summary.py:56-74`).

**Missing values (CONFIRMED).** Disparity/depth metrics are `None` for 1/56 validation and 100/5,907 test frames, centre correspondence/Chamfer for 145 test frames, and surface Chamfer for 240. These counts are identical in A, B and C and match the sealed `expected_historical_gt_valid_frames` of 55 and 5,807 (`common.yaml:72,84`). They come from missing GT support, not from model output.

**Why rendered depth is unavailable (CONFIRMED).** The production renderer returns `RenderOutput(image, visibility)` only (`rendering/production.py:539`). `decoupling_frame_metrics` therefore passes `rendered_depth=None` (`evaluation/decoupling.py:40`), and `benchmark_geometry.py:209-217` fills the four `rendered_depth_*` rows with `None` and the reason string. The planned moment-depth diagnostic is BLOCKED in `evaluation.yaml:69-97`. This is a deliberate gap, not a silent failure.

**Metric sensitivity (CONFIRMED limitations, not defects):**

- *Chamfer cannot resolve this effect.* Each cloud is capped at 1,024 equispaced points (`valid_count` 2,048 of ~159k). The nearest-neighbour distance floor is of order the sample spacing, i.e. millimetres, versus a 0.19 mm displacement. "Chamfer unchanged" (validation 3.0989 → 3.1070 mm; test C−B +0.0003 mm, CI [−0.028, +0.029]) is **not** evidence that geometry is unchanged.
- *Correspondence MAE penalises lateral moves by construction.* The target is the GT point at the original source pixel, so a centre moved sideways is scored as error even if the photometric fit improves. `centre_z_mae` is the more meaningful along-axis metric, and it changes by only −0.009 mm (validation) and −0.0127 mm [−0.033, +0.009] (test, pooled) against a 4.4 mm error. The displacement is ≈4% of that error, so these metrics could not reveal an effect of that size either.
- *Aggregation.* Test CIs come from a keyframe-cluster bootstrap with **4 keyframes per dataset** (`decoupling_summary.py:14-40`). With so few clusters the intervals are weak evidence even before adding between-seed variance (single seed: "data-cluster sensitivity only").

**Results (official, from the two `summary.json` files).**

Validation, dataset_7, 56 frames:

| | A | B | C | C−B |
|---|---|---|---|---|
| left PSNR dB | 25.470 | 28.493 | 31.380 | +2.886 (56/56 frames > 0; median +2.838; min +0.382) |
| right PSNR dB | 21.665 | 22.835 | 24.308 | +1.473 (52/56 > 0; min −0.351) |
| left / right SSIM | 0.774 / 0.718 | 0.857 / 0.783 | 0.897 / 0.829 | |
| left / right LPIPS | 0.284 / 0.334 | 0.189 / 0.254 | 0.178 / 0.238 | |

Per-keyframe C−B (right/left PSNR): kf1 (8 frames) +2.05/+3.20; kf2 (15) +1.30/+1.64; kf3 (7) +1.45/+2.87; kf4 (26) +1.40/+3.51. Frame-level correlation of mean displacement with the right-PSNR gain is 0.39 (left 0.18), but the per-frame displacement spans only 0.14–0.24 mm, so this is not usable as causal evidence.

Test, pooled / dataset_8 / dataset_9 (5,907 / 3,152 / 2,755 frames), C−B with 95% keyframe-cluster CI:

| Metric | pooled | dataset_8 | dataset_9 |
|---|---|---|---|
| left PSNR | +2.165 [1.910, 2.428] | +2.621 [2.323, 3.030] | +1.643 [1.218, 1.982] |
| right PSNR | +0.743 [0.540, 0.949] | +0.544 [0.418, 0.786] | +0.972 [0.639, 1.243] |
| left LPIPS | −0.0090 [−0.0138, −0.0037] | −0.0018 [−0.0069, +0.0020] | −0.0172 [−0.0233, −0.0095] |
| right LPIPS | −0.0072 [−0.0118, −0.0018] | +0.0016 [−0.0025, +0.0046] | −0.0172 [−0.0227, −0.0095] |
| right SSIM (means) | 0.7737 → 0.7832 | 0.8396 → 0.8395 | 0.6982 → 0.7187 |
| `centre_z_mae` mm | −0.0127 [−0.033, +0.009] | +0.0054 [−0.028, +0.039] | −0.0345 [−0.053, −0.019] |
| `centre_corr_mae` mm | +0.0092 [−0.023, +0.042] | +0.0193 [−0.020, +0.059] | −0.0029 [−0.048, +0.046] |

Test pooled B−A: left +1.770 [1.615, 1.886], right +1.041 [0.991, 1.097]. C−A: left +3.935 [3.698, 4.177], right +1.785 [1.575, 2.007]. Test C displacement: mean 0.185, p95 0.253, p99 0.618 mm.

Observations: the C>B advantage is largest on PSNR and left-view fidelity, much smaller on LPIPS, and absent for right-view SSIM/LPIPS on dataset_8. Validation overstates the test effect (right +1.47 vs +0.74 dB). The GT-referenced centre metrics show no consistent change (only dataset_9 `centre_z_mae` has a CI excluding zero, by 0.035 mm).

---

## 8. Potential confounders

| ID | Confounder | Status |
|---|---|---|
| C-01 | **Single seed, non-reproducible training, different training devices.** B and C are one trajectory each; update-1 gradient norms differ at ~1e-7–1e-6 relative between repeated launches (F-05), and B and C ran on different GPUs; there is no replicate of B or C. The C−B difference includes trajectory variance of unknown size (the size of noise amplification over 13,150 updates is not measured) | REQUIRES EXPERIMENT |
| C-02 | **C ≠ B + 99 parameters.** The centre loss also trains the shared trunk and output layer (§5). Attribute weights differ substantially between arms | CONFIRMED (difference); effect size REQUIRES EXPERIMENT |
| C-03 | **Possible pipeline offset shared by all arms** (pixel-centre convention, §9 E4) | UNVERIFIED / REQUIRES EXPERIMENT |
| C-04 | **Rendering metrics measure reconstruction of the input pair**, not novel views. The right image is both input and training target; evaluation PSNR is full-image and includes the right-edge strip no left-anchored Gaussian can cover (`actions/refinement.py:121-138` documents the strip). Same for all arms | CONFIRMED |
| C-05 | **Insensitive geometry metrics** (§7) | CONFIRMED |
| C-06 | **Operational deviations** (concurrency, GPU split, C on GPU0 in training and evaluation) | CONFIRMED, user-authorised; forward base outputs bitwise equal across devices, refiner backward not bit-identical (F-30, C-01) |
| C-07 | **Training vs evaluation data root.** 90 sampled files (42 from datasets 1/2/3/6, 48 from dataset_7; left/right images, `newpram`, disparity) are byte-identical between the training cache and `SCARED_A00`. This is a spot check, not an exhaustive comparison, and `final_test` datasets 8/9 are not in the training cache | UNVERIFIED beyond the sample |
| C-08 | **Few clusters and few datasets.** 4 keyframes per dataset, datasets 7/8/9 only | CONFIRMED |
| C-09 | **TF32/cuDNN determinism flags not recorded** (`common.yaml:92 tf32_policy: identical_record_resolved_environment`, nothing recorded in artifacts) | UNVERIFIED |
| C-10 | **Fixed colours on a stride-2 lattice.** Colours are sampled at integer source pixels; with a 2 px lattice, sub-pixel centre freedom has an obvious photometric degree of freedom that B lacks | CONFIRMED by design |
| C-11 | **Eager FP32 only.** The centre refiner cannot use the deployed capture path | CONFIRMED |
| C-12 | **All evaluated frames have 81,920 valid slots**, so invalid/padded behaviour is untested in practice | CONFIRMED |

---

## 9. Evidence-supported explanations for C's gains

**Magnitude check (ESTIMATE, labelled).** Native intrinsics (`/mnt/d/3DGS_datasets/preprocessing/SCARED_A00/dataset_7/keyframe_1/data/newpram_data/frame_data000000.json`): `fx = fy = 1184.116` px, baseline 4.364 mm. The pipeline resizes with `u' = sx·(u − crop_x + 0.5) − 0.5` (`geometry/rectified.py:56-76`). The CUDA image path used in training and evaluation accepts only identity or an exact 2× reduction, documented as the 1280×1024 → 640×512 run (`data/scared_a00.py:439-462`), so `f ≈ 592.06` px at 640 wide (the per-frame intrinsics are not dumped in any artifact; the value follows from this constraint and the dataset K). Typical depth is taken from `MAE/AbsRel` as ≈54 mm (validation) and ≈66 mm (test), a crude proxy. Then 1 mm lateral ≈ 11.0 (val) / 9.0 (test) px; 0.19 mm lateral ≈ 2.1 / 1.7 px; the bias-only x/y components (0.092, 0.081 mm) ≈ 1.0 / 0.9 px; p99 (0.54 mm) ≈ 5.9 px; max 3.7 mm ≈ 40 px. A pure Z move of 0.19 mm changes right-view disparity by `fBΔZ/Z² ≈ 0.17` px against a baseline disparity EPE of 3.54 px (validation).

**E1 — Small projected centre movements directly improve image alignment.**
Status: **plausible on magnitude, not demonstrated. REQUIRES EXPERIMENT.**
- For: the displacement is about one lattice spacing in pixel terms, large enough to alter overlap, coverage and sub-pixel alignment. Training loss falls ~29% below the unrefined reference for C vs ~16% for B in the last logged windows.
- Against or unresolved: (a) the Z component is too small to correct disparity-scale error (0.17 px vs 3.5 px), so any alignment gain must be lateral or via footprint/ordering effects; (b) base centres sit on their left source-pixel rays, so a move **along** the ray changes footprint size, not projected position, yet the **left** gain (+2.89 dB, 56/56 frames) exceeds the right gain, which a depth-only correction cannot explain; off-ray components, attribute changes or compensation of a systematic offset are required; (c) the displacement is not a pure constant translation: the bias-only displacement `5 mm·b/√(1+‖b‖²)` is (+0.092, +0.081, −0.135) mm, norm 0.182 mm (`b = (0.01845, 0.01627, −0.02699)`), close to the mean 0.185 mm, but the validation `centre_z_bias` change is **+0.046 mm** (C−A) against a bias-only z of −0.135 mm, so the state-dependent `W·h` term is of comparable size. Do not describe it as a global rigid shift.

**E2 — The centre head changes shared-feature optimisation and the learned attribute corrections.**
Status: **supported as a real difference between arms; its contribution is unquantified. REQUIRES EXPERIMENT.**
- Evidence: trunk/output weights diverge at the scale of each arm's own movement; `output.bias` moves 3× further in C; update-1 gradient is dominated by the centre head; C's training loss is lower from the first 1,000 updates (mean window loss 0.0290 vs 0.0323, reference 0.0338) and stays lower (0.0239 vs 0.0284 over updates 12,000–13,150).
- Consequence for design: a **C-zero-offset evaluation is necessary but not sufficient**. Zeroing C's offsets leaves attributes that were co-trained with offsets, so it measures "C's attributes without their centres", not "B plus centres". It could even fall below B.

**E3 — An unintended implementation/evaluation difference inflates the scores.**
Status: **largely ruled out for any C-specific evaluation artifact; not ruled out for a shared pipeline defect.**
- Ruled out by artifacts: arm-identical hashes per frame; bitwise identical stereo metrics; same code path in train and eval; colour equality asserted on every row; no opacity/visibility filtering; no state mutation. The training-loss gap is **independent of the evaluation pipeline and data root** (it is computed inside training on the training cache; B and C also ran on different GPUs). Each of the 13,150 training frames is used for one update, and the logged loss is taken before the update on that frame. This supports "not an evaluation-pipeline artifact" but not generalisation: neighbouring video frames from the same four datasets are near-duplicates and are shuffled across the run, so the loss is not an independent held-out estimate.
- Not ruled out: anything common to training and evaluation, notably E4.

**E4 — C compensates a systematic projection/lattice offset shared by A and B (additional hypothesis).**
Status: **UNVERIFIED. REQUIRES EXPERIMENT.**
- Source evidence only: the geometry stack declares integer pixel centres (`geometry/conventions.py:16-21`; `a00_protocol.json "pixel_center": "integer"`), and `backproject_depth`/`crop_resize_camera` keep that convention. The native camera builder uses `cx` directly in an OpenGL-style frustum (`rendering/production.py:84-175`). In the upstream Graphdeco rasterizer that this ABI targets, NDC maps to pixels as `((ndc+1)·S − 1)/2`, which would place a centre at `u − 0.5` relative to integer pixel indices. The native source is not vendored in this repository. No test found renders through the real native backend and asserts a projected pixel position: `tests/rendering/test_metric_native_boundary.py:55-96` uses a fake `CaptureBackend` and compares metric-vs-native projection matrices (unit scaling) and gradients only, and `test_production_renderer.py` tests covariance packing and call shapes. So this is a hypothesis, not a finding.
- **Directional evidence, consistent with but not confirming E4.** At zero offset on the first training frame, the native finite-difference/analytic gradient of the loss w.r.t. `centre_output.bias[y]` is −0.1896 (`readiness.json`), i.e. a uniform +y (downward, x-right/y-down) centre shift lowers the loss. The final learned bias is `(+0.018, +0.016, −0.027)`, positive in x and y. A render displaced by (−0.5, −0.5) px (up-left) would be compensated by +x/+y centre shifts. The sign agreement is one frame and one coordinate (the x and z gradient signs were not logged), and could be scene-specific.
- Scale and attribution: E4 cannot account for the whole displacement (0.5 px ≈ 0.046 mm lateral at 54 mm versus a 0.19 mm mean), and its share of the PSNR gain is unknown: displacement magnitude does not bound it. If E4 were real, part of A→B would also be compensation by blur/scale rather than learned refinement, and it would change how A→B, B→C and C−A are all read.

**What the existing artifacts cannot do:** they contain no per-slot displacement vectors, no renders, and no per-component (x, y, z) statistics, so the direction of movement, its constancy, and its interaction with the image-plane remain unmeasured. Do not infer causality from the mean displacement alone.

**Is a C-zero-offset evaluation needed?** Yes, to isolate the direct rendering contribution of centre movement, but alone it is confounded by co-adaptation (E2). See §10. It was **not** executed.

---

## 10. Minimal follow-up tests (prioritised; none executed or implemented)

| P | Test | Cost / needs | Resolves |
|---|---|---|---|
| 1 | **Pixel-convention probe**: native render of a single-Gaussian fixture, measure the rendered centroid against the projected `u` under the repo's integer-centre convention; if an offset exists, a global shift applied to A and B at evaluation | tiny GPU probe (seconds) + eval-only | E4, C-03; changes how A→B, B→C and C−A are all read |
| 2 | **C offsets zeroed at evaluation** (C attributes, `δ = 0`), validation first, on existing `final.pt`; optionally the cross cell (B attributes with C's centre head) flagged as out-of-distribution. With existing A, B, C this gives the 2×2 {B-attrs, C-attrs}×{offsets off, on} | eval-only, existing checkpoints, GPU | direct vs co-adapted contribution (E1 vs E2) |
| 3 | **Displacement decomposition**: per-slot signed x/y/z components in camera frame and in pixels, ray-parallel vs ray-perpendicular, constant (`b`) vs state-dependent (`W·h`) parts, by depth and by image position | eval-only, same pass as P2 | E1; "global shift?" |
| 4 | **Column/edge error profiles** (`training/field_refinement.py:317-324` exists) for A/B/C, plus error restricted to the right-edge strip | eval-only with render retention | where right/left gains live |
| 5 | **Training variance**: a second seed (or a repeat on one device) for B and C | two training runs, new approval | C-01 |
| 6 | **Detached-trunk C′** (centre head reads `hidden.detach()`), trained like B/C | one training run, new approval, new arm | separates E1 from E2 causally |
| 7 | **Data-root equivalence**: exhaustive byte hashes between training cache and `SCARED_A00` for datasets 1/2/3/6/7 | CPU only | C-07 |
| 8 | Per-group gradient norms and trunk-gradient share logged during a short probe | short GPU run, approval | §5 open item |

Why P1 is first: it is the cheapest item, and it decides whether the pairwise contrasts (A→B, B→C, C−A) measure learned refinement at all or partly measure compensation of a pipeline offset. The sign of the centre-bias gradient and of the learned bias (E4) is consistent with such an offset but proves nothing. P2 then addresses the direct-vs-co-adapted question the user asked about; it comes second because its interpretation depends on P1 and on the co-adaptation caveat (E2).

Test-set decisions must not depend on these results; P1–P4 can be run on validation first.

---

## 11. Final scientific reliability assessment

- **Experimental identity (B vs C): high confidence.** Every invariant that could be checked passed, apart from user-authorised operational deviations and one untested path (invalid slots).
- **Existence of a C>B rendering gain in these checkpoints: moderate-to-high.** It is consistent across validation (56/56 left), test pooled and both test datasets for PSNR, and in the training loss on a different data root and device. It is weaker for perceptual metrics and absent for right-view SSIM/LPIPS on dataset_8. Single seed, no replicate, 4 keyframe clusters per dataset: **low confidence that the magnitude generalises**.
- **Attribution to centre movement: not established.** E1 and E2 are both live; E4 is a plausible shared-pipeline confounder; none is excluded.
- **Geometry claim: unsupported.** GT-referenced centre metrics are flat within CIs, and the available geometry metrics could not detect an effect of this size anyway. Rendered depth was never measured.
- **Safe wording** (consistent with `AGENTS.md`: hypotheses are not results): "In this single-seed run, allowing bounded centre offsets (≤5 mm; mean 0.19 mm) coincided with higher PSNR/SSIM than fixed-centre refinement on held-out frames, with unchanged GT-referenced centre accuracy; the cause was not isolated. The proposal hypothesizes that centre freedom improves geometry; this run neither supports nor refutes that."
- **Unsafe wording:** "centre refinement improves geometry/depth", "0.19 mm displacement explains +2.9 dB", "no effect on geometry".

---

## 12. Audit method and limits

Read: source files cited above, `docs/GAD_DECOUPLING_SINGLE_SEED_PREPARATION.md`, `R/{identity,config_hashes,source_hashes,resolved_spec,test_freeze,environment}.json`, `R/config_snapshot/*`, `outputs/gad_bc_*` control directories (authorisations, adapters, readiness records), `R/{fixed_centres,trainable_centres}/{summary.json, checkpoints/latest.json, attempt-*/train.jsonl, attempt-*/calibration.jsonl}`, `R/evaluation/{validation,test}/{A,B,C}/{rows.jsonl,status.json}` and both `summary.json`, and dataset calibration JSON for dataset_7.

Computed without models or GPUs: `sha256sum` of 6 large/small files; CPU deserialisation with `CUDA_VISIBLE_DEVICES=""` and `weights_only=True` of `initial.pt`, `final.pt` and `step-00013150.pt` of B and C (tensors only); arithmetic over JSON logs and rows; byte comparison of 90 data files between two roots. Analysis scripts were ephemeral (session scratchpad), not repository files; every number above is derived from the named artifact files.

Not inspected: native CUDA rasterizer source (not in the repository), student/`build_student` internals, `context_from_geometry_prior`/`build_geometry_prior` bodies, per-slot displacement vectors, rendered images (none were saved), TF32/cuDNN flags. Evaluation of test arm B was already running when the audit began and finished at ~18:00 under the existing controller; it was not started or touched by this audit. After it finished I re-checked all three test arms directly from `rows.jsonl`: identical frame sets, identical per-frame base/support/calibration hashes, bitwise-identical stereo metrics, identical `None` counts, no infinite or missing PSNR, `base_valid_centres = 81,920` and zero displacement in B, and `complete`/`state_unchanged` true for A, B and C (A, B on GPU `8a3f4d31…`, C on `b9607686…`). Provisional per-dataset A/C means computed earlier in the session from `rows.jsonl` were superseded by the official summary used here.
