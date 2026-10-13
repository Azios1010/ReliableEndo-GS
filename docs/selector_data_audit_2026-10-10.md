# Selector data audit — 2026-10-10

## Scope and evidence

Read-only analysis of the completed quarter-cohort refit, followed by bounded
data-loader and offline-label diagnostics corrections. Original SCARED and its
processed TIFF/PNG/JSON files were not changed. Train remains dataset_1–7;
dataset_8–9 did not participate. Utility weights, risk cutoffs, ROI definitions,
partition, checkpoint weights and the completed experiment were not changed.

The completed refit is
`selector-quarter-preparation-L0xgcQNW/refit` under the external run root.
Its selector SHA-256 is
`1dff8ffba14e622db045718389a76aa5a53c11e8d2c0ddfef07bc7c715431d28`.
The original outcome-generator source SHA-256 is
`a189afa9041dfa8d629c09e2aee75096f8cd2f73baef33cdc780da8dbbda86b2`.
The new offline audit is `selector-data-audit-20261010-01/audit.json` under that
same external run root; it binds the exact audit script, source snapshot,
mask index, partition and completed result/outcome files by SHA-256. It contains
per-sample support, calibration and source-identity evidence without executing
a model. It is a derived report, not a rewrite of the experiment metrics.

## Findings and corrections

### HIGH — Resized GT sampled a different spatial convention from RGB

The processed grid is 1024×1280; this run used 512×640. RGB and intrinsics
use `u' = s*(u+0.5)-0.5`. Legacy PyTorch `nearest` picked source indices
0,2,4,…; nearest-exact picks the closest source centres 1,3,5,… at this ratio.
Both disparity and validity were consistently sampled together, but on the
wrong nearest grid relative to RGB. Constant-disparity tests missed this.

On the 112 diagnostic samples, GT validity differs at 9,301,531 output pixels
between the two conventions. This is a count of changed validity decisions,
not newly available labels. Total valid pixels are 8,253,154 before and
8,260,557 after. All 111 samples containing GT change. On pixels valid under
both rules, the macro mean absolute disparity change is approximately
0.641445 output pixels; nearest surface boundaries can produce larger changes.
These are GT-to-GT resampling differences, not model error or improvement.

**Fixed:** runtime supervision and validity use nearest-exact. Sparse holes
stay holes; no averaging, filling or pseudo-GT is introduced. Tests independently
compute source indices from the inverse pixel-centre affine for downsampling,
noninteger resizing and upsampling, including asymmetric crops and NaN/Inf.
Deployment RGB and cameras remain equal with or without supervision.

### MEDIUM — Source grid metadata reported the resized dimensions

`calibration_metadata.input_hw` was read from the already resized RGB tensor.
It therefore reported 512×640 for a 1024×1280 source.

**Fixed:** record the dimensions before crop/resize and explicitly record RGB,
GT and validity resampling semantics. This changes calibration/cache identity;
old outcomes and teacher-cache entries must not be silently reused as current.

### HIGH — GT observation coverage differs across internal TRAIN roles

The saved 5,738-image collection contains 7,143 proposed regions, of which
1,514 (21.20%) lack complete metric labels. 1,077 images (18.77%) have no known
region and contribute no selector label. This includes images with no candidates.
Both repair labels are unavailable together in these recorded unknown regions.

| Internal role | Images | Regions | Unknown regions | Images without a known region |
| --- | ---: | ---: | ---: | ---: |
| FIT | 2,788 | 3,163 | 922 | 729 |
| Validation | 1,572 | 1,962 | 204 | 66 |
| Calibration | 1,378 | 2,018 | 388 | 282 |

Known stereo harm rates are 828/2,241 in FIT, 556/1,758 in validation, and
415/1,630 in calibration. Selection bias and keyframe distribution therefore
remain relevant even after calibrating the head. These counts describe the
historical sampling rule and frozen actions; corrected labels require a fresh
collection. They do not prove that distribution shift alone caused the failures.

**Corrected observability:** fitting and calibration reports now count observed
and excluded regions/images by keyframe. Offline collection logs exact writable
pixel indices and per-action GT/fixed/lost support counts with explicit missing
reasons. Joint-label logs and evaluation diagnostics use the same evidence.
Unknown labels remain excluded and are never changed into safe labels.

### HIGH — Missing local GT mostly persists after correcting the resize

The 112-sample evaluation has 126 candidate regions. 21 have no local GT with
the old sampling rule; 20 still have none with nearest-exact. Sixteen selected
interventions lacked complete metrics; 15 had global GT but no local GT, and
one had no GT anywhere. Correcting the sampling rule cannot recover unobserved
ground truth or certify these interventions as safe.

`dataset_7/keyframe_4/frame_data001453` has an intentionally empty processed
GT TIFF. Its source provenance reports 67 usable left XYZ points before stereo
overlap filtering, but zero points inside both rectified grids. This was not a
missing file, a failed decode or a discovered frame-ID mismatch.

The deployment ROI remains a coarse RGB-reviewed operator box inherited from
temporal anchors. It is not a tissue segmentation, and absence of GT does not
mean background. No mask was cropped to GT support or modified using utility
outcomes. Better foreground identification requires an RGB-only protocol and
its own review; choosing ROI from GT would leak offline supervision.
Four other regions have only 1–31 aligned GT pixels. Their local error estimates
remain sparse measurements; this audit did not invent a minimum-count safety
threshold or certify them as statistically reliable.

### Evidence — Calibration and source identity checks passed on 112 samples

- All 672 checked source-file SHA-256 hashes match the sealed snapshot.
- The three distinct calibration-evidence files match their full declared hashes;
  their referenced original calibration YAML bytes also match. This check is
  saved separately in `calibration-provenance-check.json` beside the audit.
- Left/right RGB and GT grids match at 1024×1280.
- All 112 provenance records match sample, frame-data and scene-point IDs,
  with no declared ambiguity. Frame-data bytes match the recorded source-member
  hash. Original scene-point archive bytes were not reread in this audit.
- `R2 @ R @ R1.T` and `R2 @ T * 0.001` agree with rectified right-from-left
  transforms within 1e-5 rotation and 1e-7 m translation tolerances.
- P1/P2 agree with rectified intrinsics and the declared mm-to-m baseline.
- Recomputed Q depth versus A00 depth at up to 200 valid, aligned sampled pixels
  per image has maximum absolute error 9.095e-13 mm. This checks consistency;
  it does not independently prove metric units.
- Existing preprocessing provenance records independent metric-GT validation
  on all 111 nonempty samples; maximum recorded Q-to-GT error is 0.001953125 mm.
  This audit inspected that historical evidence; it did not rerun raw-cloud
  metric validation or establish units from baseline magnitude.

### Remaining algorithm/model limits

Nineteen selected interventions violated measured guards in the completed run.
Fourteen failed the local RGB guard; four failed right PSNR; two each failed
local MAE, outside RGB and local RMSE. Counts overlap. Missing depth supervision
therefore does not explain all harm, particularly observed render regressions.

The repairs still have only 168 updates on 56 images; the larger collection
refitted the selector with frozen repairs. Rendered-depth metrics remain
unavailable because the renderer exports no metric depth. Development shares
parent keyframes and is diagnostic evidence, not final generalization evidence.
None of the data corrections establishes that routing beats fixed Gaussian
repair, lowers violation frequency, or passes a research gate.

## Next experiment boundary

Keep the old run intact. Regenerate offline action outcomes in a fresh external
run using the corrected loader and original sealed TRAIN roles/ROIs, then refit
and evaluate on the same diagnostic samples. Existing source bindings should
reject resuming corrected labels into the old collection. Do not rewrite old
hashes or disable those checks. Processed SCARED files do not need reprocessing.

## Validation and runtime limits

- 14 CPU tests passed: the complete A00 adapter suite, fixed metric support,
  missing/local/lost GT support and metric-diagnostic aggregation.
- 18 additional CPU tests passed, 1 deselected: selector outcome collection,
  canonical labels/unknown exclusion, calibration abstention, affine calibration,
  outcome views, source/ROI preparation and immutable shard resume.
- Corrected A00 loader preflight passed on three real samples: nonempty GT,
  missing local GT in the historical intervention, and entirely empty GT.
  Source-index selection matches an independent NumPy calculation exactly;
  RGB/cameras do not depend on whether GT is loaded. CUDA was not initialized.
- Scoped Ruff lint/format passed; mypy passed on seven changed source modules;
  `git diff --check` passed. Import/path inspection found no new baseline,
  third-party or SCARED-C import in the affected data/label owners.
- Two wider attempts did not complete: the first aborted in PyTorch's CUDA
  availability probe (the stuck process was then killed for cleanup); the second
  aborted in CPU backward after using the NVML availability check. The fitting
  backward case was explicitly deselected in the successful restricted suite.
  The wider joint-gradient suite is not verified by this audit.
- A separate detached-tmux GPU0 probe reported CUDA available, then timed out
  during a two-element allocation/backward check (exit 124, 90-second bound).
  Native CUDA/backward validation therefore remains blocked in this runner.
  GPU1 and other users' processes were not modified.

Full CI, a fresh outcome collection/refit, native model smoke, teacher inference,
real-data training, and final-test evaluation were not run. The passing data
preflight is not a claim that native training is ready or that selector quality
has improved after these corrections.

## Files changed in this correction

Existing unrelated working-tree changes are retained. This correction touched:

- `src/reliable_endo_gs/data/scared_a00.py`
- `src/reliable_endo_gs/evaluation/tradeoff_outcomes.py`
- `src/reliable_endo_gs/evaluation/mask_metrics.py`
- `src/reliable_endo_gs/training/balanced_interventions.py`
- `src/reliable_endo_gs/training/balanced_learning.py`
- `src/reliable_endo_gs/training/selector_outcomes.py`
- `src/reliable_endo_gs/training/selector_fitting.py`
- `tests/data/test_scared_a00_adapter.py`
- `tests/evaluation/test_mask_validation.py`
- `tests/routing/test_selector_refit.py`
- `docs/a00_scared_protocol.md`
- `docs/selector_data_audit_2026-10-10.md`
