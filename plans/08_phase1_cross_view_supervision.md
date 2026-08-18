# Plan 08 — Phase I Cross-View Supervision

## Status

PLANNED

## Research stage

Phase I

## Recommended executor

Sol. Right-camera supervision combines renderer/camera correctness, endoscopic non-Lambertian masking, robust losses, training stability, and claim attribution.

## Estimated implementation risk

HIGH. Specularity, occlusion, tools, pose/calibration noise, and appearance compensation can create photometric gains without better geometry.

## Objective

Train and evaluate the Phase I representation with conservative real right-camera supervision, while distinguishing cross-view rendering consistency from genuine geometry improvement.

## Scientific question

Does real stereo cross-view supervision improve geometry and held-out-view reliability, or does it only smooth/fit right-view appearance?

## Why this milestone exists

Same-view rendering can let scale or opacity hide center error. A verified second camera provides an identifiability test, but it must be isolated from covariance changes and protected against invalid endoscopic pixels.

## Prerequisites

- Plan 07 representation candidate and cross-view-off checkpoint/config.
- Verified left/right cameras, rectification, masks, and renderer behavior.
- Frozen data split and no-test tuning policy.

## Inputs

- Phase I Gaussian field generated from the left/reference stereo path.
- Observed left/right images and cameras.
- Left-right, occlusion, specularity, depth-validity, tool, visibility, and saturation evidence that is actually available and validated.

## Outputs

- Cross-view mask builder, robust loss, training flow, and checkpoint.
- Cross-view-off/on ablation with left, right masked/full, and geometry metrics.
- Failure-mode report and candidate full Phase I configuration for Gate I.

## Scope

- Render the same field from true left and right cameras.
- Begin with conservative analytic masks; add learned masks only as a separately justified future ablation.
- Compare Charbonnier and robust L1; consider census/feature consistency only when raw photometry is demonstrably unreliable and the feature source is fixed.
- Preserve baseline/Phase I stereo and left-render objectives while adding the right-view term through a documented curriculum.
- Report masked and full-image right metrics, left metrics, geometry metrics, and mask coverage.
- Diagnose specularity, occlusion, tools, depth edges, low texture, and far range.

## Non-goals

- No virtual-camera substitution for the real right camera, raw full-image L2 default, test-mask tuning, router/action, or claim that binocular consistency itself is novel.
- Do not hide left-view or geometry degradation behind improved masked right-view photometry.

## Files expected to be created

- `src/reliable_endo_gs/rendering/cross_view.py`
- `src/reliable_endo_gs/training/phase1.py`
- `src/reliable_endo_gs/evaluation/cross_view.py`
- `configs/training/phase1.yaml`
- `configs/evaluation/cross_view.yaml`
- `configs/experiment/p1e_cross_view.yaml`
- `configs/experiment/p1f_full_ablation.yaml`
- `scripts/train_phase1.py`
- `scripts/evaluate_phase1.py`
- `tests/rendering/test_cross_view_masks.py`
- `tests/rendering/test_cross_view_loss.py`
- `tests/integration/test_phase1_training_smoke.py`

## Files expected to be modified

- Renderer request/evaluation code for an explicit right camera.
- Phase I artifact candidate manifest fields.
- Data masks only if authoritative SCARED evidence supports them.

## Interfaces and contracts

`build_cross_view_mask(state, evidence)` returns named component masks, a combined valid mask, counts, and reasons. `cross_view_loss(render, observation, mask, loss_config)` is pure and mask-aware. Phase I training consumes a frozen uncertainty-provider identity and cannot import Phase II.

## Scientific formulation

Owned by `rendering/cross_view.py`:

```text
I_hat_L = R(G, P_L)
I_hat_R = R(G, P_R)
M_valid = M_LR (1-M_occ) (1-M_spec) M_depth (1-M_tool)
L_xview = sum_i M_valid,i rho(I_hat_R,i-I_R,i) / (sum_i M_valid,i + epsilon)
```

The masking and robust photometric formulation are supporting prior/implementation choices, not standalone novelty. The program hypothesis is that real cross-view consequence helps verify the reliability representation. Tests cover mask algebra, camera selection, robust-loss values/gradients, and empty masks.

## Numerical and coordinate conventions

Right-camera transforms and color/range must match verified contracts. Projection handles out-of-view/behind-camera points. Empty masks return a declared skipped loss and diagnostics, not a fake zero-success metric. `epsilon` ownership is explicit. Record saturated/specular/tool/occluded fractions and avoid interpolation mismatches.

## Configuration changes

Define loss family/scale, component masks, thresholds learned only on validation, left/right/stereo/stability weights, curriculum stages, freeze/unfreeze schedule, seeds, and checkpoint selection metric. Keep cross-view-off and cross-view-on experiment identities separate.

## Tests

### Unit

- Component-mask combination, out-of-view and empty-mask behavior.
- Charbonnier/robust L1 values, gradients, and normalization.
- Correct left/right camera use on analytic scenes.

### Contract

- Render/output shapes and masks; training cannot consume Phase II fields.

### Integration

- Frozen uncertainty + probabilistic field -> right render -> masked loss -> backward pass.

### Smoke

- Tiny train/evaluate cycle with production renderer and complete provenance.

### Regression

- Analytic cross-view fixtures and mask-coverage summaries.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/rendering tests/integration
pytest -q
python -m compileall src
python scripts/evaluate_phase1.py --config configs/experiment/p1e_cross_view.yaml
git diff --check
```

## Experiments

1. Cross-view off versus on with the same Plan 07 representation.
2. Robust L1 versus Charbonnier; optional feature/census only if justified.
3. Component-mask and conservative/full mask ablations on validation.
4. Full Phase I ablation across required representation variants and seeds.

## Metrics

- Masked and full-image right PSNR/SSIM/LPIPS.
- Left PSNR/SSIM/LPIPS.
- Depth/point/normal/edge errors and outlier ratio.
- Mask coverage by cause and valid counts.
- Training/inference latency and peak memory; right render is training overhead unless used at inference.

## Acceptance criteria

- Cross-view supervision is stable and uses real verified right-camera geometry.
- Any claimed geometry gain appears in geometry/held-out-view metrics, not only masked photometry.
- Left-view and runtime/memory tradeoffs are reported and within Gate I review bounds.
- Cross-view-off/on attribution and full/masked reporting are complete.

## Failure conditions

- Gains disappear on full images or held-out sequences, geometry degrades, masks retain only easy pixels, or calibration/specularity makes loss unstable.

## Pivot / rollback path

Use more conservative masks, robust/feature/census consistency, or remove the photometric term. If only right-view appearance improves, limit the claim to cross-view rendering consistency. If no reliable gain remains, freeze the Plan 07 representation without cross-view training and report the negative result.

## Artifact/provenance requirements

Record Plan 07 artifact/config, camera/mask/loss schema, thresholds and calibration split, seeds, checkpoints, backend, data/split hashes, per-sequence full/masked/geometry metrics, mask coverage, and code/hardware identity.

## Completion checklist

- [ ] Real right camera and mask components are verified.
- [ ] Robust masked loss and empty cases are tested.
- [ ] Cross-view-off/on ablation is complete.
- [ ] Masked/full, left/right, and geometry metrics are reported.
- [ ] Claim scope follows geometry evidence.
- [ ] Gate I candidate config is immutable.

## Handoff to next plan

Plan 09 may assume complete Phase I uncertainty, geometry, representation, cross-view, and overhead evidence with immutable candidate configs. It must make a gate decision rather than continue tuning.
