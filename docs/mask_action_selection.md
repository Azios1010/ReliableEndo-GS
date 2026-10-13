# Target masks for experimental action selection

This is an opt-in implementation of the user-approved mask direction. Gate II
has not passed. The earlier fixed-grid native outcomes are historical evidence;
they do not establish utility, safety or latency for these new masks.

## Executable path

```text
deployment stereo -> student disparity + shared matching evidence
 -> canonical A03 / head / base Gaussian field / two base renders
 -> external deployment target ROI intersected with predicted/sensor support
 -> cheap stereo and render proposal scores -> bounded sparse candidates
 -> small balanced gain/harm selector -> STOP / stereo / Gaussian repair
 -> canonical composition and two renders
```

The target ROI identifies the object/surface to reconstruct. It is separate
from sensor support, GT-valid pixels and the candidate repair mask. This pass
adds an explicit input contract; it does not train a foreground segmentation
model or assume that the student predicts foreground. An externally predicted
ROI must come from deployment RGB and have a recorded producer identity.
The index declaration alone cannot prove the producer did not use GT.

The student computes its feature/correlation tensors once per reconstruction.
Entropy, top-two cost margin and feature energy are detached summaries of that
same pass. They are **proxies**, not calibrated uncertainty or action risk.
Existing local/global context also includes warp residuals, both render
residuals, Gaussian opacity/scales/normals and available visibility diagnostics.
Whether these inputs suffice for a useful selector is still an empirical question.

## Sparse candidates and writes

`configs/routing/target_masks.yaml` declares normalization scales, score
threshold and hard limits. The provided values are experimental settings, not
thresholds fitted on real mask interventions.

- Proposal scores use pre-action RGB/disparity/render evidence, never GT.
- Eight-connected coarse occupancy groups candidates across old grid boundaries.
  Coarse cells are only grouping hints; writes contain actual above-threshold
  target pixels. Good pixels, holes and background are not filled.
- `core` is a bounding box; `MaskRegion.mask()` is the actual write permission.
  `halo` authorizes context reads only.
- Ranking bounds aggregate candidate pixel and primitive counts. Oversized
  halo boxes are rejected. Masks are disjoint across components.
- Source indices map pixels to existing Gaussian slots. Every out-of-mask
  disparity/primitive attribute remains unchanged. Gaussian footprints can
  still affect rendered pixels outside the ROI; collateral losses measure this.
- No eligible target candidates means STOP in inference. Joint learning then
  updates only student/head, records `mask_joint_skipped`, and creates no fake
  action labels. Missing/corrupt ROI files fail explicitly.

These are candidate budgets, not latency guarantees. Selected inference still
executes at most one candidate/action. Joint intervention collection evaluates
STOP and both repairs on all bounded candidates, from the same pre-update state.

## Learning and evidence protocols

Mask mode requires `intervention_selection.v2`: fresh local/global utility,
outside-region collateral measurements, harm supervision, ranking and bounded
gradient balancing. The existing balanced objective uses RGB/warp proxies; it
does **not** replace the independent metric-GT/render tradeoff benchmark.
Do not interpret proxy harm as calibrated metric failure probability.

All trainable student/head/repair/selector owners stay as declared in the
authorized joint schedule. Action labels are recomputed using actual repair
implementations. The selector's inputs/labels are detached; discrete action
choice does not send a gradient into the student. Base and repair objectives
train those parameters separately within the same joint update.

`evidence` controls proposal-source ablations. `feature_evidence` independently
removes stereo or Gaussian/render source columns from the action selector.
Keep proposals/data/actions fixed when testing feature sufficiency. Excluded
columns are zero and fitted inactive by TRAIN-only normalization. Missing
matching and visibility retain their existing availability indicators. These
ablations do not omit the base model computation or certify speed savings.

Mask checkpoints use `student_joint_mask_intervention_checkpoint.v1` and pin
the proposal/feature configuration. Resume also pins the ROI index SHA-256.
Grid selector checkpoints cannot resume or deploy as mask selectors. Model-only
warm starts are explicit new experiments; repair/selector/optimizer states start
fresh. Inference may supply a different external ROI index (e.g. held-out data),
but must use the same mask configuration and records the new index identity.

## External ROI index

### Remediation smoke and recovery

The opt-in `configs/routing/intervention_remediation.yaml` retains 32 TRAIN
snapshots of normalization burn-in. Student/head and eligible repairs update
during burn-in; selector weights begin updating after the transform is fitted.
Checkpoints record separate `optimizer_update_counts` for model, actions, and
selector. Loading checks each count against its Adam state and the burn-in
snapshot count before changing live weights. Historical non-burn-in checkpoints
without this field retain their original strict step checks. Burn-in checkpoints
without the counts are rejected. Selected deployment requires fitted statistics
and an actually trained selector.

The bounded pair `configs/training/intervention_remediation_smoke.yaml` and
`configs/routing/intervention_remediation_smoke.yaml` performs one student update
and three joint updates: up to two normalization snapshots followed by a real
selector update. This is four updates of one declared training sample, not an
epoch or a quality claim. Smoke checks named parameters rather than normalization
buffers when reporting learning. The full 32-snapshot config is deliberately
rejected by bounded smoke; its scientific configuration is unchanged.

Cost calibration stores the PyTorch version as an ordinary string, so artifacts
from captured runtime metadata can round-trip through weights-only loading.
Deployment cost measurements still require the exact frozen action/runtime/grid
binding. CPU no-op serialization tests do not measure real GPU action costs.

Source identity changes with these fixes. Use explicit model-only initialization
for a new experiment with an old compatible student checkpoint; strict resume
and deployment compatibility are unchanged. Do not relabel historical artifacts.

For a configured runtime, external data, manifest, KD cache and deployment ROI:

```bash
python -m reliable_endo_gs.cli.student smoke \
  --config configs/training/intervention_remediation_smoke.yaml \
  --intervention-config configs/routing/intervention_remediation_smoke.yaml \
  --mask-config configs/routing/target_masks.yaml \
  --target-mask-index "$DEPLOYMENT_ROI_INDEX" \
  --initialize-model-from "$INITIAL_CHECKPOINT" \
  --data-root "$DATA_ROOT" --manifest "$TRAIN_MANIFEST" \
  --cache-root "$KD_CACHE" --sample-id "$TRAIN_SAMPLE" \
  --output-dir "$FRESH_OUTPUT_DIR"
```

The caller explicitly selects the allocated physical GPU before invoking Python.
This command requires native GPU verification; CPU tests cannot establish it.

Both index and arrays must live outside this repository. Each `.npy` is exactly
bool `[H,W]` on the **model output grid**, with no implicit resize. Example
schema below uses placeholders; no real medical mask is provided or fabricated.

```json
{
  "schema": "deployment_target_masks.v1",
  "samples": {
    "dataset_1/keyframe_1/frame_data000000": {
      "path": "roi/frame_data000000.npy",
      "sha256": "<full sha256 of the exact .npy bytes>",
      "output_hw": [512, 640],
      "source": "input_roi",
      "source_identity": "<ROI producer/config/preprocessing identity>"
    }
  }
}
```

Allowed sources: `input_roi`, `rgb_prediction`. Reject GT-valid/GT-error sources.
Grid dimensions, bytes/hash and boolean dtype are checked at consumption.
Index coverage, declared grids and file presence are checked before model and
teacher construction. No SCARED source/preprocessed files or splits are changed.

Add these arguments to the existing explicit student smoke/train command:

```bash
--intervention-config configs/routing/intervention_balanced.yaml \
--mask-config configs/routing/target_masks.yaml \
--target-mask-index "$DEPLOYMENT_ROI_INDEX"
```

Keep the actual external data root, manifest, joint/KD configuration and output
directory explicit. Bounded native smoke must have an eligible mask candidate;
the initial zero-action parity check now exercises that sparse candidate.
No real smoke/train command is ready until the deployment ROI producer/index
is available and validated. Do not generate ROI from metric GT to make it run.

## Execution cost and remaining verification

Gaussian repair and the reference head after stereo repair gather selected
pointwise contexts, evaluate only those slots, then scatter residuals. Masked
stereo matching convolves an expanded left ROI and a corresponding horizontal
right strip with the exact two-convolution receptive radius. The right strip
retains full width for disparity search; canonical global sampling/pixel units
are preserved. CPU tests compare values and gradients against full features,
including image borders.

A03 geometry construction, field composition, parts of balanced feature
extraction and both renders still operate on the full grid. Proposals also
compute cheap full-grid residual maps. Full sparse geometry/rasterization is
not implemented. Measure proposal, feature, decision, actual action, total
policy latency and VRAM before making efficiency claims.

Next native evidence must evaluate real-mask counterfactuals on fixed TRAIN and
held-out development samples: metric depth MAE/RMSE/p95, left/right render
quality, outside-mask damage, oracle action diversity, risk calibration and
matched-cost comparisons against STOP, Gaussian-only, stereo-only and fixed
repair. No threshold may be selected using final-test GT. This implementation
and CPU gradient smoke do not prove that stereo repair has useful real gain.

## Bounded native verification pilot

`python -m reliable_endo_gs.cli.mask_validation --help` exposes an opt-in pilot.
It validates a SHA-bound worklist within the existing original-SCARED TRAIN and
development manifests, covering datasets 1–7. A worklist does not replace the
role/split contract. Load TRAIN through its original parent manifest so its KD
cache keys remain unchanged. Missing/corrupt targets fail; cache is read-only
and the teacher model is never loaded or invoked.

The external worklist uses schema `mask_validation_worklist.v1`, with unique
`training`/`development` sample-ID lists, parent `training_manifest_hash`,
`development_manifest_hash`, `role_hash`, and `target_mask_index_sha256`.
The pilot accepts at most 112 TRAIN and 56 development frames, and 1–5 epochs.
The prepared diagnostic uses 56 TRAIN, 28 development, and three joint epochs.
All trainable student/head/repair/selector groups learn; the external KD teacher
remains an artifact provider. Model-only initialization preserves the fitted
student/head but initializes fresh mask repairs, selector and Adam states.

CPU preflight validates the exact cache, ROI hashes/grids, configurations and
source files without loading GT. Native execution first measures the initial
STOP baseline, trains the bounded worklist, then evaluates the final STOP,
fixed stereo/Gaussian, predicted-region single-action baselines, selector and
proxy oracle on the same development frames. Evaluation restores RNG and never
updates weights or normalization. GT is loaded only after proposals, learned
decisions and the shared real counterfactual actions execute.

`metric_diagnostics` independently reports fixed-support depth MAE/RMSE/p95,
both-view PSNR/SSIM, coverage and local/outside RGB guards. Missing metrics stay
unknown. It counts safe component-improvement opportunities and missed choices
under the same estimated action budget. This is not a normalized metric-utility
oracle: no utility scales are fitted on development data. The separately named
proxy oracle still uses RGB/warp labels. Guard failures and unknown outcomes
remain in the report; the pilot cannot certify Gate II.

Manual RGB input ROIs must be declared as such, with producer/review provenance.
They are approximate operator inputs, not a learned foreground model or GT
segmentation. Same-keyframe frame-disjoint development supports debugging only;
it cannot establish independent-sequence generalization or paper performance.
No latency/FPS claim follows from this pilot's single-run policy timings.
