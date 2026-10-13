# Larger-cohort selector preparation

## Experiment boundary

Reuse the approved parent 25% original-SCARED cohort: 5,738 TRAIN images
from 27 keyframes, plus its 112 diagnostic development images. Original
dataset roles remain train 1–7 / final test 8–9. No final-test data participates.

The student and repair artifact is pinned to checkpoint SHA-256
`01ee7a0d065127bb50f996e1d9247610e7826c6da54c9bbdcddc3a02679c32c3`.
Its masked repairs completed 168 updates on 56 images. The earlier parent
student completed three epochs on 5,738 images; this does **not** mean the
current masked repairs trained on that larger cohort.

This experiment freezes that exact student/action state and refits the gain
and risk heads. It tests whether more varied intervention labels help the
selector. It does not jointly retrain the student or repairs, pass the research
gates, or establish useful three-action routing. Previous negative results in
[selector_gain_risk_refit.md](selector_gain_risk_refit.md) remain evidence.

## Sealed internal TRAIN roles

`configs/routing/selector_refit_quarter.yaml` selects
`stratified_keyframe_hash`: within each training dataset, SHA-256 ordering
assigns one whole keyframe to validation, one to calibration, and the rest to
FIT. This policy uses whole keyframes rather than the legacy fraction controls.
The partition is sealed before GT labels or intervention outcomes are decoded.

| Internal role | Images | Keyframes |
| --- | ---: | ---: |
| FIT | 2,788 | 13 |
| Validation / stopping | 1,572 | 7 |
| Calibration | 1,378 | 7 |

Each role covers all seven training datasets. Development images are disjoint
from TRAIN images but share parent keyframes, so they support diagnosis rather
than a final generalization claim. Only FIT estimates normalization; validation
selects gain/risk stopping steps; calibration fits the final monotone gain and
risk corrections. Unsupported calibration remains unavailable. Existing
utility weights, physical metric guards and action thresholds remain fixed.
The parent student has already seen all 5,738 TRAIN images, including these
internal validation and calibration roles. These roles isolate fitting of the
new selector heads; final generalization still requires the reserved test data.

## External assets and ROI limitations

All image data, masks, source snapshots, worklists, contact sheets and experiment
outputs stay outside the repository. Resolve processed SCARED through
`RELIABLE_ENDO_DATA_ROOT`; no data relocation or duplication is needed.

The operator ROI protocol reviews 16 temporal RGB anchors per keyframe: 432
sampled frames in total. Revised boxes were visually reviewed. Intermediate
frames inherit the nearest reviewed anchor, using original numeric frame IDs
and a deterministic tie rule. These are coarse operator tissue ROIs, not
validated per-pixel tissue segmentation or guaranteed tool exclusion.
37 shared bool masks occupy 12,128,896 bytes for 5,850 indexed sample identities.
ROIs are selected without GT or action outcomes. GT file bytes are hashed for
provenance during preparation; GT values are not decoded at that stage.

The expanded cohort also changes ROI coverage. Original and refitted selectors
must therefore be compared on the **same new ROIs**, checkpoint, actions and
cost calibration. Cross-run metric changes cannot be attributed solely to
sample count.

Preparation seals full member SHA-256 hashes and size/mtime metadata. Resume
checks the immutable snapshot identity and current file stats; it does not
recompute every medical-data hash. RGB/calibration preflight decodes both
images and uses the canonical A00 camera contract. Reuse requires a previous
completed preflight with identical source/cohort/mask/partition bindings and an
unchanged source snapshot. Mask contents are rehashed at consumption.

## Execution and recovery

The machine-local preparation directory contains `run_selector_quarter.sh`,
`preparation-inputs.json`, `assets/`, check logs and `refit/readiness.json`.
The launcher uses physical GPU0, one visible GPU, FP32, the native renderer,
bounded CPU threads, a process lock, unique console logs and an exit-code file.

```bash
# Set PREP_ROOT to the external preparation directory reported by the agent.
bash "$PREP_ROOT/run_selector_quarter.sh" --prepare-costs
# Run only after preparation has passed; not automatically executed by preparation.
bash "$PREP_ROOT/run_selector_quarter.sh" --run
```

`--prepare-costs` validates the full deployment cohort, measures GT-free TRAIN
action-and-render cost on the first and last frame of each of 27 keyframes,
uses one warmup and three measured repeats, writes an atomic dedicated cost
artifact and stops. It executes neither GT outcome collection nor training.
Timings include synchronization/debug runtime settings and are not paper FPS.

`--run` collects real STOP/STEREO/GAUSSIAN outcomes in 64-image shards, fits the
selector with separate stopping and calibration roles, runs deployment smoke,
and compares original/refitted selectors on diagnostic development. Completed
shards are verified and reused; interrupted shards are archived. Unknown labels
stay unknown, including entirely unknown shards. The launcher requests explicit
resume for existing outputs. Source/checkpoint/config/runtime mismatches fail
instead of mixing runs. Changed code or assets require a new output experiment.

The earlier 56-image collection measured about 2.19 seconds per sample. A linear
estimate for 5,738 images is approximately 3.5 hours for outcome collection
alone; this is an estimate, not a measured larger-cohort runtime or guarantee.

## Validation evidence

Machine-local readiness and logs record the actual checks, source identities,
cost calibration and completion state. Preparation does not mark outcome
collection, selector fitting, joint training or final test as completed.

Tests cover legacy config identity, larger declared bounds, keyframe separation,
temporal ROI assignment, RGB-only preflight, shared masks, immutable source
checks, interrupted ledger tails, canonical shard provenance, entirely unknown
shards, partial-shard preservation, tamper rejection and reuse without repeated
interventions. Failed initial checks and interrupted preparation logs are kept.

### Completed preparation evidence

The full deployment preflight passed all 5,738 TRAIN and 112 development
images. Dedicated native cost calibration passed 54/54 GT-free TRAIN cases on
physical GPU0 and exited with code 0. With `CUDA_LAUNCH_BLOCKING=1`, the recorded
maximum action-and-render estimates are STOP 0 ms, STEREO 128.3951 ms and
GAUSSIAN 56.1932 ms, within the unchanged 1,000 ms action budget. The 54 cases
provide 162 measured repeats per action. These are incremental diagnostic
costs, not complete pipeline latency or an FPS benchmark.

61 focused regression tests passed on the runtime environment with GPU0
visible; 13 preparation tests also passed with GPUs hidden. Mypy passed the
nine affected source modules; Ruff checks and formatting passed ten files;
the external launcher passed `bash -n` and the working diff passed
`git diff --check`. The full CI suite was not run.

| Artifact | SHA-256 |
| --- | --- |
| Current first-party Python source tree | `a189afa9041dfa8d629c09e2aee75096f8cd2f73baef33cdc780da8dbbda86b2` |
| Refit YAML bytes | `012e503a349aec39b3d4055acd7beef30660bcdf0ddeaeeb98e79ba80aa99bb4` |
| Resolved refit configuration | `29abdf14d380a1f8d6de0cbd944a05cb3048bbf72ba3224a7497b4e26b4a8f16` |
| Worklist | `7ac5255051e4c49967e2e76d5ae7af5c5f4cd6c85928d9905b3f14e832df41f9` |
| Internal partition | `ddadbea0d6f75f87487db6740125c4e4142147c41e39902792231f363f135fb9` |
| RGB ROI policy | `149df8f8568421347db90e349de79527622aa10eae2c7a33e4bb309f53295062` |
| Target-mask index | `0b9f0ae5410d025e14505e29b2ff4f3464873d4ab3c7cd6071249e2ffad0b424` |
| Processed source snapshot, 29,277 members | `7b31bd693d7a9c32b78c093096eed1ec25e8e5c3ff4f85c4dc6d35d16ab36de6` |
| Dedicated TRAIN cost artifact | `25182b44fb369d41a68ea58edb4bbc1545bcb677df84fb9e8bdee4210dc5f54e` |

The exact ROI generator was archived with SHA-256
`9abb9b318e1b5cf452cf06513e7400df5f9773f8346f200441b78ec4aabd7672`.
After generation, metadata/progress variable names and the docstring were
corrected, and the independent source-stat verifier was given bounded I/O
workers. The current module hash is
`b45d000c48c9d3aa8314dd729875938b427d281a66651c83a1a821bfa2130b87`.
Generated masks and their original producer seal were preserved; both source
versions are identified in the external `source-identities.json`.

Preparation stops here. No larger-cohort outcome collection, selector fitting,
joint training, teacher inference/cache generation, development metric
comparison or final test was executed. The previous selector failure remains
unresolved scientifically until the larger experiment produces evidence.
