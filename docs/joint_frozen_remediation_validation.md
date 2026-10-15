# Joint/frozen implementation remediation (2026-10-15)

This pass repairs the existing opt-in experiment. It does not pass Gate I or
Gate II, certify a Phase I artifact, or promote a routing policy.

## Implemented boundaries

1. Bounded student smoke accepts zero or one student-only epoch. In frozen
   mode, student/head parameters, all model buffers and model Adam must remain
   exactly unchanged; both repairs and the selector must update. Joint smoke
   still requires all scheduled parameter groups to update.
2. A new frozen CLI run requires `--initialize-model-from`. Its compatible
   student checkpoint must contain positive model optimizer updates and matching
   model Adam ownership, moments and steps. An untrained initialization checkpoint
   is rejected. The record includes its resolved path, SHA-256, source identity
   and `checkpoint_model_only` initialization mode. These checks establish a
   trained compatible model; they do not establish scientific artifact approval.
3. Resume restores only `model_initialization` from its own checkpoint, validates
   its structure and digest, and preserves it in subsequent outputs. It does not
   reapply initialization weights or require the original file to remain at its
   old path. Current source/config/data identity checks remain strict.
4. `repair_max_disparity_px` and `cost_weight_per_ms` survive schema v1/v2
   serialization and affect identity whenever nonneutral. Historical neutral
   hashes remain unchanged; invalid types/ranges/nonfinite values are rejected.
5. Executable action availability is distinct from GT label availability.
   Harnesses and repair losses skip ineligible actions. Logs use null timings
   for skipped repairs, and cost history excludes incomplete execution rows.
   Missing GT leaves actual execution/timing intact while excluding labels.
   Per-repair Adam counts are saved and checked separately; unavailable repair
   parameters and their existing Adam moments remain unchanged.
6. Selector refit supports signed expected gain and local/global/any risks.
   CAL fitting uses an affine expected-gain transform and independently fitted
   event probabilities, with class/keyframe support recorded per event. Signed
   sign/benefit/cost components remain explicitly raw and reconstruct the raw
   expected gain. Unsupported events retain an explicitly marked identity
   transform; unsupported any-risk calibration abstains. Independent marginal
   calibration does not guarantee a consistent joint event distribution.
   Evaluation receives all event/component tensors and distinguishes raw and
   calibrated outputs, unavailable actions and unknown GT.
7. The approved `PINNED_COMMIT`, HEAD gitlink, index gitlink and local checkout
   agree on `186fa2b4a2159b28393492f6df1aa444b54391a8`. Submodule sync/update
   completed. The `170659ad...` mismatch in the historical handoff refers to a
   different server checkout; that checkout was not accessed in this pass.

The calibration support above supersedes the earlier implementation-limit
statements in `signed_gain_selector.md` and `mask_token_selector.md`. Historical
native results in those reports retain their original evidence limits.

## Validation and remaining evidence

Regression tests cover frozen smoke with real CPU optimizers and an explicit
analytic renderer substitute, checkpoint ancestry/strict resume, config
roundtrips and hashes, action availability/Adam ownership, event calibration,
artifact roundtrips and read-only evaluation.

Validation record:

- The first expanded run completed with 643 passed and 3 failed. Two failures
  came from Windows newline translation in hash-bearing test fixtures; the
  third came from the initially untracked upstream Python cache. The fixtures
  now write the exact UTF-8 bytes they hash. The cache was preserved outside
  the repository, without changing pinned upstream source or weakening checks.
- A later expanded CPU run completed with 649 passed and one failure in the
  already-collected, old shard fixture. After its correction, all 20 related
  cohort/refit/upstream-correlation tests passed. The remaining scientific
  implementation cases in the expanded run passed, including the full
  optimizer/resume matrix. This is full-run plus focused-rerun evidence; no
  single all-green expanded invocation is claimed.
- Focused availability/context/legacy-action validation passed 77 tests;
  signed/event calibration validation passed 36 tests. Full mypy passed all
  232 source files; Ruff passed `src` and `tests`, formatting passed changed
  Python files, and `git diff --check` passed. The unmodified tree has some
  pre-existing mixed-line-ending formatting differences outside this patch.

`python -m tests.diagnostics.synthetic_risk_probe --output <new-external-directory>`
runs a matched analytic 5x5 joint/frozen probe. It uses one trained synthetic
model initialization, identical initial action/selector states, fixed seeds,
48 training snapshots per mode and 12 evaluation perturbations. Normalization
fits after 32 snapshots and remains fixed. The generated JSON records full
configuration, checkpoint/source/fixture hashes, hardware, risk AUROC/Brier/ECE,
feature clipping, selected actual harm and raw gain components. Source identity
must be unchanged from start to finish. No threshold or utility weight is tuned.

This probe is synthetic implementation evidence, not SCARED risk-quality or
runtime evidence. Missing harm classes produce null AUROC; zero selected repairs
produce an undefined selected-repair harm rate. Event calibration support is
tested separately on disjoint synthetic CAL keyframes; the matched probe
reports raw risk probabilities.

The final source-stable probe completed successfully. Its combined
source/config/fixture SHA-256 was
`35b892e06735ccbafe67ee27bb0460a0cd9f82c674197da127252b44bcfaddb4`
at both start and end; the generated report SHA-256 was
`d03233960e56fe0c430ec01f7ec735ed43c9b0218ee8fb3fcef05bb0193f48f3`.
Both arms started with identical model/action/selector hashes. The frozen
model and Adam remained unchanged, while repairs and selector updated.

| Synthetic evaluation, 12 perturbations | Joint | Frozen |
| --- | ---: | ---: |
| Selected STOP / stereo / Gaussian | 8 / 4 / 0 | 12 / 0 / 0 |
| Stereo any-harm AUROC | 0.53125 | 0.59259 |
| Stereo any-harm Brier | 0.26351 | 0.38696 |
| Stereo any-harm ECE, 10 fixed bins | 0.19862 | 0.45331 |
| Gaussian any-harm Brier | 0.05250 | 0.43533 |
| Clipped feature entries | 94 / 1020 (9.22%) | 0 / 1020 |
| Harmful selected repairs / selected repairs | 1 / 4 | 0 / 0 (undefined rate) |
| Harmful decisions / all decisions | 1 / 12 | 0 / 12 |

No Gaussian harm positives occurred, so its harm AUROC is undefined in both
arms. Local/global harm labels coincide on this tiny fixture. These negative
and incomplete results do not establish useful routing safety or prefer one
arm for deployment. Normalization remained exactly fixed after snapshot 32.

Native joint/frozen verification is blocked in this workspace: no external
data/checkpoint/FIT/VAL/ROI/KD assets are configured, and the native
`diff_gaussian_rasterization` and `simple_knn` extensions are absent. A GPU is
present, but it does not supply those assets or compiled renderer. The asset
locations/server were requested; no native run or final-test evaluation is
claimed. Use the bounded command protocol in `mask_token_selector.md` after
resolving those assets, preserving the same initialization/worklist and policy
in both arms. Evaluate discrimination, calibration, clipping and actual harm;
retain negative evidence and assess the fixed 32-snapshot transform before
changing any normalization policy.
