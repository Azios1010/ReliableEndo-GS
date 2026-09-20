# Fresh patched Stage2 fixed-10k server run

This patch changes Gaussian scale to `0.01 * sigmoid(raw_scale)` and adds
observation and launch/provenance safeguards. It does not change the dataset
loader, split algorithm, preprocessing, masks, geometry, renderer kernels,
Stage1 loading method, loss weights, optimizer, scheduler, seed, or training
iteration/evaluation cadence. Stage1 60k is the fixed input artifact; this does
not newly freeze parameters inside the existing joint Stage2 optimizer.

## Transfer the complete patch

From the ReliableEndo-GS repository with its Python package installed:

```bash
python scripts/package_stage2_patch.py --output outputs/stage2-scale-sigmoid-10k.zip
```

The ZIP includes `stage2.patch`, `manifest.json` and this runbook, including all
new modules even when they are untracked in the upstream submodule. It excludes
datasets, checkpoints and previous runs. The builder refuses to overwrite an
existing ZIP. Ordinary push/pull of the parent repository alone does not transfer
dirty submodule contents.

Extract the ZIP into a new transfer directory. In an isolated upstream checkout
at the base commit recorded in `manifest.json`, first check, then apply:

```bash
git apply --check /path/to/transfer/stage2.patch
git apply /path/to/transfer/stage2.patch
```

Do not overwrite an existing server trainer/renderer/loader to force the patch
to apply. If the server has additional compatibility changes, preserve them and
review their integration; the control-source checks below must still pass.
The manifest records SHA-256 for the patch and each patched source file, with
UTF-8 universal-newline normalization for cross-platform source checks.

## Bind the old 10k control

Use artifacts preserved from the **old fresh 10k control**, never the completed
patched 3k run. The following paths/IDs must come from the server's existing run:

- `CONTROL_SOURCE_BACKUP`: original control's backed-up source directory.
- `CONTROL_CONFIG_JSON`: complete resolved `cfg.json` from that run.
- `CONTROL_SPLIT_JSON`: the ordered train and validation frame IDs actually used.
- `STAGE1_60K_CKPT`: the same provisionally frozen Stage1 checkpoint.
- `STAGE1_RECORD_JSON`: preserved Stage1 identity record containing
  `training_steps: 60000` and `checkpoint_sha256`.
- `CONTROL_RUN_ID`, and a new output path `CONTROL_MANIFEST`.

The split JSON has `dataset_id` (for example `dataset_3/keyframe_1`),
`train_frame_ids` and `validation_frame_ids`, each using the exact strings and
order in the old loader's selected metadata. Validation must contain exactly
422 unique IDs, disjoint from training. Do not generate a different split,
reorder IDs, use dataset_6, or fill unknown identities with guessed values.
If the old artifacts have another schema, transcribe their existing identities
without changing membership; retain the original evidence alongside the record.

From the patched upstream directory:

```bash
python -m lib.stage2_run \
  --source-root "$CONTROL_SOURCE_BACKUP" \
  --resolved-config "$CONTROL_CONFIG_JSON" \
  --split-manifest "$CONTROL_SPLIT_JSON" \
  --stage1-checkpoint "$STAGE1_60K_CKPT" \
  --stage1-record "$STAGE1_RECORD_JSON" \
  --control-run-id "$CONTROL_RUN_ID" \
  --output "$CONTROL_MANIFEST"
```

This command hashes the supplied identities and refuses a 3k/resumed control
or a source backup already using the patched activation. It does not infer
checkpoint training history or establish the scientific truth of supplied
records. If the records are missing, recover them from the preserved server
run before training; do not fabricate a manifest to bypass the checks.

## Launch the fresh run

Set `SCARED_STAGE2_DATA_ROOT` to the same development dataset/keyframe as the
control and `NEW_STAGE2_RUN_DIR` to a **nonexistent** directory. The data path is
explicitly supplied; loader behavior is unchanged. The launcher checks full
resolved scientific-config equality, checkpoint hash, split hash, protected
source hashes, and optimizer/schedule/loss/update/loading/seed signatures.
After constructing the existing loaders, it checks their actual ordered frame
membership against the control record before the first training step.

```bash
python train_stage2.py \
  --config config/stage2.yaml \
  --control-manifest "$CONTROL_MANIFEST" \
  --stage1-checkpoint "$STAGE1_60K_CKPT" \
  --data-root "$SCARED_STAGE2_DATA_ROOT" \
  --run-dir "$NEW_STAGE2_RUN_DIR"
```

Steps remain 10,000, `restore_ckpt` must be null, seed remains 1314, and OneCycle
retains the existing `num_steps + 100` construction. The completed 3k run is
never resumed. Reusing even an empty run directory fails; failed runs remain
available for diagnosis and a retry needs a new path.

`run_manifest.json` records resolved config/hash, control/checkpoint/split
identities, project/upstream Git state, source snapshot hashes and CUDA/hardware
metadata. `verified_split.json` exists only after membership verification.
`completion.json` records actual final steps and final checkpoint SHA-256.
The original source/config backup remains under `file/`.

## Diagnostics and final evaluation

Every 500 steps (including zero), log raw/activated scale quantiles, boundary
fractions, opacity distribution, native radii, and head gradient L2 norms after
AMP unscale and before the unchanged gradient clip. Metrics are detached and
do not consume random numbers or add a render pass.

- `render/visible_gaussian_fraction`: positive radii / submitted Gaussians.
- `render/nonbackground_pixel_fraction`: fraction of finite rendered pixels
  whose maximum RGB difference from background exceeds the recorded threshold
  (default `1e-6`). This is an RGB coverage proxy, **not alpha coverage**;
  background-colored splats cannot be distinguished from uncovered pixels.
- `render/nonfinite_pixel_fraction`: invalid rendered-pixel fraction.
- `render/ssim_loss`: `1 - SSIM`; `render/ssim`: SSIM itself.
- `disparity/...` and `render/...`: separate training/validation quantities.

Use the control's existing evaluator and exactly the matched 422-frame list
to evaluate the **final 10k checkpoint**. The preserved trainer cadence does
not automatically evaluate the final checkpoint at loop exit; completion
explicitly leaves `matched_final_validation` pending. Preserve the same metric
definitions/masks and report render and disparity results separately.

Real SCARED-C loading, actual Stage1 checkpoint compatibility, native CUDA/
Taichi/GS execution, GPU diagnostics and matched validation are
**NOT TESTED LOCALLY**. Do not promote the deterministic foundation before the
matched server evaluation is accepted. Dataset_6 remains untouched.

## Local handoff checks (2026-09-20)

- 16 Stage2 CPU/synthetic tests passed: scale mapping/autograd, detached
  diagnostics, config parsing, identity/protocol guards, no-overwrite behavior,
  provenance, and bundle application with all target source hashes checked.
- The expanded baseline selection produced 31 passes and one failure:
  `test_submodule_is_at_exact_clean_revision`. The pinned revision is unchanged,
  but this deliberately patched variant is not a clean baseline worktree.
  The safeguard was not weakened; this is not an all-green baseline suite.
- In-memory syntax checks passed for all nine patched upstream Python files;
  the standalone provenance helper imports and exposes its CLI without CUDA.
  Ruff passed for the new helpers, packaging code and tests; mypy passed for
  the packaging module. Both Git whitespace checks passed.
- Full trainer imports/native execution and real checkpoint/data compatibility
  remain **NOT TESTED LOCALLY**. Synthetic identity fixtures are not evidence
  that the server's actual control artifacts match.

## Exact patch file inventory

The transfer patch contains these paths under `third_party/endo_e2e_gs/`:

```text
config/stage2.yaml
config/stereo_config.py
gaussian_renderer/__init__.py
lib/GaussianRender.py
lib/gs_parm_network.py
lib/network.py
lib/scale_parameterization.py
lib/stage2_diagnostics.py
lib/stage2_run.py
train_stage2.py
```

Local packaging, tests and handoff documentation added/updated for this patch:

```text
src/reliable_endo_gs/baseline/stage2_bundle.py
scripts/package_stage2_patch.py
tests/baseline/conftest.py
tests/baseline/test_stage2_scale_parameterization.py
tests/baseline/test_stage2_run.py
docs/stage2_fixed10k_server.md
```

The earlier roadmap edits remain in `PLAN.md`, `docs/execution_pipeline.md`,
`plans/03_baseline_reproduction.md`,
`plans/04_phase1_uncertainty_proxies_and_oracle.md`, and
`plans/05_phase1_learned_uncertainty_and_calibration.md`; they are not part of
the upstream transfer patch. No existing experiment output is included or altered.
