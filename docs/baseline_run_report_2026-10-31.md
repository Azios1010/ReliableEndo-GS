# Extra feed-forward baseline run report — 2026-10-31

**Status: Task A complete; Task B CPU preparation complete, real-data evaluation
blocked by unresolved depth bounds. No extra-baseline benchmark numbers exist.**
The pending question is: **what fixed near/far optical-axis depths, in metres,
should MVSplat and DepthSplat use on SCARED?** Their checked RE10K configurations
use 1–100 and 0.5–100 respectively, in the source pose coordinates. Applying those
values to A00 metre cameras is not an established conversion. The task explicitly
says to stop on ambiguous units/conventions. No bounds were inferred from GT or
validation performance. The 0.01–1 m values used in synthetic CPU tests are test
fixture values only, not a deployment choice.

The shortlist was written in [the feasibility report](baseline_feasibility_2026-10-31.md)
before implementation/inference. It contains verified papers, official code and
license findings, training requirements, alternatives and unavailable releases.
Endo-E2E-GS, KD, GT-supervised student and FoundationStereo were not rerun.
**Endo-E2E-GS native parity with its paper remains unverified.** No paper claim is
accepted or rejected by this work.

## Implemented

- Added [posed_gaussians.py](../src/reliable_endo_gs/baseline/posed_gaussians.py):
  `forward(sample) -> FrozenModelPrediction`, shared by the two native encoders.
  It supplies two RGB views at their actual resolution, converts A00 integer-center
  intrinsics to upstream normalized half-pixel coordinates, and uses
  `c2w_R = inverse(T_R_from_L)` with the left camera as world. Translation remains
  in metres. Internal optical-axis depth is converted using canonical
  `depth_to_disparity`, then the left/right principal-point difference is added
  to produce `d = u_left - u_right` in input pixels.
- Added [posed_gaussians_native.py](../src/reliable_endo_gs/baseline/posed_gaussians_native.py):
  lazy CPU loading with upstream Git/source checks and exact state key, shape,
  dtype and finiteness checks. Both official native checkpoints load successfully.
  Each upstream owns a package called `src`, so models run in separate processes
  from their respective upstream working directories. No `sys.path` mutation.
- Added [posed_benchmark.py](../src/reliable_endo_gs/baseline/posed_benchmark.py) and
  the thin [entry point](../scripts/benchmark_posed_gaussians.py). The runner admits
  only the original ordered 56 `val_7` IDs and 14 controls per dataset 1/2/3/6,
  selected exactly by the existing midpoint-quantile/exclusion rule. Prepared
  worklists were generated from manifests only; no RGB, disparity or other frame
  assets were loaded. Missing near/far values stop the runner before data/GPU work.
- Prepared two immutable config files with **null near/far**. Provenance includes
  code commit plus complete uncommitted source snapshot/hash, checkpoint hashes,
  native commit, seed 20261031, ordered sample IDs, metric source identities,
  DINOv2 source/assets and compiled rasterizer hashes. The runner records physical
  GPU identity, load averages, wall time and peak allocation/reservation. It checks
  weights/buffers and source identity after evaluation and refuses output reuse.
- The prepared measurement path uses the existing `evaluate_geometry`,
  `frame_metrics`, and LPIPS AlexNet v0.1. The geometry, rendering metrics and LPIPS
  files are byte-identical to the existing frozen benchmark source. The runner
  reuses `kdattr_common.shifted_batch` and `shift_stats`, with shifts
  `{-24,-16,-8,0,8,16,24}` and width 608. It preserves raw rows for later slope
  analysis. The existing `shift_eval.py` only recognizes the three existing actors;
  it was not modified or misrepresented as accepting these new models.

The benchmark output schema is `sample_id`, `metrics` (each with `value`,
`valid_count`, `total_count`, `definition`), and `quality_pass_forward_ms`.
The last field is deliberately **null** on shared hardware. Per-frame tensor files
will store disparity, native left render and native right render. No output files
claiming measurements were made during preparation.

## Model identities and actual execution

All generated research/setup artifacts reside under:

`/mnt/d/3DGS_runs/reliableendo-gs/baselines-20261031/`

| Item | MVSplat RE10K | DepthSplat-small RE10K GS |
| --- | --- | --- |
| Official repository | `https://github.com/donydchen/mvsplat` | `https://github.com/cvg/depthsplat` |
| Upstream commit | `01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01` | `2dad25a7b9ba7c5537aba08463ef88d8511a06be` |
| Checkpoint file under artifact root | `weights/mvsplat-re10k.ckpt` | `weights/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth` |
| SHA-256 | `83d0d9eaa753fa4a1f925288dc1f90b8c3297fad0ab0f6ed1f11a1c5946da25a` | `cfeab6b10077ad9bd4097b82b029737221bfbbbd3b4dfca5fc0feaf860276506` |
| Origin | Official README's Drive folder; file ID `1Y5OAJN9AX5yMjkbyxag5OnwacdjPnl-N` | Official model-zoo HF `haofeixu/depthsplat` link; SHA prefix matches filename |
| Actual work | Exact CPU state load; actual encoder on random 64×64 CPU pair | Exact CPU state load; actual encoder on random 64×64 CPU pair |
| Loaded parameter count | 11,955,734 | 38,346,416 (model zoo describes approximately 37M) |
| Real frames / GPU renders | 0 / 0 | 0 / 0 |
| Real-data wall time / peak GPU memory | Not measured | Not measured |
| CPU smoke wall time / peak RSS (not latency) | 50.973 s / 943,423,488 bytes | 47.448 s / 1,091,678,208 bytes |
| `val_7` per-frame results path | Not created: blocked | Not created: blocked |
| `train_1236` per-frame results path | Not created: blocked | Not created: blocked |

Local first-party Git HEAD was
`96ffbc869f3362d4ed8905498d60c49aea560d1f`. The adapters are **uncommitted**; HEAD
alone does not identify them. No commit or push was run. The working tree already
contained unrelated student-training edits when work began; those were left alone.
The final preparation directory is `preparation-20261031T065247Z/`; its
`source-files.json` and each config's `source_sha256` identify the executable source.
Final source SHA-256: `aed9fe75f8a13ca5c3574c8db1c1293215500deb640b435d64547244a82fcd0f`.
The earlier `preparation-20261031T064842Z/` is preserved as superseded preparation,
not a failed benchmark or a competing configuration.

| Requested metric, on each of the two allowed cohorts | MVSplat | DepthSplat-small |
| --- | --- | --- |
| Mean EPE (px) | Unmeasured | Unmeasured |
| Stereo depth MAE / RMSE (mm) | Unmeasured | Unmeasured |
| Left / right PSNR (dB) | Unmeasured | Unmeasured |
| Left / right SSIM | Unmeasured | Unmeasured |
| Left / right LPIPS Alex v0.1 | Unmeasured | Unmeasured |
| Disparity-shift slopes | Unmeasured | Unmeasured |
| Latency / FPS | Not measured on shared GPU | Not measured on shared GPU |

## Tested and failure evidence

- **32 CPU tests passed**, including 11 new adapter/selection tests, existing import
  boundaries, adapter contracts, provenance/checkpoint contracts and frozen paired
  benchmark tests. Final log: `research/pytest-contracts-final.log` (39.25 s).
  Synthetic tests cover transform direction, half-pixel intrinsics, principal-point
  offset, crop-shift sign, native Gaussian/covariance preservation, two rendered
  output slots, GT rejection, bound validity, dirty upstream rejection, fixed
  sample selection and forbidden-dataset rejection using synthetic IDs.
- Actual **MVSplat and DepthSplat native encoders** passed finite-disparity shape
  checks on random CPU 64×64 inputs. These tests did not call a native renderer,
  use SCARED, compute benchmark quality, or establish 512×640 feasibility.
  Logs: `research/mvsplat-native-synthetic.log`,
  `research/depthsplat-native-synthetic.log`; machine-readable status:
  `research/native-synthetic-status.json`. Final frozen-snapshot smoke artifacts:
  `research/mvsplat-cpu-final-attempt2.json` and
  `research/depthsplat-cpu-final-attempt2.json`. Their CPU wall times include
  construction/checkpoint loading and the encoder call, exclude interpreter/import
  startup, and are not inference latency measurements. Peak RSS includes imports.
- The prepared CLI was also tested with null bounds: it rejected the config before
  creating output or using data/GPU. Log: `research/runner-null-bounds-test.log`.
- Ruff lint and formatting checks pass on all five added Python files.
  `git diff --check` passes. `third_party/endo_e2e_gs` remains clean.
- Official shared rasterizer built successfully for SM 12.0/CUDA 12.8, commit
  `1250c420ebb945f0dce9945086e22faab9157c92`, GLM submodule
  `5c46b9c07008ae65cb81ab79cd677ecc1934b903`.
  **First build failed** on missing `std::uintptr_t`, `uint32_t`, `uint64_t`.
  The retry added `NVCC_PREPEND_FLAGS=--pre-include=cstdint`; no source patch.
  Both logs are retained: `research/rasterizer-build.log` and
  `research/rasterizer-build-attempt2.log`. Successful wheel SHA-256:
  `9b38b2f2bf3aa512c6225b99889e5b68c2daa9f14dcea6975aa717fea32f3077`.
- **Broad baseline regression suite did not pass.** The corrected broad invocation
  aborted in existing
  `test_a00_model.py::test_metric_centers_use_principal_offset_and_integer_pixels_with_gradient`
  with `free(): double free detected in tcache 2` during CPU autograd.
  See `research/pytest-regression-attempt2.log`. Cause not established; no existing
  scientific source/test was changed to hide it. Remaining tests in that invocation
  were not executed. The separate passing subset is not a full-suite pass.
- Two earlier test-launch failures are also retained: pytest file-descriptor
  capture failed on D: before running tests (`research/pytest-posed.log`; resolved
  with `--capture=sys`), and an incorrect requested test filename collected no tests
  (`research/pytest-regression.log`; corrected next invocation).
- Early MVSplat CPU preflights failed on missing PyTorch Lightning, then on an
  omitted two-context-view global configuration. Installed dependencies in the
  new environment and fixed the new loader; successful strict load is recorded in
  `research/mvsplat-cpu-preflight-attempt4.log`. DepthSplat strict load:
  `research/depthsplat-cpu-preflight.log`. Earlier failed logs are retained.
  Final smoke script launches initially failed to resolve upstream `src` because
  Python put the script directory on its search path. Executing the same script
  via stdin from the upstream cwd resolved this; failed `*-cpu-final.log` files
  and successful `*-cpu-final-attempt2.log` files are both preserved. The recommended
  benchmark launch uses `python -m` from the upstream cwd.
- `research/environment-check.txt`: **No broken requirements found.** An initial
  inherited TensorBoard/protobuf conflict was resolved with a compatible version
  installed only in the new environment. Full package record:
  `research/environment-freeze.txt`.

## Commands and environment

`R` below is the artifact root above. The new D: venv uses a **read-only** view of
`endo_e2e_claude`'s existing site packages through `--system-site-packages`; it does
not copy or install anything into that environment. This inheritance is a disclosed
environment limitation, recorded by the complete freeze. PyTorch is 2.7.1+cu128,
torchvision 0.22.1+cu128, Python 3.10. Native project recommendations used older
PyTorch/CUDA; compatibility here is established only for the stated CPU checks and
compilation. xFormers is absent; native DINOv2 uses its warned PyTorch fallback.

```bash
export R=/mnt/d/3DGS_runs/reliableendo-gs/baselines-20261031
export PIP_CACHE_DIR="$R/cache/pip" HF_HOME="$R/cache/huggingface"
export TORCH_HOME="$R/cache/torch" XDG_CACHE_HOME="$R/cache"
export TMPDIR="$R/tmp" MPLCONFIGDIR="$R/cache/matplotlib"
export TORCH_EXTENSIONS_DIR="$R/cache/extensions" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/d/workspace/huann/reliableendo-gs/src
/home/bkai/miniconda3/envs/endo_e2e_claude/bin/python -m venv --system-site-packages "$R/env"
"$R/env/bin/python" -m pip install hydra-core==1.3.2 dacite==1.8.1 beartype==0.18.5 e3nn==0.5.1 timm==1.0.22 colorspacious==1.1.2 plyfile==1.1 svg.py==1.10.0 gdown==5.2.0
"$R/env/bin/python" -m pip install pytorch-lightning==2.4.0 wandb==0.17.7 moviepy==1.0.3 scikit-image==0.24.0 scikit-video==1.1.11
"$R/env/bin/python" -m pip install ruff==0.14.3
"$R/env/bin/python" -m pip install tensorboard==2.19.0
export TORCH_CUDA_ARCH_LIST=12.0 CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=2
export NVCC_PREPEND_FLAGS='--pre-include=cstdint'
"$R/env/bin/python" -m pip install --no-build-isolation --no-deps "$R/upstream/rasterizer"
```

Official shallow clones were made with `git clone --depth 1 URL DEST` for the six
repositories in `research/repositories-20261031T063215Z.json`; rasterizer uses
`git clone --recursive --depth 1`. No clone in this task modified the existing
third-party baseline. The actual checkpoint download commands were:

```bash
curl -L --fail --max-time 90 \
  'https://drive.usercontent.google.com/download?id=1Y5OAJN9AX5yMjkbyxag5OnwacdjPnl-N&export=download&confirm=t' \
  -o "$R/weights/mvsplat-re10k.ckpt"
# DepthSplat: urllib.request.urlopen streamed the following official URL to
# an exclusively-created file (153,661,476 bytes):
# https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth
```

Final passing contract invocation (from the first-party repository):

```bash
CUDA_VISIBLE_DEVICES='' "$R/env/bin/python" -m pytest \
  tests/baseline/test_posed_gaussians.py tests/baseline/test_import_boundary.py \
  tests/baseline/test_adapter_contract.py tests/baseline/test_provenance.py \
  tests/baseline/test_config_checkpoint.py tests/evaluation/test_paired_models.py \
  -q --capture=sys -p no:cacheprovider \
  --basetemp="$R/tmp/pytest-contracts-final"
```

The earlier broad regression command additionally requested all `tests/baseline`,
`tests/geometry/test_disparity.py`, `tests/geometry/test_analytic_cameras.py` and
`tests/evaluation/test_paired_models.py`, with CUDA hidden and the same capture/cache
flags. Its abort is reported above. The final native smoke commands below actually
ran (after setting the same D: cache variables). The script uses synthetic camera
fixture bounds only and never loads data. Output files must be new paths on replay.

```bash
export PYTHONPATH="$R/preparation-20261031T065247Z/source/src"
export CUDA_VISIBLE_DEVICES=''
cd "$R/upstream/mvsplat"
"$R/env/bin/python" - \
  --config "$R/preparation-20261031T065247Z/mvsplat.json" \
  --output "$R/research/mvsplat-cpu-final-attempt2.json" < "$R/research/cpu_preflight.py"
cd "$R/upstream/depthsplat"
"$R/env/bin/python" - \
  --config "$R/preparation-20261031T065247Z/depthsplat.json" \
  --output "$R/research/depthsplat-cpu-final-attempt2.json" < "$R/research/cpu_preflight.py"
```

## Deviations and limits

- **Zero-shot RE10K checkpoint transfer**, no SCARED training, no val-based model
  choice. Training compute/data are not matched to our models. No comparison of
  training efficiency is justified.
- Proposed data evaluation uses 512×640 rather than RE10K training resolution
  256×256. No input downsampling/resizing is silently substituted; native output
  shape mismatches fail. Near/far changes remain unresolved.
- Native two-source SH colors and native covariance renderer are retained. Both
  input views contribute appearance; these models do not share the student's
  left-only-color construction. Right-view scores would measure input-camera
  reconstruction, not held-out novel-view synthesis. No post-optimization.
- In the diagnostic `GaussianField`, `colors` stores flattened native SH, not an
  RGB replacement for rendering. Right-source primitives use source index -1;
  the optional GT-source-indexed centre Chamfer would therefore include only
  left-source primitives. Both fields still participate in native RGB rendering.
  This diagnostic is not a matched all-Gaussian geometry comparison.
- Native MVSplat lifting multiplies unit rays by its depth parameter; the encoder
  plane-sweep depth used for stereo metrics is retained separately. This port
  does not “correct” native Gaussian locations or substitute our stereo lift.
- DepthSplat's intermediate training-depth stacking is disabled for inference;
  final depth and GS heads are retained. Without a checkpoint cursor, its unused
  inference step argument is set to the declared final recipe step, disclosed in
  the native identity. MVSplat carries checkpoint step 300000.
- Native CUDA rasterizer source is unmodified; header inclusion is a build-only
  compatibility deviation. Native renderer **execution/parity is untested**.
  TorchHub downloaded DINOv2 `main`; the cached source files and initializer
  checkpoint are hashed in prepared configs rather than presented as a verified
  upstream Git commit. No claim of complete native paper reproduction.
- GPU availability was checked with `timeout 30 nvidia-smi`. GPU0 showed 12,597 MiB
  free (3,309 MiB used); GPU1 showed 2,240 MiB free (13,666 MiB used) at that check.
  This is a transient shared-device observation, not permission to assume later
  availability. No GPU work was launched. Prepared runner rechecks free memory,
  verifies UUID, then caps PyTorch allocation at **0.2**. No latency/FPS claim.

## Not tested, blocked, planned

**Not tested:** native GPU rendering, native parity, 512×640 memory feasibility,
608-wide shift inference, all 112 real frames, LPIPS on baseline outputs, final
per-frame JSON/tensor artifacts, shift slopes, quality aggregates and confidence
intervals. Full existing CPU regression suite is incomplete after the abort.

**Blocked:** the explicit metre near/far choice. No GPU shortage is claimed as the
current blocker. No benchmark failure, success, or scientific negative result is
inferred from CPU preparation.

**Planned after that question is answered:** create new resolved configs preserving
these null-bound templates; refresh hashes if code changes; check free GPU memory;
run one method/process at a time by UUID with the 20% cap; evaluate exactly the
prepared two cohorts and existing shift protocol. If native rendering/shape or
memory fails, retain the failure and partial outputs, report it, and do not change
resolution, signs, units, or checkpoint based on observed quality. Raw shift rows
must be summarized into slopes with an explicitly recorded regression definition;
that postprocessing has not yet been implemented/run.

Example **pending**, not executed, quality invocation after resolved configs exist:

```bash
export PYTHONPATH="$R/preparation-20261031T065247Z/source/src"
export CUDA_VISIBLE_DEVICES=GPU-b9607686-60e0-2545-b328-ca3f0e79e24c
cd "$R/upstream/mvsplat"
"$R/env/bin/python" -m reliable_endo_gs.baseline.posed_benchmark \
  --config "$R/NEW_RESOLVED_MVSPLAT_CONFIG.json" --output "$R/NEW_MVSPLAT_RUN"
# Then a separate serial invocation from "$R/upstream/depthsplat" using its config.
```

The sealed-64 worklist was not opened; the separate 48 development frames were not
loaded; SCARED 8/9 frame data/manifests were not inspected or evaluated. No EndoNeRF
data were downloaded, no existing job/session was interrupted, and no successful
output was overwritten. Requested repository additions are source/tests/docs only;
clones, models, caches, environments, snapshots and setup outputs are on D:.
