# Feed-forward Gaussian baseline fine-tuning plan — 2026-10-31

Task E revises the earlier no-new-training decision: fine-tuning feed-forward GS
baselines on our protocol is now a permitted **proposed experiment**. This task
authorizes planning and disposable feasibility probes, **not a training run**.
No baseline was trained for 39,450 samples, no adapted checkpoint was saved, and
no validation/test quality was used to choose anything. The earlier feasibility,
addendum and run reports are retained unchanged.

**Current decision:** train-only bounds and the uncropped Splatt3R CPU predictor
forward are verified. MVSplat, DepthSplat-small, Splatt3R and pixelSplat all hit
the existing 20% GPU cap before completing a training step. Full-16-GB training
peaks and per-model training-time extrapolations are therefore **not measured**.
The ranked fine-tuning plan is conditional on resource permission, the order
decision and calibrated-renderer checks; no model is launch-ready under the
present cap.

Evidence root, including rules, source snapshots, official weights metadata and
probe logs:

`/mnt/d/3DGS_runs/reliableendo-gs/baselines-20261031/finetune-plan-20261031T154040Z/`

## Protocol and what is actually matched

Use only the pinned `train_1236.json`: **13,150 unique stereo pairs from datasets
1, 2, 3 and 6**, repeated for **3 epochs = 39,450 pair exposures**, batch one.
Dataset 4/5 exclusions remain in force. No sealed-64, dev-48 or final_test access;
Task E also performs no val_7 evaluation. Do not create a split by scanning the
dataset root. Refuse any manifest containing another dataset before opening its
assets. Missing/malformed frames stop the experiment; no silent skipping.

The training manifest's raw SHA-256 is
`5b5249c072647b122e9b0bc89abc141161da1f11be0ecf7adf4d469d23576296`;
its canonical A01 identity in both matched student-arm `run.json` records is
`2fc2a5c57b7b20001b512641c5a53dc55ee222c4447f2970d9bae025ba678901`.
These are different serialization identities, not conflicting datasets.

**Order discrepancy to resolve before a training launch.** The student seed is
verified as **20261006** in the frozen GT and matched-KD configurations. However,
the actual student-only runner repeats **manifest order**. Its
`SeededSampleOrder` branch requires an intervention config, absent in these arms;
neither run identity records a sample-order permutation. Therefore the requested
phrase “same ordered permutation seed” does not describe those executions.
Proposed correction: replay their actual manifest order three times, with RNG
seed 20261006 and no additional shuffle. Alternatively, generate one fixed
`random.Random(20261006).shuffle(indices)` permutation, repeat it three times and
explicitly disclose that order is **not matched** to those historical student
arms. The user clarification is pending; no choice is inferred from elapsed time.
Before execution, save/hash the exact ordered ID list, not just its seed. This
ambiguity does not prevent train-only range statistics or disposable probes.

| Matched in the proposed adaptation | Not matched / must be disclosed |
| --- | --- |
| Training split, 13,150 pairs, 39,450 exposures, three passes, full 512×640 A00 input grid, batch one | Architecture, parameter count, trainable subset, Gaussian count and representation |
| Seed 20261006; exact order after the discrepancy above is resolved | Official RE10K/ScanNet++/human pretraining versus student initialization; total lifetime training data and compute |
| Both input-camera RGB targets for baseline fine-tuning | Native/students' optimizer, learning rates, schedules, precision and original loss recipes; this is not an optimizer-matched attribution experiment |
| Frozen evaluation metric implementations and valid sample IDs in later development evaluation | Two input colour sources and learned SH versus our student's colour path; memory, training time and inference deployment treatment |

The **common proposed fine-tuning objective** is the arithmetic mean of supported
RGB MSE for the left and right rendered input cameras. Use deployment support,
not GT-valid-depth or prediction-dependent masks, for photometric supervision.
No GT disparity/depth loss or teacher KD loss is added. GT is used for the
train-only range rule below: consequently this is **not entirely GT-free
adaptation**. The MSE-only objective departs from official MSE+LPIPS recipes and
from the students' losses; disclose it rather than calling this native parity.
The frozen PSNR/SSIM/LPIPS/geometry evaluation code remains unchanged.

Start from the named official checkpoints, reset optimizer state and adaptation
cursor, and retain all native learned layers. Proposed fixed settings, chosen
before validation: MVSplat/pixelSplat AdamW at `2e-5`, weight decay `0.01`;
DepthSplat-small AdamW at `2e-5`, with its pretrained monocular parameter group at
`2e-7`; Splatt3R's two Gaussian DPT heads only, Adam at `5e-6`, weight decay zero.
Use the default Adam betas `(0.9,0.999)` and epsilon `1e-8`, `foreach=False` for
bounded optimizer temporaries, and gradient norm cap `0.5`. Proposed learning
rates are constant across the three adaptation epochs; they are not optimized
on val_7. These are adaptation choices, not claims of official optimal settings.
GPS remains conditional on an accessible official checkpoint and a documented
parameter/flow interface; do not invent a trained surrogate.

Keep the official mature opacity-mapping step when loading a pretrained model;
the fine-tuning optimizer cursor starts at zero. Do not accidentally restart its
representation annealing schedule. For DepthSplat, the proposed/probed objective
uses the final Gaussian field only; intermediate-depth/render losses are disabled
and this deviation is disclosed. Mixed precision means encoder autocast FP16,
FP32 native Gaussian rasterization and loss, with GradScaler. It does not mean
that all parameters or optimizer states occupy FP16 storage.

Save restart checkpoints only for recovery, with RNG, scaler, optimizer, ordered
sample cursor, config, code and source identities. The sole comparison artifact
is the checkpoint after **exactly 39,450 successful exposures/updates**. No “best”
epoch, early stopping on quality, favourable seed, crop, loss, model-size or
checkpoint selection from val_7. A persistent failure is reported; do not replace
the final checkpoint with a favourable intermediate one.

## Train-only depth ranges, frozen before new evaluation

The rule was written and SHA-256 hashed before this task's first GT load:

`range-rule.json` →
`bdc0202ae56231f08d8f0f9f896f4bb913726aed7b1580b3e4b2a8192fe3ddf6`.

This rule is shared across eligible methods. It uses every one of the 13,150 train
frames once, at the A00 512×640 grid. Reconstruct optical-axis depth **in metres**
through canonical `reconstruct_rectified`, accounting for the two principal
points. Select GT-valid, supported, finite positive depth with the same declared
effective-disparity numerical threshold `1e-6`. Do not use val/test GT, fitted
prediction scales or model quality to set ranges.

For reproducible bounded CPU storage, take at most **4,096** valid pixels per
frame, at deterministic midpoint quantiles of row-major valid-pixel indices;
take all if fewer exist. Concatenate these selected depths, compute NumPy linear
quantiles **q0.005 and q0.995**, then set **near = 0.9 q0.005**, **far = 1.1
q0.995**. These are percentiles of a **frame-stratified sampled train-depth
distribution**, not exact all-pixel percentiles or equal-pixel-weighted
percentiles of the complete dataset. Empty-GT frames are counted and contribute
no pixels; missing/malformed frames fail the scan. Reject nonfinite, nonpositive
or reversed bounds. No post-validation widening or tuning is permitted.

The first scan decoded RGB unnecessarily and was interrupted **only in this
task's own CPU process**. Its log is retained. A tested metadata/header/GT-only
loader replaced it, using A00's existing protocol, camera resize and TIFF
supervision conversions. The selected depth population and the pre-hashed rule
are unchanged. Placeholder RGB slots in statistics samples must never be fed to
a model; model probes use the complete `ScaredA00Adapter` RGB loader.

<!-- BEGIN TRAIN RANGE RESULTS -->
Completed CPU scan: **13,150 frames**, **243 empty-GT frames**,
**853,536,173 valid depth pixels**, **52,737,433 selected pixels**.
All frames remain in the photometric training cohort, including empty-GT frames.

| Train statistic | Metres (display rounded; use full precision from JSON) |
| --- | --- |
| q0.005 | 0.032268692404 |
| q0.995 | 0.143092940450 |
| Frozen near | **0.029041823164** |
| Frozen far | **0.157402234495** |

Result: `train-depth-range-metadata.json`, SHA-256
`a42398a30a128db1f97647aa9303067b09cb92e3583f19e3ff01139150b9c1dc`. Scan wall time **1044.793 s**. No validation/test asset was read.

<!-- END TRAIN RANGE RESULTS -->

| Method | Range mapping / additional range hyperparameters |
| --- | --- |
| MVSplat | Supply the shared near/far in metres in both context and target camera dictionaries. Keep official depth-bin count and Gaussian architecture. Do not apply RE10K bounds or baseline normalization after the bridge. |
| DepthSplat-small | Same metre near/far; native matcher takes **inverse depth**, internally `min_depth=1/far` and `max_depth=1/near`. Preserve its checkpoint-compatible scale/bin configuration and the small architecture. |
| Splatt3R | Learned point-map depth remains its native prediction, without GT scale fitting or a new clamp. Its renderer wrapper hardcodes `0.1/1000`; the integration must instead pass the shared train-derived bounds to its unchanged lower-level rasterizer. This call-boundary change is explicit. |
| GPS-Gaussian | Derive any configurable metric-depth/disparity endpoints from the shared bounds, frame K/baseline and canonical conventions. Verify native signed flow before converting; do not equate flow with positive A00 disparity by guessing. If fixed checkpoint search support cannot cover this rule, record it and stop rather than changing architecture or selecting bounds on val_7. |
| pixelSplat | Bypass its dataset **bounds shim**, which otherwise replaces ranges using `near_disparity`/baseline rules. Supply explicit metre bounds through the same posed camera bridge. Preserve three Gaussians per pixel and native sampling; changing them would invalidate this checkpoint comparison. |

Dimensionless learned scale multipliers and checkpoint-compatible depth-bin
counts are architecture settings, not new metric range choices. Freeze and hash
the resolved per-model config and final numerical range file before any future
validation evaluation. Bound clipping/out-of-range failures remain part of the
result; do not remove difficult GT pixels to improve a score.

## Feasibility procedure and observations

Physical inventory: two RTX 5070 Ti, each reported as **16,303 MiB**. GPU UUIDs:

- GPU0: `GPU-b9607686-60e0-2545-b328-ca3f0e79e24c`.
- GPU1: `GPU-8a3f4d31-24ea-da6f-96fc-4af0842789aa`.

The initial unprivileged NVML query was blocked by the OS; the permitted read-only
query succeeded. GPU0 used 131 MiB and GPU1 used 0 MiB at the recorded initial
check, with no compute-app rows. Availability is rechecked with `timeout 30
nvidia-smi` before each child. This does not reserve a GPU against later user jobs.
No existing process or tmux session was reused, paused or killed.

The earlier **20% per-process cap remains binding** unless the pending user
request explicitly raises it. On these boards that is approximately **3.18 GiB**,
not a 16 GB training allowance. Every GPU probe pins GPU1 by UUID, calls
`torch.cuda.set_per_process_memory_fraction`, and runs one child at a time. A
failure at 20% **does not establish failure on a fully available 16 GB GPU**.
No 90% attempt is authorized by silence. Two physical GPUs do not justify two
concurrent processes under the current one-process rule.

For each accessible model, the disposable probe attempts batch one, full 512×640,
one allowed train RGB pair, mixed precision, both native input-camera renders,
the declared MSE objective, backward and one optimizer step including Adam state
allocation. The actual failures below prevented completion of that path. The
probe saves no adapted model. The JSON records the failing stage if
any, source/checkpoint/range hashes, GPU identity, load averages, wall time and
PyTorch peak allocated/reserved bytes. These are allocator peaks, not a claimed
measurement of all driver/external allocations. An OOM or failed attempt has no
valid completed-step time. The projector caveat below limits interpretation of
mechanical probes; none is a quality experiment.

<!-- BEGIN STEP RESULTS -->
**No candidate completed an optimizer step under the 20% cap.** These are
failed full-grid attempts, not measurements of a successful training step or
peak requirements on an unrestricted 16 GB device. No valid per-model wall-clock
extrapolation for 39,450 updates exists yet.

| Candidate | Actual result at cap 0.20 | Observed partial allocator peak: allocated / reserved (GiB) | Attempt wall time including setup (s) | Successful step time / 39,450-step extrapolation | JSON source under evidence root |
| --- | --- | --- | --- | --- | --- |
| mvsplat | oom in `native_forward` | 2.725 / 3.078 | 31.043 | Unmeasured / unavailable | `mvsplat-step-cap0.20.json` |
| depthsplat | oom in `native_forward` | 2.973 / 3.016 | 20.903 | Unmeasured / unavailable | `depthsplat-step-cap0.20.json` |
| splatt3r | oom in `native_forward` | 3.003 / 3.123 | 58.705 | Unmeasured / unavailable | `splatt3r-step-cap0.20.json` |
| pixelsplat | oom in `native_forward` | 2.965 / 3.049 | 30.776 | Unmeasured / unavailable | `pixelsplat-step-cap0.20-attempt2.json` |
| GPS-Gaussian | Official checkpoint retrieval blocked: OneDrive 403; Baidu share-init HTML only | Not measured | Not measured | Unmeasured / unavailable | `metadata/gps-retrieval.json` |

All four OOMs occurred in the native encoder before rendering, backward or Adam
state allocation. Thus the allocator peaks above **understate complete-step
requirements**, and native CUDA rendering/backward remain untested. The first
pixelSplat attempt instead failed during CPU loading on missing `lightning`;
`lightning==2.4.0` was installed only in the D: venv and the retained second
attempt strict-loaded successfully before its OOM. Other protected environments
were not modified.

The raw WSL/PyTorch OOM messages contain an impossible process-memory total;
that field is not used as a measurement. The report uses the allocator counters
recorded independently in JSON. “GPU 0” in those errors is the process-local
`cuda:0` corresponding to the explicitly selected **GPU1 UUID**.

An additional planning lower bound, **not a measured peak**, is FP32 parameter
storage plus FP32 gradients and two FP32 Adam moments for trainable parameters:
`4 × all_parameters + 12 × trainable_parameters` bytes. It excludes activations,
rendering, buffers, workspaces and driver allocations.

| Candidate | All / trainable parameter count | Parameter + gradient + Adam storage lower bound (GiB) |
| --- | --- | --- |
| mvsplat | 11,955,734 / 11,955,734 | 0.178 |
| depthsplat | 38,346,416 / 38,346,416 | 0.571 |
| splatt3r | 729,044,772 / 40,405,916 | 3.167 |
| pixelsplat | 118,979,412 / 118,979,412 | 1.773 |

In particular, Splatt3R's frozen FP32 backbone plus head-training state alone
nearly exhausts the current cap. A different precision/offload/checkpointing
policy would be a separately documented engineering variant, not evidence that
the present probe succeeded. At the current cap **zero models are launch-ready**;
full-16-GB feasibility remains blocked pending explicit cap permission.

<!-- END STEP RESULTS -->

Splatt3R's official checkpoint was downloaded and locally hashed in Task E. A
strict CPU encoder load followed by an **uncropped 512×640** pair forward passed,
including both native point-map/Gaussian heads. Parameters: **729,044,772**, of
which **40,405,916** belong to the two selected trainable Gaussian DPT heads.
The successful CPU forward took **19.5127 s**; total load+forward wall time was
**36.9387 s**, peak process RSS **6,657,454,080 bytes**, four CPU threads, FP32.
This is neither GPU training speed nor an RGB-rasterization test. The first
attempt failed on a missing native `instance` input field; it was retained, and
the retry supplied the correct metadata without editing upstream code.

The model's fixed point-map branches are frozen in a head-only experiment.
Consequently depth/EPE derived from those point maps should not be advertised as
improved by this adaptation. Gaussian offsets can change centres; that is a
different quantity from the frozen point-map depth and must be labelled.

### Extrapolation, without fabricated throughput

For a **successful measured** step of duration t seconds, core compute for 39,450
updates is `39,450 × t / 3,600` hours. A single cold step includes kernel startup
effects and is not a steady-state average. This extrapolation excludes input
decoding, checkpoint writes, startup, compilation, retries and evaluation.
Report those separately; do not use an OOM attempt duration as t, or infer GPU
speed from CPU forward time. No latency comparison is claimed on shared GPUs.

| Assumed seconds per successful step | 39,450-step core compute, **scenario arithmetic only** |
| --- | --- |
| 0.5 | 5.48 hours |
| 1.0 | 10.96 hours |
| 2.0 | 21.92 hours |
| 5.0 | 54.79 hours |
| 10.0 | 109.58 hours |

These scenarios are **not per-model estimates**. Before a real launch, budget
loader/checkpoint overhead and require the projected complete job to fit the
remaining serial schedule. A useful engineering stop bar is at most about two
days per model including overhead; failure to meet it reduces the model count,
not the full-grid resolution or prescribed sample count.

## Adapter work and tests before actual fine-tuning

Reuse `posed_context` for images `[1,2,3,512,640]`, normalized half-pixel intrinsics
and **camera-to-world** poses. A00 K uses integer pixel centres: add `0.5` to each
principal point before dividing its row by W/H. Left is world;
`c2w_L=I`, `c2w_R=inverse(T_R_from_L)`. Translations and depths remain metres.
No per-frame baseline rescaling, GT pose/scale fit or hidden square crop. Keep
rendered depth, optical stereo depth and Gaussian-centre depth distinct.

**Projection correctness is a separate launch gate.** Source inspection finds
the native CUDA projection builders construct a symmetric frustum from focal
length/FOV; passing off-centre K into that API does not by itself establish that
its principal point is rendered correctly. Existing context tests establish
encoder K/pose conversion, not this end-to-end rasterizer property. Before
training, project synthetic off-axis points with unequal principal points and
verify their rendered screen positions against the canonical A00 camera. If the
native path assumes centred K, add a thin projection adapter using the existing
canonical calibrated projection/rasterizer interface, retain the native Gaussian
representation/SH and disclose the camera-boundary correction. Do not edit
upstream or hide an image crop/warp. This check/fix remains **planned**, and
native hardware probes do not certify it.

Concretely, reuse `rendering.production.native_camera_batch_from_rectified` for
the camera matrices and explicit metre bounds, subject to the integer/half-pixel
point test. Retain a native SH/covariance rasterizer call. The production
RGB-colour renderer's `shs=None` path is **not** a substitute for these models'
SH appearance. This preserves formula ownership without replacing their learned
representation with our Gaussian construction rules.

These gates verify units, camera semantics and execution, not reconstruction
accuracy. Poor raw geometry or PSNR is negative evidence to report, not a reason
to fit a scale, discard a valid run or choose another checkpoint.

| Integration | Work remaining | Required validation |
| --- | --- | --- |
| Shared fine-tuning runtime | Add a small baseline integration module for trainability, fixed loss/optimizer groups and full-state final/resume handling. Add a thin CLI; no scientific logic solely in scripts. The existing frozen predictor loader cannot be used unchanged as a trainer: it returns eval mode with gradients disabled. | Tiny CPU tests for expected trainable keys, both-view gradient flow, exact exposure count/order, seed/RNG/scaler/cursor resume equivalence, no checkpoint ranking, and forbidden-ID rejection before file/device access. |
| MVSplat / DepthSplat | Reuse strict checkpoint loading and the posed camera bridge; add the calibrated projector check, native AMP renderer casts and explicit train/eval boundaries. Preserve final-field convention and native SH/covariances. | Synthetic unequal-K point projection, rectified plane depth/disparity including principal offset, both source cameras, finite gradients through means/covariances/SH, correct final native opacity schedule and no hidden intermediate supervision. |
| Splatt3R | Direct tensor inputs avoid its square-cropping demo loader. Preserve native `[-1,1]` normalization, `true_shape`, `instance`, patch-16 layout, first-camera frame and XYZW covariance helper. Freeze backbone/point-map branches; train only the two official Gaussian DPT modules. Use its native covariance/SH helpers and lower-level renderer with explicit bounds. Raw metric-scale/gauge validation on synthetic/calibrated fixtures remains required; never use GT similarity fitting. | Full-rectangle native forward already passed on CPU. Add real renderer/off-centre-K tests, head-only weight/buffer invariance after a probe, point-map-depth invariance, known-plane rig geometry and native-versus-bridge RGB parity on a supported synthetic camera. |
| GPS-Gaussian | Begin only after official final Stage-2 retrieval and SHA-256. Wrap signed flow for both source orders, foreground/support masks, metric cameras, native Gaussian attributes and renderer. State the human-to-tissue mask/domain change. | Synthetic horizontal shifts/sign, raw versus effective disparity, metre baseline and both input orders. Stop on ambiguous native sign/units or a need to retrain from scratch. |
| pixelSplat | Its three Gaussians per ray exceed the existing evaluation predictor's one-Gaussian assumption. The mechanical loader/render probe must not be presented as a finished geometry adapter. Preserve native epipolar sampling, bypass the bounds shim and pin DINO source/assets. | Native full-grid memory test; reproducible sampling and checkpoint-compatible counts; define and verify a native depth statistic before derived EPE. If that extraction is unverified, report geometry unavailable rather than relabelling rendered depth or choosing a favourable sample. |

New Task E code is confined to
`src/reliable_endo_gs/baseline/finetune_feasibility.py` and its tiny CPU tests.
It implements only statistics and disposable native boundaries/loaders, not the
future multi-epoch training/resume runtime. Upstream imports stay in this layer;
upstream repositories and existing `src/` modules were not edited. The native
Splatt3R dependency contains its own legacy path helpers; no first-party
`sys.path` mutation was added. All downloaded assets/caches remain on D:.

## Fair comparison and evidence boundaries

These models can use **both input RGB colour sources**. Their right-view result
is **input-camera reconstruction**, as is the left view. It is not a held-out
novel-view test and is not identical to reconstructing the right camera using
left-image colours only. Keep the rendering table explicitly labelled and
report native Gaussian count, SH capacity and both colour inputs. Strong input
reconstruction alone does not establish strong geometry.

Also compare disparity EPE and stereo depth MAE/RMSE with the frozen canonical
conversion, plus coverage/invalid counts. MVSplat/DepthSplat yield optical-axis
depth that can be converted to A00 raw disparity; other methods require the
explicit extraction checks above. Never treat point-cloud range as optical-z.
Use the existing frozen geometry metrics, `frame_metrics` and LPIPS AlexNet v0.1
in later evaluation; do not change their masks/definitions to accommodate a
method. Plan the existing disparity-shift test for every verified disparity
output. There are no new quality metrics or shift slopes from Task E.

Per-scene optimization methods remain references only. Head-only or common-loss
fine-tuning is labelled as such, not as native paper reproduction. Endo-E2E-GS
native parity remains unverified. No superiority, speed or scientific gate claim
is decided by this feasibility exercise.

## Provenance and reporting status

<!-- BEGIN PROVENANCE RESULTS -->
| Model | Official primary source / inspected commit | Official checkpoint SHA-256 |
| --- | --- | --- |
| mvsplat | [donydchen/mvsplat](https://github.com/donydchen/mvsplat/tree/01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01) / `01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01` | `83d0d9eaa753fa4a1f925288dc1f90b8c3297fad0ab0f6ed1f11a1c5946da25a` |
| depthsplat | [cvg/depthsplat](https://github.com/cvg/depthsplat/tree/2dad25a7b9ba7c5537aba08463ef88d8511a06be) / `2dad25a7b9ba7c5537aba08463ef88d8511a06be` | `cfeab6b10077ad9bd4097b82b029737221bfbbbd3b4dfca5fc0feaf860276506` |
| splatt3r | [btsmart/splatt3r](https://github.com/btsmart/splatt3r/tree/bda1dd07cf84b9189baabb1b8beac243959cfe5d) / `bda1dd07cf84b9189baabb1b8beac243959cfe5d` | `963a0e9a992150e1509fcc952a45b22d24263167edc814238abcf0dd956e0b18` |
| pixelsplat | [dcharatan/pixelsplat](https://github.com/dcharatan/pixelsplat/tree/59d420a409cd0f059366808a9f956001536b55f1) / `59d420a409cd0f059366808a9f956001536b55f1` | `73355f46a582e00811ab6ea1b3bd0aebd873eeb31fdef6cec5890421c15e4536` |
| GPS-Gaussian | [aipixel/GPS-Gaussian](https://github.com/aipixel/GPS-Gaussian/tree/0024776deee4824f270d4bb534a17ffd85f63cf2) / `0024776deee4824f270d4bb534a17ffd85f63cf2` | Not available; no official checkpoint bytes retrieved |

Native sources remain unmodified. See the earlier feasibility reports for
component licences, including MASt3R and the Graphdeco rasterizer; an integrating
MIT licence does not supersede their separate terms.

pixelSplat's native DINO torch.hub source is pinned at `7c446df5b9f45747937fb0d72314eb9f7b66930a`. Its constructor also downloaded the following official initialization assets to D:; they are overwritten by strict RE10K loading, not extra selected model variants:

| Constructor asset | Locally computed SHA-256 |
| --- | --- |
| `dino_vitbase8_pretrain.pth` | `575f0efc4838938314afa09897a62ed2b87919928b5edcd133abb907328995eb` |
| `dino_resnet50_pretrain.pth` | `156f8c4166a23dc2951ae811e39d76a06269c565932edf647c0187e65cd7aa7c` |

Native GPU-probe source manifest SHA-256:
`35f06e179f51076101ec9fac674d3f99b0923fb3c3073220e4a2692cee1b8e01` (`source-native-probes/source-files.json`).
The separate range-scan snapshot is `source/`. Project HEAD recorded by all
GPU attempts is `2be5ceaedd657d5b0785a580fd2269c8e8c97864`; first-party
changes remain uncommitted.

Actual GPU commands were the following **disposable one-step probes**, not a
training launch. The shell scripts export D: caches, the source snapshot,
upstream working directory, GPU1 UUID and cap 0.20; inspect them before replay.
Existing output files cause failure rather than overwrite.

```bash
E=/mnt/d/3DGS_runs/reliableendo-gs/baselines-20261031/finetune-plan-20261031T154040Z
bash "$E/probes/run_native_step.sh" mvsplat
bash "$E/probes/run_native_step.sh" depthsplat
bash "$E/probes/run_native_step.sh" splatt3r
bash "$E/probes/run_native_step.sh" pixelsplat
# Retained retry, after installing lightning==2.4.0 only into the D: venv:
bash "$E/probes/run_native_step-attempt2.sh" pixelsplat
```

Actual CPU scripts are `probes/range_scan_metadata.py` and
`probes/splatt3r_cpu-attempt2.py`, using the D: venv with
`CUDA_VISIBLE_DEVICES` **unset**. The former uses `PYTHONPATH="$E/source/src"`;
the latter uses the first-party source plus the pinned Splatt3R vendored
MASt3R/DUSt3R/CroCo paths and four CPU threads. Exact script text, failed initial
attempts and stdout logs are retained in `probes/`.

The post-probe package check found two metadata conflicts: Lightning 2.4.0
required older fsspec/packaging than the inherited `2026.7.0`/`26.3` versions.
The GPU-probe environment record is retained in
`metadata/environment-at-gpu-probes.txt` (Lightning was added between the two
pixelSplat attempts). Afterwards, **only the D: venv** received
`fsspec==2025.10.0` and `packaging==24.2`; protected environments were not changed.
The new package check reports **No broken requirements found**, and the same
**49 CPU tests passed again in 7.90 s**. See
`metadata/pip-check-after-dependency-fix.txt` and
`probes/pytest-after-dependency-fix.log`. The GPU OOM measurements were not rerun
or relabelled as measurements under this later dependency state.

<!-- END PROVENANCE RESULTS -->

Record project Git HEAD **and** a hash/snapshot of uncommitted adapter code;
HEAD alone does not identify this workspace. Each actual experiment additionally
needs native/dependency commits, checkpoint/config/range/order hashes, exact
commands, seed, hardware/software, load averages, wall time, allocator peaks and
per-sample results paths. Do not overwrite any run directory or edit generated
JSON measurements. No Git commit or push was made.

| Status | Task E scope |
| --- | --- |
| **Implemented** | This plan; the pre-hashed train-only range rule; minimal statistics/native-probe integration and tiny CPU tests. This is not a fine-tuning trainer. |
| **Tested** | CPU statistics isolation/conventions and equivalence to full A00 camera/GT loading, both-view loss gradients, native render-boundary tensor flow, full rectangular Splatt3R encoder/heads, and the existing posed/import boundary checks. Final generated probe status above supersedes provisional hardware status. |
| **Not tested** | A 39,450-update adaptation, checkpoint-resume runtime, validation/test evaluation, final scientific camera parity, multi-frame peak memory, steady-state throughput, model-selection robustness or shift-test slopes. Successful one-step execution, if any, is weaker than complete-run feasibility. |
| **Blocked** | Historical order discrepancy, any requested cap above 20% without explicit permission, GPS checkpoint access, projection/metric-frame checks, and any per-model OOM/dependency failure recorded above. A cap-limited OOM is not labelled a 16 GB OOM. |
| **Planned** | Resolve order/cap, finish the calibrated adapter tests, implement the small training/resume runtime, freeze config/order/range identities, then request/receive a separate instruction to launch the actual adaptation. No long run is queued by this plan. |

Final targeted CPU validation: **49 passed in 7.78 s** across the new feasibility
tests, existing posed/import boundary tests, A00 adapter tests and analytic
camera/disparity tests. Ruff check/format and `git diff --check` pass. This is not
a full-suite result and does not resolve the earlier `test_a00_model` double-free
observation. The first test invocation failed during temporary-directory setup
because its D: parent did not exist; the parent was created and separate retry
logs were retained. No test assertion or existing test was removed to obtain a
pass. `probes/pytest-final.log` is the final evidence.

## Ranked recommendation: at most four adaptation candidates

Ranking is based on protocol fit, available artifacts and engineering cost,
never val_7 quality. The first two share an existing bridge. Splatt3R supplies a
different frozen geometry prior; GPS is an explicitly conditional fourth slot.
There is no obligation to fill all four slots if a gate fails.

| Rank | Model / proposed adaptation | Deadline decision and stop conditions |
| --- | --- | --- |
| **1** | **MVSplat RE10K**, full encoder/GS fine-tune | First priority after projection and memory gates. Allow at most about one engineering day beyond shared infrastructure; require full-grid AMP backward/Adam step within the permitted cap and a complete-job budget around two days. Stop rather than silently downsample, alter the architecture or shorten 39,450 exposures. |
| **2** | **DepthSplat-small RE10K**, full small GS model with lower monocular-branch LR | Reuse the same protocol/runtime; allow about 1–2 engineering days. Keep the already selected small GS checkpoint and final-field loss. Stop on incompatible native backward, memory over the permitted budget, unresolved inverse-depth conventions or >3-day port. |
| **3** | **Splatt3R**, Gaussian DPT heads only | CPU rectangular forward is verified. Allow about 2–3 engineering days for calibrated native rendering and frozen-branch tests. Stop on unresolved metric frame, required GT fitting, incomplete head freeze, memory failure or projector work exceeding three days. Report frozen point-map geometry independently of any rendering gain. |
| **4 — conditional** | **GPS-Gaussian**, official Stage-2 warm start, photo-only adaptation | Use this slot only if an official checkpoint can be retrieved and hashed promptly, then sign/units and full-grid backward pass. Cap retrieval effort at roughly half a day and the whole port at three days. Otherwise omit it; no scratch-training substitute and no unverified mirror. |
| **Outside shortlist** | **pixelSplat** | Last priority: native patch-8 ViT/epipolar attention and three Gaussians per pixel add substantial full-grid memory risk and extra geometry-adapter work. Preserve negative probe evidence. Do not spend the deadline rewriting its backbone, reducing its Gaussian count or choosing a different checkpoint from val_7. |

Use a serial schedule under the current one-process rule: shared adapter/range
work first, then MVSplat, DepthSplat, and only then a passing third/fourth slot.
Reserve the final several days for frozen development evaluation, provenance and
paper-table checks. The two 16 GB GPUs are alternatives, not assumed 32 GB of
pooled memory or permission for concurrent jobs. If only two models pass the
gates, run those two and report why the other slots were dropped.

The default commitment should be **two models**. Add a third or fourth only if
its remaining port time plus conservative measured compute/IO extrapolation fits
the remaining serial budget **after reserving three days for evaluation/reporting**.
The two-day-per-job bar is an upper stop limit, not a promise that four such jobs
plus four ports fit the deadline. An inaccessible GPS slot can be omitted or used
as a substitute; it is not a mandatory fourth job.
