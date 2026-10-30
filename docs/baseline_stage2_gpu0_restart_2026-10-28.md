# Baseline Stage2 GPU0 restart and automatic KD comparison

Requested on 2026-10-28: restart baseline Stage2 on physical GPU0, end the Codex work session after the run is stable, and benchmark the completed baseline against final KD first.

Execution root: `/home/bkai/.local/share/reliableendo-gs/runs/baseline-stage2-gpu0-20261028T094723Z`.

**Completion follow-up:** Stage2 and both benchmarks finished successfully on 2026-10-28. Training completed at 17:25:01 local; the comparison completed at 17:26:59 local (UTC+7). The detached supervisor then exited normally. See the completion evidence and measured results below; the earlier handoff observations remain historical.

## Implemented run and original handoff

- Stage1 is complete at 100,100 updates. Stage2 restarts at update 0 from its retained checkpoint, SHA256 `d77f4cde9750a2fc3998c8c03bdc070fca19d41eab29ef5b8ef9238954671052`. The previous failed Stage2 had no saved checkpoint, so its 60 successful updates cannot be resumed.
- Stage2 uses physical GPU0, RTX 5070 Ti 16 GB, UUID `GPU-b9607686-60e0-2545-b328-ca3f0e79e24c`. The unrelated GPU1 training process was not stopped.
- Target: 10,000 Stage2 updates, 110,100 total. The original stage transition is preserved: fresh Stage2 AdamW/OneCycle/GradScaler, restored Stage1 CPU/CUDA RNG, and the original deterministic sample order. Subsequent restarts load the new Stage2 optimizer/scheduler/scaler/RNG checkpoint.
- Losses, learning rate, weight decay, clipping, schedule horizons, AMP, iteration count, split, resolution and batch size remain the existing recipe. The verified Linux byte cache and ordered input prefetch remain active. Checkpoint interval is operationally shortened to 100 updates, retaining the latest three and 10,000-update milestones.
- Isolated training source SHA256: `2f0766d3151d6f402e05273df3f19ee980f1083001b4c2634531b8894fefbe73`. Trainer SHA256: `cc7bf9741ac21350e3738735e1adb9dd1e96903b71c75ae5de809611cefc9a76`. Original upstream commit remains clean and pinned to `186fa2b4a2159b28393492f6df1aa444b54391a8`.

The initial stability gate passed at update 539: update-500 checkpoint had been saved, the previous failure point was exceeded, and the latest 100 logged losses and gradient norms were finite. At the recorded handoff review, Stage2 had reached **6,058/10,000**, with checkpoint `train/ckpt-s2-006000.pt` saved. See `STABILITY.json`, `stable-review.json` and live `status.json` under the execution root. Later checkpoints may prune the update-6,000 file according to the declared retention rule.

The final process check at 10:18:56 UTC (17:18:56 local) confirmed **6,919/10,000** updates, finite recent losses/gradients, a live trainer under the detached tmux supervisor, and the sealed automatic benchmark plan present. This is a progress observation, not a completion claim.

## Failure diagnosis and exact change

The unchanged Stage2 reproduced the failure at update 61 on `dataset_2/keyframe_3/frame_data000604`: 871 scale components became exactly zero. The upstream Gaussian head uses `Softplus(beta=100)` followed by an upper clamp of 0.01. Its head ran in FP32; sufficiently negative inputs underflowed to zero and the A00 renderer's strictly positive scale check rejected them. The failed parent outputs and unchanged diagnostic are retained.

`src/reliable_endo_gs/baseline/a00_model.py` adds an explicit, opt-in `zero_to_float32_tiny` policy. It changes only exact-zero scale components to `torch.finfo(torch.float32).tiny`. Positive values including subnormals are preserved; negative/nonfinite values still fail existing validation. The default policy remains `reject`. The original Softplus, upper clamp and trainable network remain unchanged. This is a disclosed numerical recovery, beyond an I/O-only change; it does not substitute a sigmoid or introduce a practical scale floor.

Both zero and the replacement have an FP32 squared value of zero. A native CUDA rasterizer fixture with three primitives and five zero scale components produced bitwise-equal images, radii and covariance diagonals, and zero maximum absolute differences for all six checked gradient arrays. This fixture establishes bounded numerical evidence, not bitwise equivalence of an entire training run. Native CUDA atomic operations and switching physical GPUs preclude such a claim.

The training log records the replacement count. By the handoff review the count was large, so finite training must not be interpreted as healthy learned Gaussian scales or good reconstruction. Preserve this evidence and inspect the final paired quality/coverage measurements before making paper claims. This run remains the explicitly declared **A00 metric-geometry baseline variant**, not certification of native paper-protocol parity.

The benchmark-only model factory now accepts explicit correlation and underflow policy arguments. The benchmark snapshot's scientific baseline classes/functions have identical ASTs to the active training snapshot; only construction arguments and the shared evaluator were added. The active training snapshot was not edited after launch. No repair/selector changes were imported into it.

## Automatic benchmark after training

The detached coordinator holds the global baseline lock and runs, in order:

1. Complete Stage2 and verify its result and final checkpoint counters.
2. Evaluate final baseline `train/ckpt-s2-010000.pt`.
3. Evaluate final KD `/home/bkai/.local/share/reliableendo-gs/runs/kd-train-1236-20261024T192708Z/attempt-15/training/final.pt`, SHA256 `f77547ef2cdc3d5030d537d83a203c0d9f8cf3978020c62a370af78d36cedd62` (10 epochs, 131,500 optimizer updates).
4. Write `benchmark/comparison/comparison.md` and `comparison.json` with both per-model artifacts preserved.

The first comparison uses the **same pre-existing ordered 56-frame val_7 worklist**, disjoint from train_1236. It does not select checkpoints or access final_test. Both models run frozen, base-only, with no repairs, selector or teacher at inference. GT is loaded separately after prediction. Common metrics cover both-view PSNR/SSIM/LPIPS, disparity/depth/geometry errors and coverage. Sampled point-cloud diagnostics use a declared 1,024-point cap and 256-point chunk size; they are not asserted to reproduce an unspecified paper point-sampling protocol.

Each actor runs in its own process on GPU0 with the same pinned Graphdeco renderer and input path. Baseline keeps Stage2 AMP stereo features and its FP32 GS/geometry path; KD keeps its checkpoint-declared FP32 deployment. Timing measures model, geometry, Gaussian construction and **both renders**, batch size 1, with 10 warmups and 200 synchronized repetitions cycling through the fixed worklist. Inputs are preloaded; I/O, decoding, H2D and metric computation are excluded. Peak PyTorch allocation includes the input cache and model, after removing LPIPS. Hardware, precision, individual latencies and metric provenance are saved. These measurements do not establish complete I/O-inclusive deployment latency or matched training compute. GPU1 may run other user work, so the host is not a dedicated benchmark machine.

The paired report retains missing metric counts, per-keyframe results, paired frame differences and whole-keyframe bootstrap intervals. This is one validation dataset with one seed per model: an initial development comparison, without a superiority or final-test claim. No new scientific margin is inferred from its results.

`src/reliable_endo_gs/evaluation/paired_models.py` owns the shared evaluator. The external worker constructs frozen models through existing canonical APIs, verifies assets/checkpoints, stages byte-identical validation inputs and invokes it. `benchmark-config.json`, `benchmark-preparation.json`, `benchmark_plan.json` and the separate `benchmark-source/` seal its configuration and source. Snapshot SHA256 is `30d662e4d04ab52163581ab77515b267e62fcc679177833bafbb3509ba8251d2`.

## Tested, pending and operational behavior

- **Passed:** 16 CPU tests across A00 model/underflow, optimization, ordered prefetch, and shared paired metrics/state/RNG/cohort checks; Ruff checks for changed library/test files and the benchmark worker.
- **Passed:** real GPU0 rasterizer zero-scale fixture parity, and real Stage2 integration beyond update 61 and the stability gate.
- **Passed:** benchmark asset hashes, split isolation, exact KD construction/tensors, and baseline Stage1 architecture/tensor loading on CPU. The CPU preflight suppresses only an optional CUDA availability probe because WSL driver probing stalled; production benchmarking checks the actual physical GPU UUID.
- **Pending at handoff:** completion of 10,000 Stage2 updates, loading its final checkpoint, and real final-model benchmark metrics/latencies. No benchmark result is claimed in this report.
- **Not run:** new repair/selector training, Phase I/II gate promotion, final_test, multi-seed comparisons or a whole-project test rerun. The prior repair/selector report remains a separate historical artifact.

The job lives in detached tmux session `baseline-stage2`, socket `/home/bkai/.local/share/reliableendo-gs/s2-20261028T094723Z.sock`; the supervisor is parented by tmux rather than Codex. Ending the Codex work session does not stop it. It permits at most three retries for explicit OOM/allocation/I/O failures; numerical or other fatal errors stop with a recorded failure. Benchmark failure is recorded and does not fabricate completion or overwrite successful outputs.

WSL `watch -n 0.5 nvidia-smi` polling was paused after its query blocked in `dxgvmb_send_sync_msg`. Only the recorded monitoring PIDs were paused, not other training. The coordinator restores those still-matching monitor processes after the train-and-benchmark chain finishes or fails; details are in `gpu-monitor-pause.json`. Read `status.json` for progress during the run.

Repository HEAD at handoff: `ac97de8a60ac4e396ca6ffd8ecfb153bf942994e`, with existing uncommitted work preserved. Execution identity is the retained isolated source/config/checkpoint hashes; the dirty workspace is not represented as a clean Git experiment.

## Verified completion and measured comparison

`status.json` records `event=complete`; `train/result.json` records Stage2 10,000/10,000 and 110,100 total updates. All 10,000 logged training losses and gradient norms were checked and are finite. Final checkpoint `train/ckpt-s2-010000.pt` has SHA256 `7c48dcc83994a4622d925b8b47c6c877d2db117054647a23a931979595f1106b`.

Both benchmark actors completed all 56 ordered frames on physical GPU0. Their checkpoint and per-frame output hashes were rechecked against the recorded identities. Both report unchanged weights and restored RNG. The tmux server exited after the coordinator completed its chain; no baseline process is expected to remain running.

Source artifacts: `benchmark/baseline/result.json`, `benchmark/kd/result.json`, their respective `samples.jsonl`, and `benchmark/comparison/comparison.{md,json}` under the execution root. The following values are rounded from those generated artifacts; they were not edited.

| Metric | Baseline | Final KD | Available paired frames |
| --- | ---: | ---: | ---: |
| Disparity EPE (px), lower is better | 15.4671 | 3.61528 | 55 |
| Stereo depth MAE (mm), lower is better | 18.8035 | 4.37328 | 55 |
| Left PSNR (dB), higher is better | 26.8851 | 25.4427 | 56 |
| Left SSIM, higher is better | 0.937525 | 0.788457 | 56 |
| Left LPIPS, lower is better | 0.048358 | 0.262777 | 56 |
| Right PSNR (dB), higher is better | 17.3291 | 22.0984 | 56 |
| Right SSIM, higher is better | 0.553114 | 0.725686 | 56 |
| Right LPIPS, lower is better | 0.321479 | 0.317083 | 56 |
| Inference median (ms), both renders | 31.9523 | 75.2497 | 200 timing repetitions |
| Inference mean (ms), both renders | 33.8428 | 84.7134 | 200 timing repetitions |
| FPS from mean latency | 29.5483 | 11.8045 | 200 timing repetitions |

These data show a tradeoff on this fixed development cohort: KD has lower stereo/geometry errors and higher right-view PSNR/SSIM, while the baseline has higher left-view fidelity and lower measured latency. KD does not demonstrate a speed advantage here. Baseline runs its original Stage2 AMP path and KD its original FP32 path; this is the declared deployment comparison, not an equal-precision ablation. Timing exclusions and the one-seed/development-only limitations above continue to apply.

One frame, `dataset_7/keyframe_4/frame_data001678`, has unavailable disparity metrics for both actors; it remains in the 56-frame image comparison. Sampled surface metrics have 54 available paired frames. Neither backend produced rendered-depth outputs for this benchmark, so the rendered-depth metrics remain explicitly unavailable, not zero. See per-frame records for metric counts and definitions.

The final training sample logged 980,261 exact-zero scale replacements. The recovery permitted completion but does not establish a healthy Gaussian scale distribution or fix the underlying parametrization. Preserve this negative evidence alongside the baseline's reconstruction/geometry tradeoff. No scientific recipe was retuned after seeing these results; final_test and later research gates remain untouched.
