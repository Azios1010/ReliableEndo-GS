# Frozen A04 benchmark harness

`benchmark a04` evaluates an existing A04 checkpoint. It has no optimizer and
does not update the MLP or teacher. Training config, teacher identity and renderer
identity are checked against the checkpoint. `final_test` is passed at forward
time; the saved `final_train` config is never edited to bypass checkpoint checks.
Original SCARED dataset_8–9 is the only evaluation pool. The immutable manifest
and role contract remain unchanged. No SCARED-C loader or baseline internal is
imported by this path.

## Metrics and meaning

| Metric | Definition / status |
| --- | --- |
| Left/right PSNR | Full RGB [0,1], per-frame MSE, no tool/occlusion exclusions |
| Left/right SSIM | Gaussian window 11x11, sigma 1.5, zero padding 5, C1=0.01²/C2=0.03²; parity checked with pinned upstream lib/loss.py |
| Left/right LPIPS | AlexNet v0.1, full RGB, [0,1] mapped to [-1,1], official scaling and calibrated layers |
| Disparity EPE | Output-grid pixel error on joint valid GT/prediction; coverage reported |
| Disparity 1px accuracy | Fraction of all valid GT pixels with valid prediction and strict error <1px; invalid predictions are failures |
| Stereo depth MAE/RMSE | Millimetres; GT and predicted raw disparity converted by canonical rectified K/baseline geometry |
| Stereo depth AbsRel | Relative depth error on joint valid pixels |
| Stereo surface CD / NC | Defined sampled unsquared symmetric Chamfer (mm) and symmetric nearest-neighbour absolute normal dot |
| Gaussian-centre CD | Actual Gaussian centres with GT-valid source pixel IDs vs GT point cloud, same declared sampled Chamfer definition |
| Rendered depth | Reported separately; **unavailable with current RGB-only native renderer**, never replaced by teacher depth |
| Runtime | Synchronized complete live-teacher/A03/MLP/both-view render with H2D transfer; excludes disk, GT and metric calculation |
| VRAM | Model-call PyTorch allocated/reserved peaks in profile.json; whole-harness peaks also include LPIPS/geometry metrics |

Only canonical `reconstruct_rectified` performs principal-point correction,
metric depth conversion and backprojection. Ground-truth disparity is read
**after inference**, as separate supervision. It is never passed to the model.
GT masks are nearest-resized and disparity scales with horizontal resize through
the existing A00 adapter; K uses the same pixel-centre transform. GT validity,
prediction coverage, surface-normal validity, hashes and calibration provenance
are reported per frame. GT depth here is derived from rectified structured-light
GT disparity, not an independent extra depth sensor file.

CD has a configured deterministic cap (default 2048 points), choosing evenly
spaced positions in each row-major valid point list. The nearest-neighbour
distance is unsquared Euclidean distance and the two directional means are
averaged. NC excludes fallback normals; it uses those same nearest neighbours
and absolute unit-normal dot products. This is a documented approximation,
**not a claim that the paper used the same CD/NC protocol**.

Per-frame JSONL retains denominators. Reports contain per-sequence/per-dataset
and overall frame means, valid-weighted values, and an unweighted sequence macro
mean. Valid-weighted RMSE pools squared error before taking a square root.
Missing/empty metrics are null with explicit available/unavailable frame counts.
`metrics.csv` exports these generated scalars and denominators for comparison tables.
Finite zero-visibility RGB renders are scored and flagged; errors stop the job
and preserve the completed prefix, rather than silently removing difficult frames.

## Local LPIPS assets

No package is installed and no weights are downloaded at runtime. The first-party
metric uses existing torchvision and two explicit local files. Both are checked
against pinned SHA-256 identities before any teacher inference:

- AlexNet IMAGENET1K_V1 `alexnet-owt-7be5be79.pth`:
  `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`.
- Official LPIPS `weights/v0.1/alex.pth`:
  `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`.

The calculation follows [official PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
(BSD-2-Clause). AlexNet architecture comes from torchvision (BSD-3-Clause).
Parity against locally installed official `lpips==0.1.4`, loaded with those same
backbone/calibrated weights, is part of server verification. An arbitrary random
backbone cannot pass asset validation.

## Run

Set `RELIABLE_ENDO_DATA_ROOT` to the external processed SCARED_A00 root,
`RELIABLE_ENDO_TEACHER_HOME` to the existing FoundationStereo installation,
`A04_CHECKPOINT` to the frozen checkpoint, `LPIPS_BACKBONE` and
`LPIPS_CALIBRATION` to the existing assets, and `A04_BENCH_ROOT` to a fresh external
run folder. Execute from the repository root:

```bash
env CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$PWD/src:$RELIABLE_ENDO_TEACHER_HOME/source" \
  RELIABLE_ENDO_TEACHER_CONFIG="$RELIABLE_ENDO_TEACHER_HOME/runtime.yaml" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$RELIABLE_ENDO_TEACHER_HOME/venv/bin/python" \
  -m reliable_endo_gs.cli.main benchmark a04 \
  --checkpoint "$A04_CHECKPOINT" \
  --data-root "$RELIABLE_ENDO_DATA_ROOT" \
  --manifest "$RELIABLE_ENDO_DATA_ROOT/a01_manifest.json" \
  --roles configs/teachers/a01_scared.yaml \
  --protocol configs/evaluation/a04_benchmark.yaml \
  --lpips-backbone "$LPIPS_BACKBONE" \
  --lpips-calibration "$LPIPS_CALIBRATION" \
  --output-dir "$A04_BENCH_ROOT/results" \
  --max-samples 8
```

The bounded selection spans keyframes before inference; sample IDs are recorded.
Remove `--max-samples 8` to evaluate the entire final_test manifest. Use
`--sample-id dataset_8/keyframe_0/frame_data000000` for a single explicit sample.
There is no need to run a second complete RGB decoding preflight: each selected
sample is validated by the A00 loader at evaluation time.

To resume an incomplete run, repeat exactly the same command with `--resume`.
Checkpoint, code, dataset/role identities, weights, protocol and selected IDs must
match. Successful reports are never overwritten. Previous error artifacts remain.
An output-directory lock prevents concurrent writers and is released when the process exits.
An incomplete final JSONL line is rejected rather than silently truncated.

## Paper comparisons

The [Endo-E2E-GS paper](https://www.sciencedirect.com/science/article/pii/S0141938226000168)
reports SCARED at 1280x1024; the current checkpoint input grid is 640x512.
`resolution_hw: null` preserves that checkpoint grid. An external protocol with
`resolution_hw: [1024,1280]` evaluates a different grid without changing the saved
training config or weights; validate GPU memory on a bounded sample first.

`paper_protocol_equivalence` deliberately stays false. Matching metric names
does not establish matching split, frame list, crop, mask, view, LPIPS backbone,
point sampling or aggregation. The metric functions/aggregation can be shared
with a separately reproduced baseline; this CLI runs A04 only and does not add
baseline integration or alter historical baseline code. A matched baseline run
and protocol audit are still required before comparing paper table values.

`dedicated_gpu: false` records timings as shared-device diagnostics. Disk-loaded
full evaluation duration is not inference FPS. Model-only profiling repeats the
first selected sample with declared warmup/repetitions; it is not an all-scene
latency distribution. Compare efficiency on an idle GPU with the same workload,
precision, views, warmup and repetitions for both methods. PyTorch memory omits
untracked native allocations; device-wide nvidia-smi includes other processes and
must not be relabelled as this model's peak.
