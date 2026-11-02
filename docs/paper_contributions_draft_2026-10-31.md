# Paper contributions: draft wording and claims ledger (2026-10-31)

Status: **DRAFT, updated 2026-10-31 evening.** Development numbers come from the 56 benchmark `val_7` frames (one unseen sequence, four keyframes), one seed unless stated.
The confirmatory run (sealed 64 + `final_test` datasets 8 and 9, frozen L6 refiner, B1 primary and C2 secondary) is DONE: pre-registered verdict **"not supported"** (one failed
non-inferiority condition, N3, on dataset_9), see ledger rows C3e-C3g and `~/.local/share/reliableendo-gs/reports/confirmatory-20261031/`. The base-model comparison on `final_test`
(COMPARISON_PLAN.md) and the cost line have NOT been run. Rules for reading the runs are in `~/.local/share/reliableendo-gs/reports/kd-attribution-20261031/CRITERIA.txt` and
`CRITERIA_ACTION.txt` (hashed before the results; official numbers in its `results/REPORT.txt`).
Per `AGENTS.md`: write "the proposal hypothesizes" until evidence exists; do not state a claim in the left-hand column of section 3 before its condition is met.

## 1. Positioning (one paragraph)

Feed-forward Gaussian splatting from a stereo pair is established outside medicine (GPS-Gaussian, Stereo-GS, StereoGaussians) and has one recent
endoscopic instance (Endo-E2E-GS, Displays 2026). Distilling a stereo foundation model into a fast student is established for natural images
(Fast-FoundationStereo). Per-Gaussian residual refinement is established for feed-forward GS (DeltaSplat and others). The paper therefore cannot
claim these ideas. What the evidence supports is an **empirical account of why a ground-truth-trained feed-forward endoscopic model fails across
sequences, a ground-truth-free distillation recipe that does not, and a very small scale-correction module**, evaluated across sequences and datasets.

## 2. Candidate contributions (English wording for the paper, each tied to a ledger row)

* **C1 - Diagnosis.** We show that a ground-truth-supervised end-to-end feed-forward endoscopic GS model degenerates to a near-constant disparity
  prior when the test sequence leaves the training disparity range: its prediction is insensitive to a controlled disparity shift (slope 0.0 against
  1.0 for a stereo foundation model and our student), giving 15.5 px EPE on an unseen sequence against 3.6 px. We provide the shift test as a
  reusable protocol (complementary crops, constant disparity change at fixed scale).
* **C2 - Ground-truth-free recipe.** Distilling a stereo foundation model into a fast student removes the need for disparity ground truth, keeps most of the teacher's accuracy
  (3.62 vs 3.19 px) and, at equal architecture, schedule and seed, reaches a lower EPE than ground-truth supervision on an unseen sequence (3.45 vs 4.17 px, matched arm); the difference did not meet the
  pre-registered threshold (r = 1.21 < 1.25: "mixed/inconclusive"), so it is reported as a measurement, not as a significant improvement.
* **C3 - Lightweight scale correction.** A 1.7k-parameter, single-pass, feedback-free refiner improves cross-view PSNR by about +1.0 dB on three differently supervised/trained students
  and under input perturbations; the gain comes from the Gaussian-scale residual alone. On `final_test` the right-view gain is +0.60 dB (dataset 8) and +1.33 dB (dataset 9) with the pre-registered
  verdict "not supported" (N3 on dataset 9); see rows C3e-C3g.
* **C4 - Speed.** The deployed pipeline runs at 55 FPS (bit-exact graph) to 61 FPS (compiled kernels) at 512x640 with a 0.8 ms marginal cost for the refiner.
  **Only if** the compared baseline receives the same deployment treatment; otherwise state our FPS without a comparison.

## 3. Claims ledger

| ID | Claim | Evidence now | Required before it may be written | Wording allowed today |
| --- | --- | --- | --- | --- |
| C1 | Baseline ignores the stereo signal across sequences | Shift test: baseline slope -0.01 (train) / 0.01 (val_7), student 0.99 / 1.01, teacher 1.00; two baseline checkpoints fail on val_7 (11.3 px released SCARED-C stage 1, 15.5 px retrained) | Same on `final_test` and on one external dataset; baseline reproduction disclosed (native parity not verified, retrained on 4 sequences) | "In our protocol, the baseline we trained and the released checkpoint ..." |
| C2a | Student retains the teacher's accuracy | EPE 3.62 vs 3.19 px, depth MAE 4.37 vs 3.97 mm, 55 frames | `final_test`, external data | descriptive |
| C2b | KD vs GT supervision at equal architecture | Matched arm (39,450 samples, same snapshot/init/order/schedule, one seed): EPE 3.45 vs 4.17 px, depth MAE 4.38 vs 5.17 mm, KD lower on 46/55 frames (r = 1.211, w = 0.84); KD lineage (unmatched) r = 1.172; verdict "mixed/inconclusive" for both | More seeds do not fix the 4-keyframe limit; needs more sequences (e.g. the `final_test` comparison with a re-registered rule) | "lower EPE than the GT-only student in a matched comparison (one seed, one sequence, below the pre-registered threshold)"; never "better", "significant" or "improves generalisation" |
| C2c | KD explains the advantage over the baseline | **Not supported**: GT-only student of our architecture also reaches 4.17 px | - | forbidden |
| C3a | Refiner gain is generic across students | Right-view PSNR +0.99 (10-epoch KD), +1.16 (3-epoch GT-only), +0.92 dB (3-epoch matched KD), positive on 96-98% of frames; pre-registered rule "generic" met | - | "on the three students tested" |
| C3b | Refiner is stable across training seeds | 3 seeds on the 10-epoch KD student: +0.994 / +0.951 / +1.126 dB (range 0.175 dB <= 0.3): pre-registered rule "STABLE" | - | "stable across three seeds on one student" |
| C3c | What the refiner fixes | Zeroing scale: +0.99 -> -0.19 dB; zeroing rotation or alpha: unchanged | - | "the gain comes from the scale residual" (not "repairs rotation/opacity") |
| C3d | Robust to input perturbation | 7 conditions, +0.57 to +1.11 dB, positive on >= 95% frames (thinnest: noise 0.05) | - | descriptive |
| C3e | Repair gain on unseen sequences (confirmatory, frozen config B1) | Sealed 64 (dataset 7): +1.154 dB [0.996, 1.312]. `final_test`: dataset 8 +0.599 dB [0.581, 0.616], 90.3% frames positive, keyframe_0 only +0.12; dataset 9 +1.328 dB [1.307, 1.349], 99.5% positive. Right SSIM +0.030 / +0.063, LPIPS -0.033 / -0.062, no holes. C2 (compiled kernels) identical within 0.001 dB | Cost line (latency) not measured | State the measured gains per dataset AND the verdict: pre-registered result "not supported" because of N3 on dataset 9; never "validated" |
| C3f | Failed condition | N3 (right-view MSE of the last 80 columns <= 1.05x STOP): dataset 8 0.986, dataset 9 **1.054**; bar and verdict not edited | - | report as is |
| C3g | Where the N3 excess comes from (EXPLORATORY, post hoc, dataset 9 every 10th frame + dev) | Refined is worse than STOP only in the last ~one-disparity columns of the right view (ratio 1.15 at the edge, 1.0 about 40 columns in; dev: crossing ~39 / ~54 / >=100 columns for edge disparity 42 / 61 / 83 px); left view better everywhere; STOP error in the strip is ~7.6x the interior median and the strip carries ~41% of STOP's right-view MSE (dev) | Mechanism (columns the left view cannot see) is a hypothesis, per-frame correlation weak | "consistent with the hypothesis that columns without a left-view counterpart are not repaired by a refiner that only reshapes existing Gaussians" |
| C3h | Edge-strip guard (keep the unrefined right image in the strip) | Developed on 56 dev frames only (spec hashed before running): N3 1.028 -> 0.995, right-PSNR +0.996 -> +1.102 dB; fixed 60-column strip fails mid/high disparity terciles; implemented opt-in, uncommitted, extra right-view rasterisation NOT timed | Evaluation on unseen data (the `final_test` frames are already exposed: only a labelled re-evaluation); latency | "repair withheld where the right view has no left-view counterpart" (development evidence); never "edge error fixed" or a guarded-path FPS |
| C4 | Faster than the baseline | Ours 18.0 ms (bit-exact) / 16.3 ms; baseline measured 32 ms eager, **not graphed** | Baseline with the same CUDA-graph treatment on the same GPU | forbidden; state our FPS only |

## 4. Facts about the added data (to verify before planning around them)

* **EndoNeRF** (pulling and cutting, one da Vinci prostatectomy viewpoint; roughly 63 and 156 frames according to papers I could not check against the release).
  The public release is described as `images/`, `depth/`, `masks/`, `poses_bounds.npy`. **Depth comes from STTR-Light stereo matching and tool masks were drawn by hand: it is
  not ground truth.** I found no explicit statement that right-view images and calibration are released with it. **Check this first: the pipeline needs a rectified stereo pair,
  intrinsics and baseline, and so does the compared baseline.** If they are absent, EndoNeRF cannot be used for the stereo-input methods.
  If usable: report image metrics (PSNR/SSIM/LPIPS, tool pixels masked) and disparity agreement with the STTR depth as "agreement with a stereo-network reference", never as accuracy.
  Per-scene methods (EndoNeRF, EndoGaussian) use a different protocol (training and test frames of the same sequence); report them only as non-feed-forward references.
* **SERV-CT** (16 stereo pairs from two ex vivo porcine torsos, CT-derived reference disparity; reference accuracy about 2 px RMS and 2 mm): true reference, tiny, and its own
  reference noise is comparable to our EPE. Useful as a second zero-shot depth check; do not over-read differences below its noise.
* Datasets 4 and 5 of SCARED are excluded (17-22 px vertical misalignment not described by the calibration); disclose it and the fact that 2 of 7 training sequences are unused.

## 5. Experiments still needed (priority order)

1. Confirmatory run: sealed 64 and `final_test` (datasets 8, 9) with the frozen configuration and the pre-registered success bar (still unset; B1-vs-C2 undecided).
2. Zero-shot external evaluation (EndoNeRF if stereo pairs exist; SERV-CT) of the SCARED-trained student, refiner and baseline.
3. Stronger baselines: Endo-E2E-GS (retrained and released checkpoint), a stereo-lift feed-forward GS baseline retrained on SCARED (GPS-Gaussian-style or StereoGaussians, if code exists),
   and a general feed-forward GS model zero-shot; a per-scene optimised method as a slow reference.
4. Fair speed: give the baseline the same CUDA-graph treatment, same GPU, same protocol, load average logged.
5. Matched KD arm, A2 seeds and A1b (running now); at least two seeds for the headline comparison.
6. Evaluation beyond reconstructing the two input cameras: the right view with colours taken from the left is the current cross-view test; check whether a held-out viewpoint exists in the data (kinematic poses, temporal neighbours) before claiming novel-view synthesis.

## 6. Risks a reviewer will raise

* "A learned scale calibration of a heuristic Gaussian-size rule": answered only by the A3 evidence and by showing the gain on another backbone or dataset.
* "The baseline was not reproduced faithfully": disclose; show the released checkpoint fails the same way.
* "One dataset": section 5, items 1 and 2.
* "The right view is in the training loss": say so; the claim is cross-view consistency on unseen frames, with colours from the left image only.
* "Application paper": the technical weight must come from C1 (protocol and finding) and C2 (matched control); C3 alone will not carry a CVPR submission.
