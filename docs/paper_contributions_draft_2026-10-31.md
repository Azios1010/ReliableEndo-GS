# Paper contributions: draft wording and claims ledger (2026-10-31)

Status: **DRAFT, development evidence only.** Nothing here is a confirmatory result: no sealed-64 or `final_test` frame has been evaluated, and
every number comes from the 56 benchmark `val_7` frames (one unseen sequence, four keyframes) with one seed per run unless stated.
Items marked **[PENDING]** depend on runs that are queued or not started. Rules for reading the runs are in
`~/.local/share/reliableendo-gs/reports/kd-attribution-20261031/CRITERIA.txt` and `CRITERIA_ACTION.txt` (hashed before the results).
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
  (3.62 vs 3.19 px) and is at least as accurate as the same architecture trained on ground truth (3.56 vs 4.17 px on an unseen sequence). **[PENDING: matched arm]**
* **C3 - Lightweight scale correction.** A 1.7k-parameter, single-pass, feedback-free refiner improves cross-view PSNR by about +1.0 dB on two differently supervised
  students and under input perturbations; the gain comes from the Gaussian-scale residual alone. **[PENDING: matched-KD student, seeds, confirmatory run]**
* **C4 - Speed.** The deployed pipeline runs at 55 FPS (bit-exact graph) to 61 FPS (compiled kernels) at 512x640 with a 0.8 ms marginal cost for the refiner.
  **Only if** the compared baseline receives the same deployment treatment; otherwise state our FPS without a comparison.

## 3. Claims ledger

| ID | Claim | Evidence now | Required before it may be written | Wording allowed today |
| --- | --- | --- | --- | --- |
| C1 | Baseline ignores the stereo signal across sequences | Shift test: baseline slope -0.01 (train) / 0.01 (val_7), student 0.99 / 1.01, teacher 1.00; two baseline checkpoints fail on val_7 (11.3 px released SCARED-C stage 1, 15.5 px retrained) | Same on `final_test` and on one external dataset; baseline reproduction disclosed (native parity not verified, retrained on 4 sequences) | "In our protocol, the baseline we trained and the released checkpoint ..." |
| C2a | Student retains the teacher's accuracy | EPE 3.62 vs 3.19 px, depth MAE 4.37 vs 3.97 mm, 55 frames | `final_test`, external data | descriptive |
| C2b | KD is at least as good as GT supervision | GT-only 4.17 vs KD lineage ckpt 3.56 px (unmatched), KD lower on 49/55 frames; pre-registered verdict "mixed/inconclusive" (r = 1.17 < 1.25) | Matched KD arm; ideally >= 2 seeds | "no worse than" only after the matched arm; never "better" unless r >= 1.25 and w >= 0.70 |
| C2c | KD explains the advantage over the baseline | **Not supported**: GT-only student of our architecture also reaches 4.17 px | - | forbidden |
| C3a | Refiner gain is generic across students | Right-view PSNR +0.99 (10-epoch KD), +1.16 dB (3-epoch GT-only) | Matched-KD student (rule: both 3-epoch students >= +0.5 dB on >= 80% frames) | "on the students tested" |
| C3b | Refiner is stable across training seeds | one seed | A2 (range <= 0.3 dB) | forbidden until A2 |
| C3c | What the refiner fixes | Zeroing scale: +0.99 -> -0.19 dB; zeroing rotation or alpha: unchanged | - | "the gain comes from the scale residual" (not "repairs rotation/opacity") |
| C3d | Robust to input perturbation | 7 conditions, +0.57 to +1.11 dB, positive on >= 95% frames (thinnest: noise 0.05) | - | descriptive |
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
