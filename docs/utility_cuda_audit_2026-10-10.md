# Utility coefficients and CUDA execution audit — 2026-10-10

## Scope and evidence

This pass audits physical utility preferences and optimizes selector execution.
It does not train the student, repairs, or selector. Original SCARED roles remain
dataset_1–7 TRAIN and dataset_8–9 final TEST. TEST is not used for tuning.

New immutable external results:
`/home/bkai/.local/share/reliableendo-gs/runs/utility-cuda-audit-20261010-Iev07x7s`.
Historical artifacts and source references are preserved separately.

The 5,738-image historical TRAIN outcome table was generated before the GT
half-pixel resampling correction documented in
[`selector_data_audit_2026-10-10.md`](selector_data_audit_2026-10-10.md).
Its coefficient sweep is sensitivity evidence, not a production optimum.
The current parent student trained on the quarter cohort; masked repairs had
only 168 updates on 56 images. Frozen outcome collection does not train them.

## Physical coefficients

Canonical utility remains signed and bounded: each raw metric change is divided
by its declared scale, bounded by `x / (1 + abs(x) / cap)`, and combined by the
canonical metric policy. Raw-unit safety guards are applied independently.
Negative gains retain their sign. Increasing only positive stereo gains would
bias the expected utility and is not implemented.

Twenty-four common policies are compared: geometry priority
0.5/1/1.5/2/3/4 and local geometry weight 0.25/0.5/1/2, plus the historical
stereo action multiplier 1.5 comparator. Common candidates use action multiplier
1.0. FIT, validation, and calibration use disjoint complete TRAIN keyframes.
Safety tolerances and risk cutoffs are retained. Action frequency is not the
objective. The analysis is a **conditional quality oracle** over measured,
physically safe interventions, without costs or predicted risk; it is not a
learned policy evaluation or a matched-cost benchmark.

Historical validation keyframe macro means (1,506 images with known labels):

| Geometry priority; local weight 1 | Global MAE gain (mm) | Local MAE gain (mm) | Left PSNR gain (dB) | Right PSNR gain (dB) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.002473 | 0.142507 | 0.018082 | 0.013104 |
| 1.5 | 0.003258 | 0.186097 | 0.016558 | 0.012154 |
| 2 | 0.003496 | 0.199835 | 0.015706 | 0.011570 |
| 4 | 0.003800 | 0.218655 | 0.013621 | 0.010031 |

Geometry priority 1.5–2 is a candidate to verify, not a selected optimum.
Beyond that range, historical depth gains diminish while render gains fall.
The analysis reports 500 keyframe cluster bootstrap draws and raw-unit changes;
these intervals do not remove the old GT alignment limitation.

Historical FIT stereo absolute median changes are 0.002560616 mm global MAE
and 0.208444595 mm local MAE, approximately an 81-fold difference. A small
repair region naturally has a small global effect. Multiplying the global term
by an arbitrary 1,000 would change scientific preferences. New analysis reports
FIT-only per-action scale diagnostics instead of silently changing scales.

**Measurement limitation:** current depth metrics use stereo disparity.
Gaussian repair leaves that disparity unchanged and therefore has exactly zero
stereo-depth gain. Rendered Gaussian depth is unavailable in this evaluation.
This prevents a claim about Gaussian geometric quality from these depth metrics.

### Corrected TRAIN outcome results and next pilot

The bounded native audit finished with exit code **0**. All 108 images were
collected with `nearest_exact_half_pixel_centres` explicitly recorded.
There were 116 regions, 96 known regions, and 20 unknown regions across 16
images without a complete repair label. Unknowns remain excluded from this
conditional oracle; they are not counted as safe STOP decisions.

Corrected validation comprises 27 observed images in seven held-out TRAIN
keyframes. Equal-keyframe macro means are:

| Geometry priority; local weight 1, action multiplier 1 | Global MAE gain (mm) | Local MAE gain (mm) | Left PSNR gain (dB) | Right PSNR gain (dB) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.0018811 | 0.1002149 | 0.0111645 | 0.0082979 |
| 1.5 | 0.0032827 | 0.1764304 | 0.0079553 | 0.0075588 |
| 2 | 0.0037520 | 0.1915454 | 0.0074824 | 0.0071949 |
| 4 | 0.0038035 | 0.1958707 | 0.0068369 | 0.0070117 |

The recommended **next pilot** is common geometry priority **1.5**, global/local
geometry weights **1/1**, PSNR/SSIM weights **0.5/0.5**, and signed stereo action
multiplier **1.0**. It reaches much of the measured geometric gain while retaining
positive render gains. Moving from 1.5 to 2 adds 8.6% local depth gain and reduces
left/right render gain by about 5.9%/4.8%. Confidence intervals overlap widely:
at priority 1.5, local MAE gain CI is [0.03954, 0.40370] mm and left PSNR gain CI
is [0.00220, 0.01470] dB. This is a preference for a conservative pilot, not proof
of an optimum or a learned selector's performance.

`configs/routing/intervention_metric_common_utility.yaml` declares this pilot.
It preserves the existing selector-fit optimization settings and physical/risk
guards. Both gain and risk heads may learn; gain-only refinement is false.
The historical input artifact used gain-only refinement, so this pilot requires
a new run with model-only initialization and fresh labels/normalization. It
cannot resume the historical full selector/optimizer state. No default or
historical configuration is overwritten. The configuration has not been trained.

Corrected FIT medians are 0.004103 mm global and 0.284755 mm local absolute stereo
MAE changes, approximately a 69-fold difference. PSNR/SSIM reference scales and
the stereo training priority 3 remain declared historical settings, not newly
optimized coefficients. Local/collateral RGB remain raw-unit safety guards with
zero direct utility weight. The audit does not claim every coefficient or risk
cutoff has been optimized.

Paired observations match all 116 region IDs and boxes with the historical table
(exact historical write indices are unavailable). Among 96 jointly known regions,
STEREO has 31 harmful outcomes before and after, with two individual label flips;
GAUSSIAN has 10 before and after, with no flips. The alignment correction does
not explain away the repair reliability problem. Broader repair learning and a
new learned-policy validation remain required before long training or deployment
claims based on these coefficients.

## Execution changes

- Reuse gain and risk logits: one normalization and one risk-network forward
  for each calibrated prediction and selector training objective.
- Replace dynamic Boolean tensor compaction for context statistics with fixed
  shape masked reductions. Keep a centred two-pass population variance and
  explicit availability/empty-support semantics.
- Precompute scale, anisotropy, and log-scale arrays once per frame.
- Transfer allocator validation and winning index/gain/cost in one packet.
  Preserve STOP, strict minimum gain, risk support, budgets, and stable ties.
- Add an immutable paired native component profiler. It checks feature parity,
  bitwise identical predictions on identical features, unchanged weights, and
  restores RNG. It reports synchronized wall times separately from CUDA events.
  These are component measurements, not full-pipeline FPS.
- Add collection-only mode that stops before selector fitting, evaluation,
  deployment smoke, or training. Offline GT logs record the actual loader's
  resampling identity separately from deployment inputs.

CUDA measurement uses physical GPU0, an RTX 5070 Ti, with
`CUDA_LAUNCH_BLOCKING` unset. Costs from the historical blocking runner are not
reused. GPU1 and other users' sessions are left untouched.

## Corrected cohort and validation status

The diagnostic cohort has 108 original TRAIN images: four temporal IDs per
each of 27 keyframes, selected before reading GT or action outcomes. Existing
RGB-reviewed masks are reused. Original manifests, parent roles, data files,
and checkpoints remain unchanged. The 112 original development identities
are retained for RGB/calibration preflight; no development GT outcome is used.
The source snapshot is a sealed subset of the original content-hashed snapshot.

Native component profiling completed on
`dataset_1/keyframe_1/frame_data000003` (one eligible mask).
Three warmups and ten alternating paired repetitions were used:

| Component | Before median synchronized wall (ms) | After (ms) | Median reduction |
| --- | ---: | ---: | ---: |
| Context | 15.19533 | 12.75561 | 16.06% |
| Calibrated head | 0.57501 | 0.354915 | 38.28% |

All ten pairs improved for both components. Context timings varied considerably
(one historical-path wall stall was 177.7 ms); the median paired after/before
ratio was 0.82094 for context and 0.65079 for the head. Raw wall/CUDA-event
observations are retained in `corrected-cohort/routing_profile.json`; the two
clock domains are not substituted for each other. These measurements cover one
frame and mask, not arbitrary region counts, training throughput, or full FPS.
Allocator speed has not been measured separately.

Feature maximum absolute difference was `7.62939453125e-06`, within the declared
`atol=2e-5, rtol=2e-5` parity gate. Availability bits were exact and head
predictions on identical features were bitwise identical. Model/action/head
weights were verified unchanged. Process peak allocated CUDA memory through
profiling was 589.84 MiB; driver memory and subsequent collection peaks differ.
Runtime: Torch 2.7.1+cu128, CUDA 12.8, cuDNN 90701, compute capability 12.0,
physical GPU0 RTX 5070 Ti, WSL2 Linux 6.18.40.1. Blocking was unset.

Focused synthetic validation passed:
36 tests across `test_routing_execution`, `test_utility_sensitivity`,
`test_selector_refit`, `test_metric_joint`, and `test_selector_cohort`.
The runtime used GPU0 for CUDA-capable focused tests and CPU for the independent
scalar/statistics tests. Pytest reported 369.44 seconds, excluding initial
runtime startup. Both new/extended CLI `--help` paths pass.
Two utility/config tests then passed in 20.17 seconds (one existing test repeated,
one pilot-config test added), covering parsing, a distinct config identity,
unchanged physical guards, and full gain/risk learning eligibility.
Ruff and mypy (12 affected source files) pass. `git diff --check` passes.
CUDA allocation, kernel execution, and backward were verified separately on
GPU0. Startup showed slow NVRTC loading; later traces showed Python filesystem
`stat` during imports. Neither startup delay is a measured selector latency.
The profiler's `peak_cuda_allocated_bytes` is the process peak since CUDA
initialization, including cost preparation; it is not isolated component VRAM.

No full training, teacher inference, cache generation, TEST tuning, dependency
installation, or dataset relocation is performed in this pass.

## Changed files in this pass

Execution: `routing/balanced.py`, `routing/calibrated_selector.py`,
`routing/context.py`, `routing/repairability.py`, and
`training/selector_fitting.py` under `src/reliable_endo_gs`.

Analysis/profiling: `evaluation/utility_sensitivity.py`,
`profiling/routing_components.py`, and `cli/utility_audit.py` (new).

Bounded collection/provenance: `cli/selector_refit.py`,
`training/selector_refit.py`, `training/selector_outcomes.py`,
`training/metric_quality.py`, and `data/scared_a00.py`.

Tests: `tests/routing/test_routing_execution.py` and
`tests/evaluation/test_utility_sensitivity.py` (new). This report is new.
The common-utility pilot YAML is new.
Other existing working-tree changes, including the previous data audit, are
preserved; the whole Git diff is not attributable to this pass.

First-party Python source SHA-256 used by native measurements:
`b9223d6e45439168073a7389f1d6c97f20896fc3e8edcb883fdb5b6a3f194812`.
Frozen input checkpoint:
`01ee7a0d065127bb50f996e1d9247610e7826c6da54c9bbdcddc3a02679c32c3`.
New cost artifact:
`cbc40fb45c6d21db7078b76702ff5a0553c03e09caffe8ade831aebc88136c55`.
External worklist:
`500416f3fe051c6cc57448dc03693925b7bcbd5eb8eb7e890882f2d1a31d75d4`.
Partition:
`e874fb6ac47452574e296354de9da223ac24e3f7bcac247a5fcca663b48f7792`.
Subset source snapshot:
`f7a64e424509763ef11d8eb26baa51659c535c2a0fabbd915e7aebf621c0bfcb`.
The full preflight, original snapshot parent hashes, reference file hashes,
runtime identity, and checkpoint/mask identities are retained externally.
Corrected outcome table SHA-256:
`832eb843a357eba4c128544553e73ae6a1abf0587af4508bc69ea01a40d766fe`.
Analysis module SHA-256:
`54b17eed2949d4b27cab0778d5fdd08b9f6d37e4415fed3f42de77c6bf333c2c`.
