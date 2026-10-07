# Geometry repair candidate screen

## Scope

The user approved a bounded oracle trial after the stereo-priority pilot
worsened metric geometry. Two deterministic candidates are implemented under
`actions/geometry_repair.py`. They accept only deployment RGB/calibration,
predicted raw disparity and a core mask. GT and teacher outputs are absent
from the action interface. Neither candidate replaces the existing trainable
stereo action or becomes the default selector path.

These are hypotheses about useful repairs, not accepted geometry contributions
or Gate I/II evidence. The first screen freezes the weight-1 stereo checkpoint;
it does not use the degraded weight-3 predictor.

## Shared deployment implementations

- **Correspondence:** search integer offsets ±4 current-grid pixels around
  predicted raw disparity. Score eight local binary grayscale comparisons and
  truncated RGB differences over a 5×5 support window. At subpixel locations,
  interpolation makes descriptor distance a soft L1 cost, not exact Hamming.
  Require cost improvement and a unique minimum. For the same matched right
  query, independently search neighboring left locations; require reverse
  uniqueness and an endpoint within 1 px of the original left position.
  This is a local reverse query check, not a complete global right disparity.
- **Support median:** search color-similar supported neighbors in a 5×5 window,
  excluding the center. Correct only a disparity outlier beyond both 1 px and
  2.5 times the neighbor median absolute deviation. Require at least five
  neighbors and a valid right-image footprint.

Both candidates use the canonical full-right sampler and preserve raw
disparity/principal-point conventions. Changes are bounded by min(4 px,
0.25 × positive effective disparity), and stay inside the requested core and
sensor support. Unsupported, ambiguous and rejected pixels retain their exact
original disparity. Canonical A03 owns metric depth and any future Gaussian
reconstruction; these candidates contain no duplicated triangulation formula.

Thresholds are declared in the two `geometry_*_candidate.yaml` configs before
the trial. They are exploratory settings, not calibrated confidence. Full-grid
search/context is recomputed for each core; no ROI-proportional cost claim.

## Bounded CPU evaluation

```bash
python -m reliable_endo_gs.cli.selection geometry-screen \
  --checkpoint "$RELIABLE_ENDO_GEOMETRY_CHECKPOINT" \
  --data-root "$RELIABLE_ENDO_DATA_ROOT" \
  --manifest "$RELIABLE_ENDO_DEVELOPMENT_MANIFEST" \
  --training-manifest "$RELIABLE_ENDO_TRAIN_MANIFEST" \
  --output-dir "$RELIABLE_ENDO_GEOMETRY_OUTPUT" --per-dataset 2
```

The command chooses two equally spaced identities within each fold's existing
development entries, without GT-based selection. It checks disjointness from
the unchanged train manifest, original SCARED roles 1–7 and checkpoint identity.
No split is changed. It loads the checkpoint's frozen stereo module on CPU,
executes both candidates on four image cores, and only then loads GT for offline
metrics. Local/global metrics use fixed pre-action valid GT denominators;
losing valid geometry is an error. No renderer, Gaussian head, teacher, cache
producer, selector or optimizer is executed.

The offline oracle chooses a candidate only if local depth MAE improves by at
least 0.1 mm; otherwise STOP. This is a **metric depth oracle**, not the RGB/warp
utility oracle used by the selector. It does not evaluate competition with
Gaussian repair and cannot establish deployable action choice.

## Initial 14-frame result

There are 56 cores; 53 have valid GT. Correspondence improves 15 cores by at
least 0.1 mm and worsens one core by that margin. Mean local MAE improvement is
0.0836 mm. Support median has a mean improvement of 0.0320 mm, with no core
crossing either ±0.1 mm threshold. The offline oracle chooses correspondence
on 15 cores and STOP on 41, including the three unavailable GT cores.

This is modest evidence for keeping correspondence as a candidate. Benefits
vary by fold; datasets 4/5 show small mean degradation. It does not justify
replacing the action globally, training a router, or claiming final test
performance. The next useful test is native A03/head/render intervention from
the same frozen state, with RGB/geometry harm and Gaussian repair comparisons.
Training another selector is premature before that comparison.

## Native frozen comparison

`GeometryRepair.execute` now calls the same `geometry_repair_outcome` function
as the established learned stereo action: canonical A03, frozen Gaussian head,
core-only Gaussian merge, then both camera renders. Candidate abstention returns
the actual base field/renders without a numerical re-composition. Existing
learned-action training gradients are preserved; this shortcut applies only to
the deterministic candidate's explicit refusal to change disparity.

The native comparison runs STOP, correspondence, historical learned stereo and
the checkpoint's learned Gaussian repair from the same frozen state. It uses
the CPU report's exact 14 identities and configuration; every region/action
outcome, both-view PSNR/SSIM, geometry metrics, fixed-mask quality components,
harm and action time are retained. GT is loaded only after all interventions.
The RGB/warp proxy oracle and depth-only GT oracle are reported separately.

```bash
python -m reliable_endo_gs.cli.selection geometry-native \
  --checkpoint "$RELIABLE_ENDO_GEOMETRY_CHECKPOINT" \
  --probe-report "$RELIABLE_ENDO_GEOMETRY_PROBE_REPORT" \
  --data-root "$RELIABLE_ENDO_DATA_ROOT" \
  --manifest "$RELIABLE_ENDO_DEVELOPMENT_MANIFEST" \
  --training-manifest "$RELIABLE_ENDO_TRAIN_MANIFEST" \
  --output-dir "$RELIABLE_ENDO_GEOMETRY_NATIVE_OUTPUT"
```

This is a new frozen intervention experiment, not checkpoint resume or
production deployment. The historical model/action snapshot is imported only
when its SHA-256 matches the CPU report and construction/protocol/parameter
shapes match exactly. Historical and current source identities are recorded.
Selector, Adam, RNG, normalization and cost states are not imported. Existing
strict source checks on normal inference/resume remain unchanged.

The entry point requires one visible CUDA GPU and the exact declared native
renderer. It never falls back to a CPU renderer. CPU tests explicitly substitute
an analytic renderer to verify orchestration and GT ordering; they do not
certify native image quality, kernels, timing or memory. No training, teacher
inference, cache generation or router forward occurs in this comparison.
