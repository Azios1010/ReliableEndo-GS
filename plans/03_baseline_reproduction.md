# Plan 03 — Baseline Reproduction

## Status

PLANNED

## Research stage

Baseline

## Recommended executor

Sol. Numerical reproduction requires careful source/config parity, metric interpretation, GPU profiling, and diagnosis without contaminating the baseline.

## Estimated implementation risk

MEDIUM. Components are available, but checkpoint, preprocessing, split, renderer, and reference-metric differences can create misleading gaps.

## Objective

Reproduce the pinned Endo-E2E-GS reference on the frozen SCARED protocol, establish left-view quality and right-view diagnostics, profile end-to-end cost, and finalize an immutable baseline artifact.

## Scientific question

Can the official baseline behavior be reproduced reliably enough to serve as the unmodified reference for all later comparisons?

## Why this milestone exists

Phase I gains are uninterpretable without a faithful baseline. This milestone freezes exactly what “baseline” means before uncertainty, covariance, or cross-view training changes enter the system.

## Prerequisites

- Plans 01 and 02 accepted.
- Pinned upstream/checkpoint identity and adapter parity evidence.
- Frozen SCARED split and preprocessing conventions.

## Inputs

- Official or approved upstream checkpoint/config and reference reporting.
- Baseline adapter, production renderer, SCARED tensors, and split manifest.
- Existing runtime/artifact provenance infrastructure.

## Outputs

- Resolved reproduction config and evaluation protocol.
- Original left/self-view metrics plus additional right/cross-view diagnostics that do not change training.
- Stereo/depth, rendering, per-sequence, latency, FPS, and peak-memory summaries as supported.
- Preliminary variability study and a frozen reproduction tolerance.
- Immutable baseline artifact with representative regression outputs.

## Scope

- Match checkpoint, resolution, normalization, crop/resize, calibration, model mode, renderer settings, and metric definitions to upstream.
- Evaluate the original baseline path without research modules.
- Add read-only right-view rendering/diagnostics where camera support exists; label these as added evaluation, not upstream training behavior.
- Report global and per-sequence metrics with valid counts.
- Profile warm-up, synchronization, repetition, batch size, device, precision, latency distribution, FPS, and peak memory.
- First estimate run-to-run/reference variability, then freeze the acceptable reproduction tolerance before declaring pass/fail.

## Non-goals

- No uncertainty signal, calibration, covariance, modified renderer, cross-view loss/training, action, or router.
- Do not tune the baseline on test sequences or weaken its settings for later comparisons.
- Do not invent a tolerance before reference variability and metric equivalence are understood.

## Files expected to be created

- `src/reliable_endo_gs/evaluation/stereo.py`
- `src/reliable_endo_gs/evaluation/depth.py`
- `src/reliable_endo_gs/evaluation/rendering.py`
- `src/reliable_endo_gs/evaluation/aggregation.py`
- `src/reliable_endo_gs/profiling/runtime.py`
- `scripts/reproduce_baseline.py`
- `scripts/evaluate_baseline.py`
- `configs/experiment/p0_baseline_reproduction.yaml`
- `tests/evaluation/test_baseline_metrics.py`
- `tests/profiling/test_runtime_protocol.py`
- `tests/integration/test_baseline_reproduction_smoke.py`

## Files expected to be modified

- `configs/baseline/endo_e2e_gs.yaml`
- Runtime/artifact manifest code to support baseline artifact fields.
- `docs/execution_pipeline.md` or reproduction instructions only with verified details.

## Interfaces and contracts

Evaluation accepts immutable contract outputs and masks; it never calls training or modifies state. Profiling wraps the same end-to-end inference path. Baseline artifact creation requires upstream commit/checkpoint, adapter schema, dataset/split identity, resolved config, metric schema, and hardware profile.

## Scientific formulation

Use upstream loss/metric formulas and standard evaluation definitions, each documented with source and aggregation. No formula here is a proposed contribution. Metric owners live under `evaluation/`; runtime measurement lives under `profiling/`.

## Numerical and coordinate conventions

Use conventions verified in Plans 01-02. Report resolution and valid masks for every metric. Ensure CUDA synchronization for latency, defined warm-up, no hidden autocast difference, deterministic settings where supported, and explicit behavior for empty valid masks.

## Configuration changes

Create `configs/experiment/p0_baseline_reproduction.yaml` selecting frozen data, baseline, evaluation, and profiling settings. Component behavior remains in `configs/data/`, `configs/baseline/`, and planned `configs/evaluation/`; the experiment config contains one comparison, not duplicated logic.

## Tests

### Unit

- Metric formulas, masks, aggregation, empty cases, and per-sequence weighting.
- Profiling synchronization, warm-up, and unit reporting with mocks.

### Contract

- Evaluators accept only supported contract/schema versions and do not mutate inputs.

### Integration

- Dataset -> baseline adapter -> renderer -> metrics -> baseline manifest on a development subset.

### Smoke

- One small production-backend inference/evaluation completes with full provenance.

### Regression

- Representative normalized disparity/Gaussian/render summaries for the pinned checkpoint and config remain within justified tolerances.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/evaluation tests/profiling tests/integration
pytest -q
python -m compileall src
python scripts/reproduce_baseline.py --config configs/experiment/p0_baseline_reproduction.yaml
git diff --check
```

## Experiments

1. Upstream/reference-parity run on the closest matching official protocol.
2. Preliminary repeated-run variability study on validation/development sequences.
3. Frozen SCARED baseline evaluation on the approved split.
4. Read-only right-view diagnostics and end-to-end GPU profiling.

## Metrics

- Original reported left-view rendering metrics where definitions match.
- EPE, D1/bad-pixel, depth MAE/RMSE where supervision supports them.
- Left and diagnostic right PSNR/SSIM/LPIPS where valid.
- Per-sequence valid counts and confidence intervals/descriptive variation.
- Latency mean/median/p90, FPS, peak memory, and model/renderer context.

## Acceptance criteria

- Reproduction configuration and metric definitions are frozen before Phase I changes.
- Baseline reference metrics fall within a tolerance established and documented after preliminary variability analysis, not chosen to force a pass.
- Per-sequence and right-view diagnostics are reproducible and do not alter training.
- End-to-end performance and memory are measured on recorded hardware.
- Immutable baseline artifact validates and can be loaded by identity.

## Failure conditions

- Reference behavior remains outside the pre-frozen tolerance.
- Metric/preprocessing parity cannot be established.
- Baseline artifact omits upstream, checkpoint, split, or hardware identity.

## Pivot / rollback path

Stop Phase I. Diagnose checkpoint, release, preprocessing, split, renderer, and metric differences in that order. Publish a scoped reproduction discrepancy if necessary; do not tune research modules against an unreproduced baseline or redefine the reference silently.

## Artifact/provenance requirements

The baseline artifact records resolved config/hash, project commit/dirty state, upstream commit/patches, checkpoint hash, dataset config and split hashes, contract/metric schemas, seeds, environment, hardware, raw per-sequence metrics, profiling protocol, and frozen tolerance rationale.

## Completion checklist

- [ ] Baseline preprocessing and evaluation match verified upstream behavior.
- [ ] Preliminary variability study freezes the tolerance.
- [ ] Left-view reference and right-view diagnostic results are retained.
- [ ] Runtime and memory are measured end to end.
- [ ] Immutable baseline artifact validates.
- [ ] No Phase I modification entered the path.

## Handoff to next plan

Plan 04 may assume one immutable, reproduced baseline artifact and frozen evaluation/split protocol. It must add uncertainty evidence outside the baseline path and preserve baseline outputs for every comparison.
