# Local implementation handoff: student and joint learning

This file retains the handoff snapshot and its historical evidence. The next
local student/joint implementation is now documented in
[Student and joint learning](student_joint_learning.md); statements below about
the student being unimplemented describe the handoff snapshot, not current code.

## Authority and task boundary

Read this file first, then `AGENTS.md` and the active sections of `PLAN.md`.
The user approved this schedule on 2026-10-05:

1. Keep the existing teacher-backed A04 reference fixed and learn Phase II
   repair heads, then a small action selector with a scientific guiding harness.
2. Subsequently implement a student and joint learning of the two phases.

Step 1 is implemented and its full run is in progress at the handoff snapshot.
Step 2 is the intended next local implementation. It is **not implemented**.
The explicit user instruction takes precedence over the historical gate order;
neither Gate I nor Gate II is certified. Keep the existing frozen experiment,
baseline, checkpoints and evidence reproducible as separate modes.
Some ownership/status tables in older documentation describe scaffolds and
historical SCARED-C experiments. They do not describe the active executable path.

## Actual current implementation

```text
A00 deployment stereo RGB + rectified metric camera + sensor support
 -> frozen FoundationStereo disparity (live inference or aligned A01 target)
 -> canonical A03 geometry, surface priors and source-index sampling
 -> A04 attribute MLP (13 -> 128 -> 128 -> 7)
 -> GaussianField and native full-frame left/right rendering
 -> fixed regions and 18 pre-action features
 -> utility MLP (18 -> 32 -> 32 -> 2; STOP utility = 0)
 -> STOP / stereo residual repair / Gaussian attribute repair
```

- A04's original training objective is masked **left** RGB Charbonnier loss.
- Stereo repair is a pointwise convolution `10 -> 32 -> 32 -> 1`: bounded
  pixel-disparity residual, canonical A03 recomputation, frozen A04 attributes.
- Gaussian repair is an MLP `13 -> 32 -> 32 -> 7`: rotation/log-scale/opacity
  residuals; centers and disparity stay fixed.
- Both repair implementations are shared by learning, intervention collection
  and selected inference. Repairs currently render both complete frames.
- The scientific harness runs each alternative from the same base with fixed
  pre-action masks. Utility is the decrease in mean left/right RGB Charbonnier
  loss minus `cost_weight_per_ms * measured_action_ms`. STOP utility is zero.
- The selector learns the two non-STOP utilities with MSE. It has no GT,
  post-action labels or invented uncertainty features in deployment input.
- Deployment selects at most one core/action per frame, using positive
  predicted utility and estimated cost. The cost budget is not a hard bound.
- The current schedule trains repair heads for one epoch, then fixes them and
  trains the selector for one epoch. FoundationStereo and A04 stay fixed.
- No student, KD, joint trainer or RL is implemented. Calibrated uncertainty
  is not connected to this active deterministic A04 experiment, although
  historical uncertainty packages exist elsewhere in the repository.

## Reading order and code map

| Area | Read these files |
| --- | --- |
| Governance and active policy | `AGENTS.md`, `PLAN.md`, `docs/scientific_integrity.md` |
| Active architecture and experimental limits | `docs/frozen_a04_action_learning.md`, `configs/routing/frozen_a04.yaml` |
| Deployment and camera contracts | `docs/a00_deployment_contract.md`, `docs/a00_scared_protocol.md`, `src/reliable_endo_gs/contracts/deployment.py` |
| Dataset identity and roles | `docs/scared_multi_sequence_contract.md`, `configs/teachers/a01_scared.yaml`, `src/reliable_endo_gs/data/scared_a00.py`, `src/reliable_endo_gs/data/scared_a00_identity.py` |
| External teacher and target/cache | `docs/a01_external_teachers.md`, `src/reliable_endo_gs/teachers/external.py`, `src/reliable_endo_gs/teachers/base.py`, `src/reliable_endo_gs/teachers/targets.py`, `src/reliable_endo_gs/teachers/workflow.py` |
| Canonical geometry and Gaussian construction | `docs/a03_geometry_gaussian_priors.md`, `src/reliable_endo_gs/geometry/rectified.py`, `src/reliable_endo_gs/geometry/surface.py`, `src/reliable_endo_gs/gaussians/priors.py`, `src/reliable_endo_gs/gaussians/composition.py`, `src/reliable_endo_gs/gaussians/residual_head.py` |
| Reference forward/config/checkpoints | `docs/a04_reference_reconstruction.md`, `configs/reconstruction/a04_reference.yaml`, `src/reliable_endo_gs/reconstruction/model.py`, `src/reliable_endo_gs/reconstruction/config.py`, `src/reliable_endo_gs/reconstruction/checkpoint.py`, `src/reliable_endo_gs/training/a04_reference.py` |
| Actual actions and regions | `src/reliable_endo_gs/actions/repair.py`, `src/reliable_endo_gs/regions/grid.py` |
| Selector and utility supervision | `src/reliable_endo_gs/routing/selector.py`, `src/reliable_endo_gs/routing/config.py`, `src/reliable_endo_gs/training/interventions.py` |
| Phase II optimizer/run/recovery | `src/reliable_endo_gs/training/action_learning.py`, `src/reliable_endo_gs/training/action_run.py`, `src/reliable_endo_gs/training/action_checkpoint.py`, `src/reliable_endo_gs/cli/phase2.py` |
| First-party native rendering | `src/reliable_endo_gs/rendering/production.py`, `src/reliable_endo_gs/rendering/field.py` |
| Independent benchmark harness | `docs/a04_benchmark.md`, `configs/evaluation/a04_benchmark.yaml`, `src/reliable_endo_gs/evaluation/a04_benchmark.py` |
| Focused tests and synthetic fixtures | `tests/routing/test_frozen_a04.py`, `tests/architecture/test_a04_reference_reconstruction.py`, `tests/integration/test_a04_training_recovery.py`, `tests/rendering/test_metric_native_boundary.py` |
| Historical proposal implementation context | `docs/architecture.md`, `plans/05_phase1_learned_uncertainty_and_calibration.md`, `plans/06_phase1_geometry_uncertainty_propagation.md`, `plans/07_phase1_probabilistic_gaussian_representation.md`, `plans/10_phase2_regions_and_repair_actions.md`, `plans/11_phase2_oracle_intervention_collection.md`, `plans/13_phase2_router_and_budget_allocator.md` |

The benchmark harness measures evaluation metrics. The intervention harness
creates supervision for the utility MLP. Neither is the student architecture.

## Gradient boundaries to inspect before implementing joint learning

Simply changing `requires_grad` is insufficient:

- `ReferenceReconstructionModel` freezes its injected teacher and performs
  teacher target inference without gradients. The teacher is deliberately
  not registered as a trainable reconstruction submodule.
- `frozen_state` in `training/action_run.py` runs the reference under no-grad.
- `ActionLearner` fixes the reference; `RepairActions.execute` rejects a
  trainable reference in the current frozen mode.
- Selector features and measured utility targets are detached. Selection is
  discrete; utility MSE alone does not train an upstream student.
- Primitive sampling, invalid masks, action/core selection and detached
  disparity bounds need explicit gradient semantics and tests.

Add an explicit student/joint mode with clear parameter ownership and a
separate checkpoint/config contract. Preserve the frozen mode and its strict
identity checks. Keep scientific formulas canonical and actions reusable;
Phase I source must not start importing routing or Phase II losses.
Student architecture, KD terms, joint loss weights, discrete-selection gradient
strategy and model-selection protocol remain design decisions, not approved
implementations. Document these choices before making scientific claims.
The user has not requested permanent student freezing. Teacher fine-tuning is
a separate possible experiment, not a prerequisite for a trainable student.

## Data and external assets

Active data is **original SCARED only**:

- `final_train`: dataset_1 through dataset_7, 22,950 manifest entries.
- `final_test`: dataset_8 and dataset_9, 5,907 manifest entries.
- No SCARED-C, no tuning/model selection on final_test, no invented split.
- Raw and processed medical data, manifests generated from it, weights and
  training outputs remain external. Git transfers source/config/docs/tests.
- GT may be used only through explicitly separate supervision/evaluation
  interfaces; never place it in `DeploymentStereoBatch` or selector features.

Configure local external paths; server filesystem paths are not portable:

```bash
export RELIABLE_ENDO_DATA_ROOT=/absolute/path/to/processed/SCARED_A00
export RELIABLE_ENDO_TEACHER_CONFIG=/absolute/path/to/teacher/runtime.yaml
export RELIABLE_ENDO_TEACHER_SOURCE=/absolute/path/to/FoundationStereo/source
export A04_REFERENCE_CHECKPOINT=/absolute/path/to/frozen/reference/final.pt
export PYTHONPATH="$PWD/src:$RELIABLE_ENDO_TEACHER_SOURCE"
```

For CPU synthetic implementation work, real SCARED, FoundationStereo weights
and CUDA are not required. For real execution, additionally obtain the
external processed manifest/protocol provenance, exact frozen A04 artifact,
teacher runtime/config/source/checkpoint and compiled native rasterizer.
These assets are not supplied by `git pull`; no dependencies are installed
by this handoff. Native GPU runs use one visible physical GPU as `cuda:0`.

Verified server artifact identities at handoff:

| Identity | SHA-256 / immutable revision |
| --- | --- |
| Frozen A04 checkpoint | `96dfb96929141dbd96fbc52338ec2332e99729ced27252583daef66d2d440f22` |
| Full A01 manifest | `02af69f731075da7c8b8a019ea8422a5babbe4d0d6144dbed4e98bd0b753c482` |
| Active roles | `8f6752e322b2642734ebfc9c3c8f30a93c45e74773e1dc02a67bbc8debf75109` |
| FoundationStereo checkpoint | `60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1` |
| FoundationStereo source revision | `6e8806816b533e4d13ddbb95ffa907b797060a62` |
| Native rasterizer revision | `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d` |

## Evidence and validation boundaries

Earlier implementation validation reported 74 focused tests passing, a
one-sample real GPU smoke/save-resume check and a 14-sample GPU trial across
training folds. The trial trained both repairs and selector. It did not
establish useful selection: one selected action had negative measured
utility after its runtime penalty. Full Phase II training and held-out
benchmark results were not complete when this handoff was prepared.
Tests are not rerun solely for this documentation/snapshot handoff.

Useful local commands once the existing development environment is available:

```bash
PYTHONPATH=src python -m reliable_endo_gs.cli.phase2 --help
PYTHONPATH=src python -m pytest -q tests/routing/test_frozen_a04.py
PYTHONPATH=src python -m pytest -q tests/architecture/test_a04_reference_reconstruction.py
git diff --check
```

For a new student/joint implementation, add synthetic tests for optimizer
ownership, nonzero student gradients, unchanged teacher/frozen reference,
GT exclusion, reused action semantics, zero-action identity, checkpoint
round-trip/resume, config compatibility and evaluation state immutability.
GPU validation is a separate bounded step, not a long training default.

## Git/submodule and running-job caveats

The active server main checkout has uncommitted implementation changes and
an Endo-E2E-GS submodule locally advanced from the parent's pinned
`186fa2b4a2159b28393492f6df1aa444b54391a8` to
`170659adcd766650ced93f264e765fc85976efd0`. The latter includes the historical
scale-gradient baseline change; it is not required by the first-party
A00/A04/Phase II path. This handoff retains the parent's original upstream
pointer and leaves the server submodule untouched. Do not silently present
the locally patched baseline as unmodified upstream reproduction.

Phase II checkpoint loading compares its entire identity, including Git
commit/dirty state, source hash and hardware identity. A new Git commit or
different local GPU can invalidate a server checkpoint for strict resume.
Do not edit checkpoint metadata or weaken checks to make it load. Preserve
the recorded original checkout/artifacts; any deliberate artifact migration
needs an explicit, validated policy. The push snapshot is prepared in a
separate worktree so the current running checkout keeps its HEAD and source.

## Suggested instruction for the local implementation agent

> Read `docs/local_student_joint_handoff.md`, `AGENTS.md`, the active sections
> of `PLAN.md`, and the code map above. Implement the next user-approved
> stage: a trainable student and explicit joint learning of Phase I student
> reconstruction and Phase II repairs/selection. First report the proposed
> student contract, gradient paths, optimizer ownership and loss schedule.
> Preserve the existing frozen teacher/A04 experiment and its evidence.
> Keep train dataset_1–7/test dataset_8–9, GT separation and canonical geometry.
> Reuse real action implementations. Add focused synthetic tests, validate
> gradients and checkpoint recovery, and report implemented/tested/planned
> separately. Do not launch long training, download data or replace the
> running server experiment as part of implementation.

## CPU CI follow-up

The handoff branch required full-repository validation beyond the earlier
focused checks. Repairs include Ruff formatting, preprocessing JSON typing,
scoped SciPy stub handling, an explicit OmegaConf development dependency for
serialized checkpoint-metadata tests, and `PYTHONDONTWRITEBYTECODE=1` in CI to
keep the pinned upstream clean when its CPU helpers are imported.

The A00 processors have a newly pinned typing/formatting hash pair. A review
of the before/after executable AST, excluding annotations, typing imports and
identity `typing.cast` calls, found identical runtime logic. Compatibility
accepts only the exact reviewed pairs, retains historical metric gates and
never rewrites dataset provenance. Synthetic resume/preflight tests cover it.

Local full-suite validation: 635 passed, 18 skipped (16 unavailable optional
upstream diagnostics and two unavailable historical real-data fixtures). Lint,
formatting, mypy over 159 source files, infrastructure smoke, Phase II CLI help
and compilation passed. No real training/inference or dependency installation
was performed in this repair. The existing development tools were used; the
full test command also exposed the already installed teacher environment's
site-packages to obtain OmegaConf. An initial test attempt with no visible GPU
aborted inside this server's CUDA PyTorch CPU backward; the successful run
kept GPU 0 visible while exercising CPU synthetic tests. Hosted CI was not
retrieved because the GitHub connector was disconnected and the repository
was unavailable through the anonymous API.

Formatting also changes the external-teacher wrapper's file hash and therefore
its recorded API identity. Existing A04/cache/Phase II artifacts remain tied
to their original identities; do not relabel them or disable compatibility
checks. Running the old frozen artifact on this new source requires an explicit
validated migration policy, which this CI repair does not implement. The
original server checkout and running experiment keep their original source.
