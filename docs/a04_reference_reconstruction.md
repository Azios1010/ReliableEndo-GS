# A04 reference reconstruction

The independent frozen benchmark CLI is documented in [a04_benchmark.md](a04_benchmark.md).
It evaluates existing checkpoints without changing the training harness or weights.

The A04 reference path is an independent, quality-first composition:

```text
injected frozen teacher
  -> A01 aligned disparity/validity/support
  -> A00 metric rectified geometry
  -> A03 GeometryPrior and deterministic sampling
  -> bounded reference attribute residual head
  -> GaussianField
  -> injected FieldRenderer
```

`reliable_endo_gs.reconstruction.ReferenceReconstructionModel` owns this
forward boundary. It accepts a live injected teacher or an A01
`AlignedTeacherTarget`, and an injected renderer implementing the independent
`FieldRenderer` contract. FoundationStereo and DEFOM-Stereo therefore use the
same path through A01's neutral `TeacherPrediction`/alignment contracts; no
candidate package or checkpoint is imported by A04.

The final residual projection is zero-initialized, so the initial field is the
A03 priors-only field. Composition uses local-frame right-multiplied
axis-angle rotation, bounded log-scale residuals and bounded alpha-logit
residuals. Invalid primitive masks are propagated separately from opacity;
invalid slots are finite placeholders and are omitted by the renderer.
Numerically sensitive composition is promoted to FP32 for FP16/BF16 inputs.
The A04 forward explicitly disables autocast for A03 geometry, attribute
composition, and renderer dispatch, including native camera projection.
FP32 deployment geometry stays FP32 under the trainer's AMP scope; analytic
FP64 inputs retain their precision. The residual head retains the caller's
AMP policy. CPU FP16/BF16 regression tests check these boundaries and finite
head gradients with a renderer surrogate; they do not establish CUDA parity.

The pinned Graphdeco kernel culls camera depth at `0.2` in native scene units.
The A00 field renderer declares a metric near plane of `0.01 m` and uses
`native_units_per_metre = 0.2 / znear_m` (20 by default). At the native ABI
boundary it converts centres, standard-deviation scales, camera translations
and clip distances by this common factor; packed covariance, when present,
is converted by its square. Intrinsics, rotations, RGB and opacity stay the
same. A00/A03 geometry and the returned GaussianField retain metres, and
autograd includes the conversion. The scene-unit policy is part of the new
renderer identity, so older checkpoints do not silently resume under it.
Historical ProductionRenderer users retain a factor of one by default;
the pinned native kernel and baseline source are unchanged.

A04 smoke and evaluation reject a supervised render with explicit visibility
but no visible primitives. During training, an empty supported field or a
field whose valid centres all precede the renderer's declared metric near
plane consumes the sample without an optimizer update. The manifest ID, skip
reason, depth range, near plane and visibility counts are logged. Other
zero-visibility failures still raise an error. This policy does not change
disparity, calibration, geometry or the near plane, and does not consult GT.
`--max-consecutive-skips 100` stops a systematic failure with explicit evidence;
skip counts remain part of the saved cursor. A black image alone is not a failure:
visible black content is valid. A successful optimizer call does not by itself
prove learning; bounded GPU verification must also confirm nonzero gradients
and a change in the head's initially zero output projection.

`ReferenceProvenance` records teacher identity, target/cache/alignment metadata,
calibration identity, A01 data role, supervision versus assessment views,
renderer identity, and resolved configuration hash. A04 trains only on the
final training role (`dataset_1`–`dataset_7`) and evaluates the final-test role
(`dataset_8`–`dataset_9`). Training rejects the final-test role. SCARED-C and
`dataset_10+` are excluded. Real data and teacher caches remain external.

`save_reference_checkpoint` stores only the independent reference head and
provenance/config identities. Teacher weights and renderer state are explicitly
external. `ReferenceReconstructionModel.profile` delegates to the measured
runtime profiler and accepts synchronization and peak-memory hooks; no latency
optimization or final quality claim is implied by the CPU contract tests.

Injected or cached targets must carry the A00 alignment schema, input-grid
disparity units, disparity sign and calibration metadata. The A04 boundary
detaches those target tensors, requires a resolved teacher identity, and binds
checkpoint loading to the injected runtime teacher identity. Assessment must
include a view held out from supervision; the default trains/observes the left
appearance path and renders both left and right views.

This is engineering implementation only. The final-test loader role is defined,
but the current A04 eval reports reconstruction diagnostics rather than
GT-backed benchmark metrics. Teacher qualification, native renderer parity,
measured GPU timing/memory, reconstruction quality, and the Stage 1 reference
gate remain unverified research evidence.

## Training harness and exact resume

`a04 train` streams flushed JSONL progress to stdout and a new session file.
`--log-file` selects the file; otherwise it is derived from the output
checkpoint with a `.jsonl` suffix. Existing logs are never replaced. Losing
the stdout pipe records `console_disconnected` in the file and leaves training
running. The default
`--log-every 25` reports consumed samples, loss, valid pixels, target source,
successful optimizer updates and the next data cursor. First, last and skipped
samples are also logged. AMP overflow and empty supervision consume a sample
but do not count as optimizer updates.

`--checkpoint-every 500` writes immutable `step-000000500.pt` style files under
`--checkpoint-dir`. Without that option, the directory is derived beside
`--output-checkpoint`. The final checkpoint is also saved on normal completion
or a `--max-steps` stop. SIGINT, SIGTERM and SIGHUP request a stop after the
current sample completes, followed by a final checkpoint. On a Python failure
before an optimizer update begins, a separate recovery checkpoint retains the
last committed sample cursor and its RNG state. A partially executed optimizer
update or an unsafe CUDA context cannot produce a recovery checkpoint; use the
last periodic file. SIGKILL cannot be handled. Resuming an older step into a directory containing
later steps is rejected before training. Use its latest checkpoint or a new
directory. Checkpoint writes retain the existing atomic write/fsync path.

The versioned training state binds the full canonical manifest hash (including
sample order and roles), role, dataset length, completed epochs, next sample
index, optimizer update count, Python/NumPy/Torch CPU and active CUDA RNG state.
Adam and AMP scaler state are retained. A sequential cursor sampler starts
directly at the next index; earlier images are not loaded again. Iterator
creation uses a separate DataLoader generator. Frozen teacher initialization
and inference preserve the head's RNG, including when the teacher is loaded
lazily in a new process. The supported loader remains batch size one, sequential
manifest order and zero workers. No shuffle or split change is introduced.

`--epochs` is the **total** epoch budget, including completed epochs in a
checkpoint. `--max-steps` caps additional successful updates in the current
invocation. An already completed epoch budget is rejected. Legacy checkpoints
with only `optimizer_steps` remain readable for assessment with matching model
identity, but cannot resume training: their sample position is unknown.

Session metadata records Git HEAD/dirty status, a SHA-256 of first-party Python
source (sorted relative filenames, a NUL delimiter, then file contents),
resolved config hash, seed, PyTorch/CUDA versions, GPU and logging/checkpoint
cadence, consecutive-skip limit and resume checkpoint. This metadata is
retained in the session log and checkpoints. Exact
resume requires the same config, teacher and manifest; it does not establish
bitwise CUDA determinism or scientific convergence. Process RNG snapshots do
not capture arbitrary mutable state in a third-party teacher or renderer.

### Deployment data preflight

Before a long run, `a04 preflight` fully decodes every selected stereo pair and
checks its calibration and protocol using CPU workers with bounded memory.
It preserves the manifest order and roles. It does not load GT, construct a
teacher/model or access CUDA; it therefore does not qualify teacher predictions
or prove reconstruction quality. A corrupt/missing file or invalid calibration
produces a FAIL report and a nonzero command exit. Reports are never overwritten.

```bash
PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m reliable_endo_gs.cli.main a04 preflight \
  --config configs/reconstruction/a04_reference.yaml \
  --roles configs/teachers/a01_scared.yaml \
  --data-root "$RELIABLE_ENDO_DATA_ROOT" \
  --manifest "$RELIABLE_ENDO_DATA_ROOT/a01_manifest.json" \
  --data-role final_train --workers 8 --report "$A04_RUN_DIR/data-preflight.json"
```

### Example: one complete epoch on physical GPU 1

Run from the repository in Bash/WSL. Supply external dataset and teacher roots;
the example creates a fresh external run directory. It trains the A04 head
against the frozen teacher. It does not fit the teacher or generate A01 caches.

```bash
export RELIABLE_ENDO_DATA_ROOT=/mnt/d/3DGS_datasets/preprocessing/SCARED_A00
export RELIABLE_ENDO_TEACHER_HOME="$HOME/.local/share/reliableendo-gs/teachers/foundation_stereo"
export RELIABLE_ENDO_TEACHER_CONFIG="$RELIABLE_ENDO_TEACHER_HOME/runtime.yaml"
export PYTHONPATH="$PWD/src:$RELIABLE_ENDO_TEACHER_HOME/source"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HOME/.local/share/reliableendo-gs/runs"
A04_RUN_DIR=$(mktemp -d "$HOME/.local/share/reliableendo-gs/runs/a04-full-XXXXXXXX")
"$RELIABLE_ENDO_TEACHER_HOME/venv/bin/python" -m reliable_endo_gs.cli.main a04 train \
  --config configs/reconstruction/a04_reference.yaml \
  --roles configs/teachers/a01_scared.yaml \
  --data-root "$RELIABLE_ENDO_DATA_ROOT" \
  --manifest "$RELIABLE_ENDO_DATA_ROOT/a01_manifest.json" \
  --teacher-factory reliable_endo_gs.teachers.external:build_foundation_stereo \
  --cache-root "$A04_RUN_DIR/no-cache" \
  --renderer native --device cuda:0 --data-role final_train \
  --epochs 1 --log-every 25 --checkpoint-every 500 \
  --checkpoint-dir "$A04_RUN_DIR/checkpoints" \
  --log-file "$A04_RUN_DIR/train.jsonl" \
  --output-checkpoint "$A04_RUN_DIR/final.pt"
```

To resume, repeat the command with `--checkpoint` pointing to the latest
periodic file, preserve the same total `--epochs` and checkpoint directory,
and choose unused `--log-file` and `--output-checkpoint` filenames. If the
checkpoint already completed the epoch, increase the total epoch budget only
when another epoch is intended. Masking physical GPU 1 as logical `cuda:0`
also avoids the pinned native extension's missing device guard.
For a long server run, launch the command in a dedicated detached tmux session
with stdout/stderr redirected to a new console log, so closing the client
terminal does not terminate the training process.
