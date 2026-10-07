# A01 external teacher bindings

The first-party frozen adapters are
`reliable_endo_gs.teachers.external:build_foundation_stereo` (primary candidate)
and `reliable_endo_gs.teachers.external:build_defom_stereo` (independent
comparator). They plug into the existing `a01 teacher-smoke`, `cache`, and
`cache-validate` commands. This implements inference integration, not teacher
fine-tuning or a claim about SCARED accuracy. Real checkpoint compatibility and
CUDA execution still require a separate smoke.

## Verified API boundaries

Bindings follow the official
[FoundationStereo demo](https://github.com/NVlabs/FoundationStereo/blob/master/scripts/run_demo.py)
and [DEFOM-Stereo demo](https://github.com/Insta360-Research-Team/DEFOM-Stereo/blob/main/demo.py):

- Input: rectified A00 RGB `[B,3,H,W]`, floats `[0,1]`. Convert once to RGB
  floats `[0,255]`; the upstream models own normalization.
- Call the upstream `InputPadder(..., divis_by=32)` and unpad the output.
  The adapter does no crop, resize, channel swap, or disparity rescaling.
- FoundationStereo: `model(..., iters=..., test_mode=True)`, or the explicitly
  configured upstream `run_hierachical(..., small_ratio=...)` method.
- DEFOM-Stereo: `model(..., iters=..., scale_iters=..., test_mode=True)`.
- Require floating disparity `[B,1,H_padded,W_padded]`; return FP32 on the
  original A00 grid, in pixels, with `u_left - u_right` sign. No automatic
  tuple extraction, sign correction, checkpoint-format guessing, or fallback.
- Validity is finite positive disparity. Support covers the unpadded input
  grid and is intersected with A00 sensor support by the existing alignment
  layer. This is not an occlusion/confidence estimator. No confidence or
  calibrated risk is invented.
- `LazyTeacherAdapter` owns eval/freeze, device movement and inference mode.
  Only left/right RGB is passed to the model; camera/GT/split labels are not.

## External configuration and artifacts

Runtime templates are `configs/teachers/foundation_stereo.yaml` and
`configs/teachers/defom_stereo.yaml`. Their null pins intentionally fail until
real artifacts are supplied. Keep a filled runtime config outside the repo if
it contains machine-specific paths. For DEFOM, resolve the architecture path
explicitly when copying its template elsewhere.

Set `RELIABLE_ENDO_TEACHER_CONFIG` to the filled YAML. Source, checkpoint and
license paths support `${ENV_VAR}` expansion; relative paths resolve against
the YAML directory. No source/checkpoint is installed, cloned, downloaded,
imported or deserialized by configuration parsing or factory construction.
Factory construction reads/hashes the artifacts. Only explicit `load`/`infer`
imports the external model and deserializes the checkpoint on CPU using
`weights_only=True`, followed by strict state-dict loading. A checkpoint that
requires unrestricted pickle is rejected.

The official FoundationStereo training checkpoint also contains NumPy floating
scalars and OmegaConf configuration metadata. Its ZIP container is scanned for
globals; only explicitly reviewed built-in, NumPy and OmegaConf metadata types
are allowed inside a scoped `torch.serialization.safe_globals` context. Unknown
globals fail before model construction. Only the declared parameter mapping is
retained; optimizer/configuration metadata is discarded. ZIP tensors are memory
mapped on CPU, and legacy tensor-only files keep their weights-only read. This
compatibility path applies to FoundationStereo `checkpoint_format: model`;
DEFOM and raw state dicts retain their restricted read. There is no
`weights_only=False` fallback. Metadata acceptance and successful checkpoint
reading do not establish strict model compatibility or GPU inference readiness.

Required pins:

- `source_revision`: full external Git HEAD, `git -C "$TEACHER_ROOT" rev-parse HEAD`.
- `source_sha256`: relative Python file names/content hashes in sorted order,
  excluding `.git` and `__pycache__`. It includes local Python modifications
  and vendored source rather than claiming Git HEAD alone identifies them.
- SHA-256 of checkpoint, license and architecture YAML. FoundationStereo uses
  the matching checkpoint's `cfg.yaml`; DEFOM's architecture template lists
  official demo defaults and must be checked against the selected checkpoint.
- `checkpoint_format`: `model` for a mapping under `model`, or `state_dict`
  for a verified raw mapping. Prefixes such as `module.` are not stripped.
- Explicit precision and iteration/mode settings. `mixed_precision` in the
  architecture must agree with runtime precision; FP16 requires CUDA.

Compute the source identity without importing a model:

```bash
env PYTHONPATH="$PWD/src" python -c \
  'import os; from pathlib import Path; from reliable_endo_gs.teachers.external import source_tree_sha256; print(source_tree_sha256(Path(os.environ["TEACHER_ROOT"]).resolve()))'
sha256sum "$TEACHER_CHECKPOINT" "$TEACHER_LICENSE" "$TEACHER_ARCHITECTURE"
```

Artifact pins are checked at factory construction and immediately before model
loading. The teacher identity binds these pins, the resolved inference settings
and the first-party adapter's own SHA-256, so they flow into existing cache
keys. Mount paths do not change the scientific identity. Cache generation and
resume/checksum validation are provided by the existing A01 workflow.

## Import namespace and constructor dependencies

Both projects define `core`. Run one candidate per process, with its source
root on `PYTHONPATH` at process startup. The adapter checks loaded `core`
modules and import search locations against the pinned root. It does not edit
`sys.path` or swap imported modules to switch candidates.

Upstream dependencies remain external. The metadata preflight probes a small
set of top-level dependencies; it does not guarantee that every transitive
package or CUDA extension imports. In particular, FoundationStereo's upstream
constructor calls timm with pretrained weights and loads DINOv2 through
`torch.hub`. These may need external caches and may perform network access
even with a checkpoint present. Cache contents and those constructor paths
have not been verified by CPU fake-model tests. Preflight never invokes them.
Prepare/verify those assets separately before a real smoke; this implementation
does not install them or claim offline runtime readiness. Upstream code may
also alter its own import paths; the first-party binding does not copy that
behavior. DEFOM's constructor may read a CWD-relative auxiliary Depth Anything
checkpoint; full strict teacher loading is still required and constructor
behavior must be checked on the selected source revision.

## Commands after configuring real artifacts

Artifact preflight only, with no dataset read, model construction, checkpoint
deserialization, CUDA initialization, or inference:

```bash
export RELIABLE_ENDO_TEACHER_CONFIG=/absolute/external/teacher-runtime.yaml
env PYTHONPATH="$PWD/src:$TEACHER_ROOT" python -m reliable_endo_gs.cli.main \
  a01 teacher-preflight --teacher-config "$RELIABLE_ENDO_TEACHER_CONFIG"
```

The report always distinguishes verified artifacts from unverified runtime
and lists missing directly probed dependencies. Missing/mismatched pins or a
wrong `core` namespace fail with an explicit error.

Future one-sample inference, **not executed as part of this implementation**:

```bash
env PYTHONPATH="$PWD/src:$TEACHER_ROOT" python -m reliable_endo_gs.cli.main \
  a01 teacher-smoke \
  --data-root "$A00_ROOT" --manifest "$A01_MANIFEST" \
  --roles configs/teachers/a01_scared.yaml \
  --teacher-factory reliable_endo_gs.teachers.external:build_foundation_stereo \
  --device cuda:0 --resolution-hw "$RESOLUTION_H" "$RESOLUTION_W"
```

`A00_ROOT` points to processed outputs outside the raw SCARED tree and repo;
`A01_MANIFEST` must index those outputs. Choose the grid explicitly to match
the intended runtime. For DEFOM use its config, source root and
`reliable_endo_gs.teachers.external:build_defom_stereo` in a fresh process.
Existing role policy is unchanged: `dataset_1–7` are `final_train` and cache
eligible; `dataset_8–9` are protected `final_test`. No cache generation,
teacher inference on real SCARED, teacher fitting, A04 or training is run by
the artifact preflight.
