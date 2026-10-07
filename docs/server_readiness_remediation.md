# Server-readiness remediation

The active executable path is original SCARED dataset_1 through dataset_9.
Dataset_1–7 form `final_train`; dataset_8–9 form the protected `final_test`.
The data root is external and must be resolved through `RELIABLE_ENDO_DATA_ROOT`;
SCARED-C and dataset_10+ are rejected by the A01 manifest builder.

The active role mapping is `dataset_1`–`dataset_7` as `final_train` and
`dataset_8`–`dataset_9` as `final_test`, from
`configs/teachers/a01_scared.yaml`. The earlier `teacher_fit`/`qualification`
development split and historical SCARED-C / dataset_6 holdout records are not
the final train/test policy.

## A00 original-SCARED sample preparation

The first-party single-sample processor is configured by
`configs/data/a00_scared_sample_v1.yaml`. It uses OpenCV
`CALIB_ZERO_DISPARITY`, `alpha=0.0`, `CV_32FC1` maps, linear image remapping,
integer pixel centres, no crop/resize, and requires all output pixels to have
source support. It writes the processed sample only to an external output
root. It reads the top/bottom halves of the video and scene-point TIFF according
to the declared input schema and verifies that the selected archive IDs match.

The inspected external preprocessing candidate `MICCAI_challenge_preprocess/stereo_rectify.py` is not the
script identified by the historical processing record: its SHA-256 is
`90abb8d17b088fa54b36b3e64dc55ea184450657008a21f6d66559c0e91b9623`, while the
historical record names
`ABE34F324870CE8DDEB8729FFD796285106B4B810576D80FFC3BB3D526D45AB6`. The
first-party processor records its own exact code/config hashes for any output;
it does not claim either historical identity produced that output.

The selected sample's sibling `endoscope_calibration.yaml` has SHA-256
`597878f105e38ff2c2b7973d2c3a6af31293b904ca3b73bbe0a5f239466e6e9e`; its R/T
match `frame_data000000.json`. The config pins it as the numeric-value source
only. The inspected community preprocessing candidate
`MICCAI_challenge_preprocess/stereo_rectify.py` (commit
`6493ecf66b953fcd5387004ecd48d3c345fd9ac3`, SHA-256
`90abb8d17b088fa54b36b3e64dc55ea184450657008a21f6d66559c0e91b9623`) passes
R/T to `cv2.stereoRectify`, but no provenance links that community code to the
official calibration producer or outputs; neither it nor OpenCV serialization
is treated as direction authority. The YAML does not state translation units.
`right_from_left` and `mm` are selected empirically from matched left/right
scene-point clouds against the official metric SCARED GT scale. The measured
left-to-right candidate must beat the opposite-use direction control and cm/m
scale controls under configured thresholds. `a00 prepare` verifies the pinned
YAML hash and exact R/T match, then recomputes these gates. This remediation did
not create an output sidecar or run A00 smoke.

When intentionally preparing the selected sample, use these bounded commands:

```bash
export RELIABLE_ENDO_DATA_ROOT=/path/to/external/datasets
SCARED_ROOT="$RELIABLE_ENDO_DATA_ROOT/SCARED"
A00_ROOT="$RELIABLE_ENDO_DATA_ROOT/SCARED_A00"
# Selected sample output: "$A00_ROOT/dataset_1/keyframe_1/data"

conda run -n endo_e2e_gs env PYTHONPATH="$PWD/src" python -m reliable_endo_gs.cli.main a00 prepare \
  --data-root "$SCARED_ROOT" \
  --sample-id dataset_1/keyframe_1/frame_data000000 \
  --output-root "$A00_ROOT" \
  --config configs/data/a00_scared_sample_v1.yaml \
  --calibration-evidence "$RELIABLE_ENDO_DATA_ROOT/calibration_evidence.json"

conda run -n endo_e2e_gs env PYTHONPATH="$PWD/src" python -m reliable_endo_gs.cli.main a00 preflight \
  --data-root "$A00_ROOT" \
  --sample-id dataset_1/keyframe_1/frame_data000000 \
  --roles configs/teachers/a01_scared.yaml
```

The preflight hashes the recorded external calibration-evidence JSON, checks
its pinned direction-source and raw-member identities against provenance, parses
the complete `A00Protocol` schema, and checks the output files. It does not decode
images or call the A00 loader. It prints a GT-free `a00 smoke` command with the
active Python executable and checkout `src` path. The smoke is the separate
call that loads the A00 deployment batch; do not run it automatically. A01
teacher/cache, A04, renderer, and training commands remain separate and are
outside this remediation.

Each processed SCARED keyframe used by A00 must carry the protocol described
in [the A00 protocol](a00_scared_protocol.md), either as the keyframe sidecar
or as the documented embedded object. The protocol certifies image
range, rectification, disparity sign/units, extrinsic direction/units and
the preprocessing grid. A01 cache generation and A04 execution use the same
`--resolution-hw` values so cache alignment and reference input geometry have
the same identity.

The commands below are bounded entry points. First-party teacher factories are
now available; configure the external source/checkpoint and artifact pins using
[A01 external teacher bindings](a01_external_teachers.md), then run the metadata
preflight before scheduling a separate real teacher smoke. The model runtime
has not been verified by the first-party CPU adapter tests.

```bash
export RELIABLE_ENDO_DATA_ROOT=/mnt/data
DATA_ROOT="$RELIABLE_ENDO_DATA_ROOT/preprocessing/SCARED_A00"
MANIFEST=artifacts/manifests/a01_scared_1_9.json
CACHE=artifacts/targets/a01
TEACHER_FACTORY=reliable_endo_gs.teachers.external:build_foundation_stereo
# Set RELIABLE_ENDO_TEACHER_CONFIG and add the selected external teacher source
# root to PYTHONPATH as documented in a01_external_teachers.md.
RESOLUTION_H=512
RESOLUTION_W=640

python -m reliable_endo_gs.cli.main data a01-index \
  --data-root "$DATA_ROOT" --output "$MANIFEST"

python -m reliable_endo_gs.cli.main a01 teacher-smoke \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" \
  --teacher-factory "$TEACHER_FACTORY" --device cuda:0 \
  --resolution-hw "$RESOLUTION_H" "$RESOLUTION_W"

python -m reliable_endo_gs.cli.main a01 cache \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" --cache-root "$CACHE" \
  --teacher-factory "$TEACHER_FACTORY" --device cuda:0 \
  --resolution-hw "$RESOLUTION_H" "$RESOLUTION_W"

python -m reliable_endo_gs.cli.main a01 cache-validate \
  --cache-root "$CACHE" --manifest "$MANIFEST" --data-root "$DATA_ROOT" \
  --teacher-factory "$TEACHER_FACTORY" --device cuda:0 \
  --resolution-hw "$RESOLUTION_H" "$RESOLUTION_W"

python -m reliable_endo_gs.cli.main a04 smoke \
  --config configs/reconstruction/a04_reference.yaml \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" --cache-root "$CACHE" \
  --data-role final_train \
  --teacher-factory "$TEACHER_FACTORY" --renderer native --device cuda:0

python -m reliable_endo_gs.cli.main a04 train \
  --config configs/reconstruction/a04_reference.yaml \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" --cache-root "$CACHE" \
  --data-role final_train \
  --teacher-factory "$TEACHER_FACTORY" --renderer native --device cuda:0 \
  --max-steps 1 --output-checkpoint artifacts/a04_reference/server-smoke.pt

python -m reliable_endo_gs.cli.main a04 eval \
  --config configs/reconstruction/a04_reference.yaml \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" --cache-root "$CACHE" \
  --data-role final_test \
  --teacher-factory "$TEACHER_FACTORY" --renderer native --device cuda:0 \
  --checkpoint artifacts/a04_reference/server-smoke.pt

python -m reliable_endo_gs.cli.main a04 profile \
  --config configs/reconstruction/a04_reference.yaml \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" --cache-root "$CACHE" \
  --data-role final_test \
  --teacher-factory "$TEACHER_FACTORY" --renderer native --device cuda:0 \
  --checkpoint artifacts/a04_reference/server-smoke.pt
```

The `--allow-incomplete` manifest option is reserved for local fixtures. The
server command must use the default all-fold requirement. A04 training accepts
only `final_train`; `a04 eval` accepts only `final_test`. A01 target caching
includes training samples only. Final-test teacher targets are inferred live,
and GT remains outside `DeploymentStereoBatch`.

The current A04 `eval` command reports held-out reconstruction diagnostics; a
separate GT-backed metric evaluation protocol is still required before final
benchmark claims.

FoundationStereo and DEFOM-Stereo source trees, licenses, preprocessing
configuration and checkpoint files remain external requirements. Their
factory must provide a 64-character checkpoint digest and verified API/output
metadata. The native path additionally requires the CUDA build of
`diff_gaussian_rasterization` at revision
`59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d` and a Linux CUDA toolchain. The
repository currently provides the first-party ABI/camera boundary only; it
does not claim native execution or parity until that extension is built and
run on the target server.

An A04 cache-only run may omit `--teacher-factory` only when its config has a
fully resolved `teacher_identity` mapping that exactly matches the identity
used to create the cache. The distributed config intentionally says
`injected_at_runtime`, so it fails closed until a real teacher identity is
resolved; this prevents a cache from being treated as provenance-free data.
