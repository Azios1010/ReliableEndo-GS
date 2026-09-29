# Server-readiness remediation

The active executable path is SCARED dataset_1 through dataset_7. The data
root is external and must be resolved through `RELIABLE_ENDO_DATA_ROOT`; no
SCARED-C directory is accepted by the A01 manifest builder.

Each processed SCARED keyframe used by A00 must carry the protocol described
in [the A00 protocol](a00_scared_protocol.md), either as the keyframe sidecar
or as the documented embedded object. The protocol certifies image
range, rectification, disparity sign/units, extrinsic direction/units and
the preprocessing grid. A01 cache generation and A04 execution use the same
`--resolution-hw` values so cache alignment and reference input geometry have
the same identity.

The commands below are bounded entry points. Replace the teacher factory with
an external module that loads a pinned FoundationStereo or DEFOM-Stereo source
revision and checkpoint, returns the frozen `Teacher` adapter, and declares
the verified preprocessing/output identity.

```bash
export RELIABLE_ENDO_DATA_ROOT=/mnt/data
DATA_ROOT="$RELIABLE_ENDO_DATA_ROOT/scared"
MANIFEST=artifacts/manifests/a01_scared_1_7.json
CACHE=artifacts/targets/a01
TEACHER_FACTORY=my_teacher_integration:build_foundationstereo
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
  --data-role teacher_fit \
  --teacher-factory "$TEACHER_FACTORY" --renderer native --device cuda:0

python -m reliable_endo_gs.cli.main a04 train \
  --config configs/reconstruction/a04_reference.yaml \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" --cache-root "$CACHE" \
  --data-role teacher_fit \
  --teacher-factory "$TEACHER_FACTORY" --renderer native --device cuda:0 \
  --max-steps 1 --output-checkpoint artifacts/a04_reference/server-smoke.pt

python -m reliable_endo_gs.cli.main a04 eval \
  --config configs/reconstruction/a04_reference.yaml \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" --cache-root "$CACHE" \
  --data-role qualification \
  --teacher-factory "$TEACHER_FACTORY" --renderer native --device cuda:0 \
  --checkpoint artifacts/a04_reference/server-smoke.pt

python -m reliable_endo_gs.cli.main a04 profile \
  --config configs/reconstruction/a04_reference.yaml \
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" --cache-root "$CACHE" \
  --data-role qualification \
  --teacher-factory "$TEACHER_FACTORY" --renderer native --device cuda:0 \
  --checkpoint artifacts/a04_reference/server-smoke.pt
```

The `--allow-incomplete` manifest option is reserved for local fixtures. The
server command must use the default all-fold requirement. A04 `--data-role`
can select `teacher_fit` or `qualification`; the resolved config defaults to
`teacher_fit` and the trainer filters the manifest to that role.

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
