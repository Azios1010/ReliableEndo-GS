#!/usr/bin/env bash
# Thin bounded entry point. Set external paths; no dataset/cache mutation.
set -euo pipefail
RELIABLE_ENDO_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RELIABLE_ENDO_PROJECT_ROOT=$(cd -- "$RELIABLE_ENDO_SCRIPT_DIR/.." && pwd)
: "${RELIABLE_ENDO_PYTHON:?Set the existing runtime Python executable}"
: "${RELIABLE_ENDO_DATA_ROOT:?Set the external processed SCARED root}"
: "${RELIABLE_ENDO_CHECKPOINT:?Set the pinned frozen checkpoint}"
: "${RELIABLE_ENDO_PROBE_REPORT:?Set the verified CPU candidate report}"
: "${RELIABLE_ENDO_TRAIN_MANIFEST:?Set the immutable training manifest}"
: "${RELIABLE_ENDO_DEV_MANIFEST:?Set the disjoint development manifest}"
: "${RELIABLE_ENDO_RUN_ROOT:?Set an existing external output parent}"
export CUDA_VISIBLE_DEVICES=${RELIABLE_ENDO_GPU:-0}
export PYTHONPATH="$RELIABLE_ENDO_PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd -- "$RELIABLE_ENDO_PROJECT_ROOT"
# Validate GPU and paths before allocating output. No fallback to CPU.
"$RELIABLE_ENDO_PYTHON" - <<'PY'
import os
from pathlib import Path
import torch
repo = Path.cwd().resolve()
root = Path(os.environ['RELIABLE_ENDO_RUN_ROOT']).resolve()
if root == repo or repo in root.parents or not root.is_dir():
    raise SystemExit('Output parent must exist outside repository')
for key in ('RELIABLE_ENDO_DATA_ROOT', 'RELIABLE_ENDO_CHECKPOINT', 'RELIABLE_ENDO_PROBE_REPORT',
            'RELIABLE_ENDO_TRAIN_MANIFEST', 'RELIABLE_ENDO_DEV_MANIFEST'):
    if not Path(os.environ[key]).exists():
        raise SystemExit(f'Missing external input: {key}')
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('Need exactly one visible GPU; this pilot has no CPU fallback')
print(f'Physical GPU {os.environ["CUDA_VISIBLE_DEVICES"]}: {torch.cuda.get_device_name(0)}')
PY
RELIABLE_ENDO_PILOT_DIR=$(mktemp -d "$RELIABLE_ENDO_RUN_ROOT/outcome-pilot-XXXXXXXX")
printf 'Bounded outcome pilot: %s\n' "$RELIABLE_ENDO_PILOT_DIR"
exec > >(tee "$RELIABLE_ENDO_PILOT_DIR/console.log") 2>&1
RELIABLE_ENDO_TRADEOFF_CONFIG=${RELIABLE_ENDO_TRADEOFF_CONFIG:-"$RELIABLE_ENDO_PROJECT_ROOT/configs/routing/metric_tradeoff.yaml"}
RELIABLE_ENDO_TRAIN_PER_DATASET=${RELIABLE_ENDO_TRAIN_PER_DATASET:-8}
RELIABLE_ENDO_DEV_PER_DATASET=${RELIABLE_ENDO_DEV_PER_DATASET:-4}
RELIABLE_ENDO_TRAINING_ARGUMENTS=()
if [[ -n "${RELIABLE_ENDO_TRAINING_POLICY:-}" ]]; then
    RELIABLE_ENDO_TRAINING_ARGUMENTS=(--training-policy "$RELIABLE_ENDO_TRAINING_POLICY")
fi
for RELIABLE_ENDO_COLLECTION_ROLE in train development; do
    RELIABLE_ENDO_PER_DATASET=$RELIABLE_ENDO_TRAIN_PER_DATASET
    if [[ "$RELIABLE_ENDO_COLLECTION_ROLE" == development ]]; then
        RELIABLE_ENDO_PER_DATASET=$RELIABLE_ENDO_DEV_PER_DATASET
    fi
    "$RELIABLE_ENDO_PYTHON" -m reliable_endo_gs.cli.selection collect-outcomes \
        --checkpoint "$RELIABLE_ENDO_CHECKPOINT" \
        --probe-report "$RELIABLE_ENDO_PROBE_REPORT" \
        --data-root "$RELIABLE_ENDO_DATA_ROOT" \
        --training-manifest "$RELIABLE_ENDO_TRAIN_MANIFEST" \
        --manifest "$RELIABLE_ENDO_DEV_MANIFEST" \
        --roles "$RELIABLE_ENDO_PROJECT_ROOT/configs/teachers/a01_scared.yaml" \
        --config "$RELIABLE_ENDO_TRADEOFF_CONFIG" \
        --collection-role "$RELIABLE_ENDO_COLLECTION_ROLE" \
        --per-dataset "$RELIABLE_ENDO_PER_DATASET" --timing-repeats 3 \
        --output-dir "$RELIABLE_ENDO_PILOT_DIR/$RELIABLE_ENDO_COLLECTION_ROLE"
done
"$RELIABLE_ENDO_PYTHON" -m reliable_endo_gs.cli.selection outcome-pilot \
    --training-table "$RELIABLE_ENDO_PILOT_DIR/train" \
    --development-table "$RELIABLE_ENDO_PILOT_DIR/development" \
    --config "$RELIABLE_ENDO_TRADEOFF_CONFIG" \
    --device cuda:0 --budgets-ms 250 500 1000 \
    "${RELIABLE_ENDO_TRAINING_ARGUMENTS[@]}" \
    --output-dir "$RELIABLE_ENDO_PILOT_DIR/selector"
printf 'Completed bounded pilot: %s\n' "$RELIABLE_ENDO_PILOT_DIR/selector/result.json"
