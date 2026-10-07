#!/usr/bin/env bash
# Thin operational entry point; all scientific logic is in src/.
set -euo pipefail
: "${RELIABLE_ENDO_PYTHON:?Set the existing CUDA runtime Python}"
: "${RELIABLE_ENDO_DATA_ROOT:?Set the external processed SCARED A00 root}"
: "${RELIABLE_ENDO_TRAIN_MANIFEST:?Set the unchanged training subset manifest}"
: "${RELIABLE_ENDO_DEVELOPMENT_MANIFEST:?Set the disjoint development manifest}"
: "${RELIABLE_ENDO_KD_CACHE:?Set the existing pinned teacher cache}"
: "${RELIABLE_ENDO_INITIAL_CHECKPOINT:?Set the prior student checkpoint for model-only initialization}"
RELIABLE_ENDO_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RELIABLE_ENDO_PILOT_STEPS=${1:-256}
RELIABLE_ENDO_PILOT_CONFIG=${RELIABLE_ENDO_INTERVENTION_CONFIG:-configs/routing/intervention_selection.yaml}
RELIABLE_ENDO_PILOT_SMOKE_CONFIG=${RELIABLE_ENDO_SMOKE_INTERVENTION_CONFIG:-$RELIABLE_ENDO_PILOT_CONFIG}
RELIABLE_ENDO_PILOT_SMOKE_SCHEDULE=${RELIABLE_ENDO_SMOKE_CONFIG:-configs/training/intervention_smoke.yaml}
if [[ ! "$RELIABLE_ENDO_PILOT_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Pilot steps must be a positive integer\n' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RELIABLE_ENDO_PROJECT_ROOT/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$RELIABLE_ENDO_PROJECT_ROOT"
nvidia-smi -i 0 --query-gpu=index,name,uuid,memory.used,memory.total --format=csv,noheader
RELIABLE_ENDO_PILOT_ROOT=$(mktemp -d "${RELIABLE_ENDO_RUNS_ROOT:-/tmp}/intervention-pilot-XXXXXXXX")
printf 'New pilot artifacts: %s\n' "$RELIABLE_ENDO_PILOT_ROOT"
exec > >(tee "$RELIABLE_ENDO_PILOT_ROOT/console.log") 2>&1
"$RELIABLE_ENDO_PYTHON" -m reliable_endo_gs.cli.selection quality \
  --data-root "$RELIABLE_ENDO_DATA_ROOT" --manifest "$RELIABLE_ENDO_TRAIN_MANIFEST" \
  --cache-root "$RELIABLE_ENDO_KD_CACHE" --max-samples 64 \
  --config "$RELIABLE_ENDO_PILOT_CONFIG" \
  --output-dir "$RELIABLE_ENDO_PILOT_ROOT/quality"
RELIABLE_ENDO_SMOKE_ID=$("$RELIABLE_ENDO_PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["entries"][0]["sample_id"])' "$RELIABLE_ENDO_TRAIN_MANIFEST")
"$RELIABLE_ENDO_PYTHON" -m reliable_endo_gs.cli.student smoke \
  --data-root "$RELIABLE_ENDO_DATA_ROOT" --manifest "$RELIABLE_ENDO_TRAIN_MANIFEST" \
  --cache-root "$RELIABLE_ENDO_KD_CACHE" --sample-id "$RELIABLE_ENDO_SMOKE_ID" \
  --config "$RELIABLE_ENDO_PILOT_SMOKE_SCHEDULE" \
  --intervention-config "$RELIABLE_ENDO_PILOT_SMOKE_CONFIG" \
  --initialize-model-from "$RELIABLE_ENDO_INITIAL_CHECKPOINT" \
  --output-dir "$RELIABLE_ENDO_PILOT_ROOT/smoke"
"$RELIABLE_ENDO_PYTHON" -m reliable_endo_gs.cli.student train \
  --data-root "$RELIABLE_ENDO_DATA_ROOT" --manifest "$RELIABLE_ENDO_TRAIN_MANIFEST" \
  --cache-root "$RELIABLE_ENDO_KD_CACHE" --max-steps "$RELIABLE_ENDO_PILOT_STEPS" \
  --config configs/training/intervention_joint.yaml \
  --intervention-config "$RELIABLE_ENDO_PILOT_CONFIG" \
  --initialize-model-from "$RELIABLE_ENDO_INITIAL_CHECKPOINT" \
  --output-dir "$RELIABLE_ENDO_PILOT_ROOT/train"
if [[ -f "$RELIABLE_ENDO_PILOT_ROOT/train/final.pt" ]]; then
  RELIABLE_ENDO_PILOT_CHECKPOINT="$RELIABLE_ENDO_PILOT_ROOT/train/final.pt"
else
  RELIABLE_ENDO_PILOT_CHECKPOINT="$RELIABLE_ENDO_PILOT_ROOT/train/stopped.pt"
fi
"$RELIABLE_ENDO_PYTHON" -m reliable_endo_gs.cli.selection evaluate \
  --data-root "$RELIABLE_ENDO_DATA_ROOT" --manifest "$RELIABLE_ENDO_DEVELOPMENT_MANIFEST" \
  --training-manifest "$RELIABLE_ENDO_TRAIN_MANIFEST" \
  --checkpoint "$RELIABLE_ENDO_PILOT_CHECKPOINT" --max-samples 112 \
  --output-dir "$RELIABLE_ENDO_PILOT_ROOT/evaluation"
printf 'Pilot completed. Review oracle diversity, independent GT and matched-cost baselines: %s\n' "$RELIABLE_ENDO_PILOT_ROOT"
