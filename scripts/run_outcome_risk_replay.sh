#!/usr/bin/env bash
# CPU replay of immutable native outcomes. No GPU/teacher/renderer execution.
set -euo pipefail
RELIABLE_ENDO_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RELIABLE_ENDO_PROJECT_ROOT=$(cd -- "$RELIABLE_ENDO_SCRIPT_DIR/.." && pwd)
: "${RELIABLE_ENDO_PYTHON:?Set an existing runtime Python executable}"
: "${RELIABLE_ENDO_OUTCOME_TABLE_ROOT:?Set the completed native train/development table root}"
RELIABLE_ENDO_RUN_ROOT=${RELIABLE_ENDO_RUN_ROOT:-/tmp}
export PYTHONPATH="$RELIABLE_ENDO_PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd -- "$RELIABLE_ENDO_PROJECT_ROOT"
RELIABLE_ENDO_RISK_RUN_DIR=$(mktemp -d "$RELIABLE_ENDO_RUN_ROOT/selector-risk-XXXXXXXX")
exec > >(tee "$RELIABLE_ENDO_RISK_RUN_DIR/console.log") 2>&1
printf 'Bounded CPU predictor replay: %s\n' "$RELIABLE_ENDO_RISK_RUN_DIR"
for RELIABLE_ENDO_RISK_ARM in reference risk_discrimination conditioned_risk; do
    case "$RELIABLE_ENDO_RISK_ARM" in
        reference) RELIABLE_ENDO_TRAINING_CONFIG=outcome_regularized.yaml ;;
        risk_discrimination) RELIABLE_ENDO_TRAINING_CONFIG=outcome_risk_discrimination.yaml ;;
        conditioned_risk) RELIABLE_ENDO_TRAINING_CONFIG=outcome_conditioned_risk.yaml ;;
    esac
    "$RELIABLE_ENDO_PYTHON" -m reliable_endo_gs.cli.selection outcome-pilot \
        --training-table "$RELIABLE_ENDO_OUTCOME_TABLE_ROOT/train" \
        --development-table "$RELIABLE_ENDO_OUTCOME_TABLE_ROOT/development" \
        --config "$RELIABLE_ENDO_PROJECT_ROOT/configs/routing/metric_tradeoff_balanced.yaml" \
        --training-policy "$RELIABLE_ENDO_PROJECT_ROOT/configs/training/$RELIABLE_ENDO_TRAINING_CONFIG" \
        --device cpu --budgets-ms 1000 --relabel-policy \
        --output-dir "$RELIABLE_ENDO_RISK_RUN_DIR/$RELIABLE_ENDO_RISK_ARM"
done
printf 'Completed CPU predictor replay: %s\n' "$RELIABLE_ENDO_RISK_RUN_DIR"
