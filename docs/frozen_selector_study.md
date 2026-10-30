# Frozen selector development runbook

Opt-in implementation of [Plan 18](../plans/18_frozen_repair_selector_corrections.md).
These templates have not launched a scientific run. Use an immutable source/config
snapshot and new external outputs. Set `RELIABLE_ENDO_DATA_ROOT`, `PREPARATION_ROOT`,
`EXPERIMENT_ROOT`, `KD_CHECKPOINT` to approved external assets. Native stages require
one visible CUDA GPU; `--gpu-input` requires `OMP_NUM_THREADS=1`.

## Prepare before outcomes

```bash
python -m reliable_endo_gs.cli.selector_prepare automatic \
  --data-root "$RELIABLE_ENDO_DATA_ROOT" \
  --training-manifest "$PREPARATION_ROOT/train_1236.json" \
  --development-manifest "$PREPARATION_ROOT/val_7.json" \
  --roles-path configs/teachers/a01_scared.yaml \
  --refit-config-path configs/routing/selector_refit_frozen.yaml \
  --study-protocol-path configs/routing/selector_study.yaml \
  --per-keyframe 16 --output-dir "$EXPERIMENT_ROOT/cohort"
```

Seals sample IDs, whole-keyframe partition and source-byte hashes without GT decode.
No operator ROI index. Do not use a partial data cache missing manifest members.

## Stage A after prerequisites

Keep Plan 16's frozen student, model-only KD initialization and repair objectives.
Use the same automatic candidates as Stage B:

```bash
python -m reliable_endo_gs.cli.student train \
  --config configs/training/student_frozen_repair.yaml \
  --construction-config configs/reconstruction/a04_reference.yaml \
  --routing-config configs/routing/frozen_a04.yaml \
  --intervention-config configs/routing/intervention_metric_geometry_frozen_student.yaml \
  --study-protocol-path configs/routing/selector_study.yaml \
  --roles configs/teachers/a01_scared.yaml \
  --manifest "$PREPARATION_ROOT/train_1236.json" \
  --data-root "$RELIABLE_ENDO_DATA_ROOT" --gpu-input \
  --initialize-model-from "$KD_CHECKPOINT" --output-dir "$EXPERIMENT_ROOT/stage-a"
```

Use the established supervisor/strict resume. Verify student/model-Adam invariance
and freeze the full artifact; discard Stage A's temporary selector. Keep the active
baseline snapshot separate. Set `REPAIR_CHECKPOINT` and its verified `REPAIR_SHA256`.

## Preflight, TRAIN costs, Gate A and collection

The following shell array supplies common arguments to `cli.selector_refit`:

```bash
REFIT_ARGS=(
  --checkpoint "$REPAIR_CHECKPOINT" --checkpoint-sha256 "$REPAIR_SHA256"
  --data-root "$RELIABLE_ENDO_DATA_ROOT"
  --training-manifest "$PREPARATION_ROOT/train_1236.json"
  --development-manifest "$PREPARATION_ROOT/val_7.json"
  --roles-path configs/teachers/a01_scared.yaml
  --worklist-path "$EXPERIMENT_ROOT/cohort/worklist.json"
  --refit-config-path configs/routing/selector_refit_frozen.yaml
  --study-protocol-path configs/routing/selector_study.yaml
  --output-dir "$EXPERIMENT_ROOT/collection"
)
python -m reliable_endo_gs.cli.selector_refit "${REFIT_ARGS[@]}" --preflight-only
python -m reliable_endo_gs.cli.selector_refit "${REFIT_ARGS[@]}" --resume --prepare-costs
```

Cost preparation is GT-free, not an end-to-end FPS benchmark. Record hardware and
contention when running it. A bounded smoke checkpoint cannot enter Stage B.

Evaluate repairs with `cli.selection evaluate`: checkpoint/digest, both manifests,
roles, data root, `--study-protocol-path`, `--sample-worklist` pointing to the sealed
cohort, `--max-samples 64`, and `--cost-calibration` pointing to
`collection/cost_calibration.pt`. Use a new output `repair-evaluation`. Supply
`--lpips-backbone`, `--lpips-calibration`, `--require-lpips` for pinned local weights.
This mode discards the temporary Stage A selector.

```bash
python -m reliable_endo_gs.cli.selector_evidence \
  --native-evaluation "$EXPERIMENT_ROOT/repair-evaluation" \
  --output-dir "$EXPERIMENT_ROOT/gate-a"
python -m reliable_endo_gs.cli.selector_refit "${REFIT_ARGS[@]}" \
  --resume --collect-only --gate-a-report "$EXPERIMENT_ROOT/gate-a/gate_a.json"
```

Gate A must pass for the exact frozen actor/candidates. Do not change thresholds
after failure. Collection preserves full feature banks, raw outcomes and unknowns;
resume verifies immutable completed shards.

## Fit once per arm/seed, then native development

```bash
python -m reliable_endo_gs.cli.selector_study \
  --collection-dir "$EXPERIMENT_ROOT/collection" \
  --study-protocol-path configs/routing/selector_study.yaml \
  --refit-config-path configs/routing/selector_refit_frozen.yaml \
  --gate-a-report "$EXPERIMENT_ROOT/gate-a/gate_a.json" \
  --output-dir "$EXPERIMENT_ROOT/study" --device-name cpu
```

`--audit-only` in a separate directory stops before fitting. Gate B uses FIT only.
A passing utility arm fits C0/C1/C2 × three seeds and saves four CAL operating
points per fit. All share actors/store/partition. Unsupported points abstain.
Variant choice uses VALIDATION; no data-root or development table enters fit-only.

Native-evaluate every chosen artifact/seed with the same worklist/costs and
`--selector-refit-path`, using the identical fitting source snapshot. Create an
external JSON index mapping chosen artifact filenames to native evaluation dirs:

```bash
python -m reliable_endo_gs.cli.selector_evidence \
  --study-dir "$EXPERIMENT_ROOT/study" \
  --evaluation-index "$EXPERIMENT_ROOT/native-evaluation-index.json" \
  --output-dir "$EXPERIMENT_ROOT/promotion"
```

All declared seeds are required. Best fixed is reselected within keyframe bootstrap
resamples. Unknown comparator outcomes block promotion. LPIPS availability is explicit.
`paper_claim_ready` remains false: final margins, untouched final_test and separately
authorized end-to-end latency/FPS measurements are still required.
