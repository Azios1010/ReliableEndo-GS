# Plan 15 — External Validation and Reproducible Release

## Status

PLANNED

## Research stage

Final evaluation / Release

## Recommended executor

Terra with Sol review. Terra is suitable for packaging, manifests, commands, and release automation; Sol must review dataset compatibility, external analyses, and claim boundaries.

## Estimated implementation risk

MEDIUM. Engineering is bounded, but dataset-task mismatch, licensing, unavailable checkpoints, or incomplete provenance can make an apparently polished release scientifically misleading.

## Objective

Evaluate approved artifacts on scientifically compatible external protocols and publish a reproducible, license-aware release containing code, configs, manifests, permitted checkpoints, reports, limitations, and negative pivots.

## Scientific question

Do the supported reliability conclusions transfer beyond the primary SCARED protocol under clearly stated dataset/task limitations?

## Why this milestone exists

Primary evidence comes from SCARED. External validation tests scope/generalization, while release packaging ensures every supported claim can be traced and reproduced without exposing restricted data.

## Prerequisites

- Plan 14 final report artifact accepted.
- Approved claim/limitation matrix and artifact chain.
- License/redistribution review for source, weights, datasets, generated artifacts, and dependencies.

## Inputs

- Frozen baseline, Phase I, oracle, applicable router, and final report artifacts.
- Existing external-data adapter scaffolds for EndoNeRF and C3VD.
- Authoritative dataset documentation and authorized mounts.

## Outputs

- Separate external validation reports for compatible EndoNeRF and C3VD protocols.
- Reproducible release bundle/manifests, commands, configs, split manifests, permitted checkpoints/artifact references, and paper-table scripts.
- Model/data cards, license/attribution inventory, limitations, negative results, and environment instructions.

## Scope

- Keep SCARED as the primary genuine stereo benchmark.
- Use EndoNeRF for external rendering/generalization and qualitative geometry only where cameras/depth/tool masks are compatible; do not elevate limited depth to primary metric geometry proof.
- Use C3VD as an auxiliary geometry/robustness stress test. Do not describe it as simultaneous stereo unless authoritative inspection verifies such a protocol; nearby-pose pseudo-pairs are reported separately.
- Implement external tensor pipelines only after layout/calibration/license audits analogous to Plan 02.
- Freeze external protocols before results and avoid retuning the core representation/router on external test data.
- Package minimal reproduction commands from immutable IDs and verify them in clean CPU/GPU environments as applicable.

## Non-goals

- No clinical superiority/safety/deployment claim, no dataset redistribution without permission, no synthetic-stereo result presented as genuine stereo, no rewriting failed claims, and no mutable release checkpoint aliases.

## Files expected to be created

- `docs/reproducibility.md`
- `docs/limitations.md`
- `docs/model_card.md`
- `docs/data_card.md`
- `docs/licenses/third_party_inventory.md`
- `configs/experiment/external_endonerf.yaml`
- `configs/experiment/external_c3vd.yaml`
- `scripts/run_external_validation.py`
- `scripts/reproduce_report.py`
- `scripts/verify_release.py`
- `reports/external/` generated report manifests/templates.
- `tests/release/test_manifest_completeness.py`
- `tests/release/test_reproduction_commands.py`
- External adapter/tensor tests and synthetic fixtures only when their authoritative layouts are implemented.

## Files expected to be modified

- `src/reliable_endo_gs/data/adapters/endonerf.py` and `c3vd.py` after authoritative inspection.
- Data loaders/indexing through shared interfaces, not duplicated pipelines.
- `configs/data/endonerf.yaml` and `c3vd.yaml` with verified options.
- `README.md`, citation/license files, environment files, and release metadata.
- `docs/data_architecture.md` and server instructions with verified external workflows.

## Interfaces and contracts

External adapters reuse the Plan 02 index/tensor contracts and stable IDs. Compatibility adapters declare absent supervision rather than fabricate fields. Release manifest resolves every command/config/checkpoint/report to immutable identities and verifies required external inputs without embedding machine paths or credentials.

## Scientific formulation

No new method equation. External metric owners remain the Plan 14 evaluation modules. Any pseudo-pair construction is a versioned evaluation transformation with a documented test, not proposed model novelty. External validation tests generalization of already approved claims.

## Numerical and coordinate conventions

Audit each dataset's camera/pose/depth/image/mask units independently. Convert once at the adapter boundary and test analytic/round-trip cases. Do not assume SCARED rectification, depth quality, or simultaneity transfers. Report unsupported metrics as unavailable.

## Configuration changes

Add external data and experiment configs with explicit artifact IDs, protocol/split, supported metrics, resize/camera conversion, and report scope. Release config pins environments, dependencies, upstream commit, schemas, and artifact references. No private absolute paths.

## Tests

### Unit

- External layout/calibration/ID parsing on synthetic legal fixtures.
- Release manifest, license inventory, artifact hash, and command generation.

### Contract

- External samples satisfy supported `StereoBatch` subset or an explicitly different evaluation contract.
- Missing ground truth cannot produce a metric value.

### Integration

- Mounted external subset -> frozen model/policy -> report artifact.
- Clean release install -> manifest resolution -> representative reproduction.

### Smoke

- CPU documentation/config validation and minimal supported GPU inference.

### Regression

- Release manifest/CLI schemas and representative external result-source summaries.

## Validation commands

```text
ruff check .
ruff format --check .
mypy src
pytest -q tests/release tests/data tests/integration
pytest -q
python -m compileall src
python scripts/verify_release.py --manifest <immutable-release-manifest>
git diff --check
```

## Experiments

1. EndoNeRF external rendering/generalization under a predeclared compatible protocol.
2. C3VD auxiliary geometry stress test, reported separately from genuine stereo.
3. Clean-environment reproduction of representative SCARED/external tables and plots.
4. Release privacy/license/artifact availability audit.

## Metrics

- Only dataset-supported Plan 14 metrics, with per-sequence results and confidence intervals where meaningful.
- Domain-shift failure rates, validity coverage, and qualitative examples selected by a fixed policy.
- Reproduction command success, artifact/hash completeness, and environment setup time/errors.

## Acceptance criteria

- SCARED remains the primary claim basis; external protocols and limitations are explicit.
- EndoNeRF/C3VD results use only scientifically compatible metrics and never misstate stereo/depth status.
- Release commands resolve immutable artifacts and reproduce representative outputs in clean declared environments.
- Licenses, attribution, split manifests, configs, permitted checkpoints, cards, limitations, and negative results are complete.
- No restricted/private data, credentials, or machine paths enter the release.

## Failure conditions

- External calibration/task compatibility cannot be verified, licenses prohibit planned release, artifacts cannot be reproduced, or external results contradict claims without disclosure.

## Pivot / rollback path

Omit an incompatible external dataset and publish the audit/limitation instead. Release checkpoint hashes/acquisition instructions rather than weights when redistribution is prohibited. Narrow generalization claims to SCARED and retain negative external evidence.

## Artifact/provenance requirements

Release manifest records every upstream/source/checkpoint/artifact identity, code commit, schema, config/split hash, environment/container, hardware, license/attribution, reproduction command, report-source hash, limitation, and omitted payload reason. Large artifacts remain external and content-addressed.

## Completion checklist

- [ ] External layouts, licenses, and task compatibility are verified.
- [ ] EndoNeRF and C3VD roles remain correctly limited.
- [ ] Release bundle contains traceable configs/manifests/scripts/cards.
- [ ] Clean-environment reproduction checks pass.
- [ ] Privacy and redistribution audit passes.
- [ ] Claims, limitations, pivots, and negative results match evidence.

## Handoff to next plan

This is the terminal roadmap milestone. A subsequent paper/release decision may assume only the validated artifact chain and claims recorded here; any new method or dataset protocol requires a new approved plan and fresh gates.
