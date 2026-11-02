# Experiment provenance, 2026-11-02

Repository commit at the time of writing: `9a96d317c706613a8d49827664ad8c47e49f5f2d`.

```
9a96d31 Allow ablation controls: zero photometric weight and restricted refiner outputs
ebd7fae Add matched-comparison pipeline: baselines, evaluation runners, statistics and tests
0aa7495 Add opt-in right-edge strip guard for the refined field
2be5cea Add opt-in GT-disparity loss, KD-attribution configs and confirmatory statistics
```

Every reported result was produced from a hashed source snapshot (a copy of `src/`, the needed `scripts/` and `configs/`), not from the live working tree. This note maps each snapshot to the commits above:
the files of a snapshot that are byte-identical to the committed file, and the files that differ (list below). The snapshots were taken before these commits, so the commits are the first versions in git.

## E1 evaluation (eval-final-20261101T015251)
- tree hash: `8b3664433a44ad125c2f39ee875d99f58e22e5c93b307344419110c38c1ce373`; files: 289; identical to the committed/working file: 284; differing: 5; not in the repository: 0
- differing files (the snapshot has the older version): `src/reliable_endo_gs/baseline/comparison/posed_ft.py`, `src/reliable_endo_gs/baseline/posed_gaussians.py`, `src/reliable_endo_gs/evaluation/comparison_runner.py`, `src/reliable_endo_gs/gaussians/refinement.py`, `src/reliable_endo_gs/training/student_config.py`

## E5 EndoNeRF (e5-20261101T180708)
- tree hash: `85aad5c7d645bbb9e9853294662092f82e2a63ad08729b1fccafa15588322ddd`; files: 292; identical to the committed/working file: 287; differing: 5; not in the repository: 0
- differing files (the snapshot has the older version): `src/reliable_endo_gs/baseline/comparison/posed_ft.py`, `src/reliable_endo_gs/baseline/posed_gaussians.py`, `src/reliable_endo_gs/evaluation/comparison_runner.py`, `src/reliable_endo_gs/gaussians/refinement.py`, `src/reliable_endo_gs/training/student_config.py`

## P1 pair distance (p1-20261102T001632)
- tree hash: `1c7fa2c8f55d2b4a6d497d1c8fe4e8aa056d95c655771549a0c1812d6916bd7b`; files: 295; identical to the committed/working file: 290; differing: 5; not in the repository: 0
- differing files (the snapshot has the older version): `src/reliable_endo_gs/baseline/comparison/posed_ft.py`, `src/reliable_endo_gs/baseline/posed_gaussians.py`, `src/reliable_endo_gs/evaluation/comparison_runner.py`, `src/reliable_endo_gs/gaussians/refinement.py`, `src/reliable_endo_gs/training/student_config.py`

## left-source-only (leftonly-20261101T235802)
- tree hash: `0325d18d7020a19b5f1a3c63832077a56efc77e64219c22abf9a23fe9bc039c8`; files: 289; identical to the committed/working file: 287; differing: 2; not in the repository: 0
- differing files (the snapshot has the older version): `src/reliable_endo_gs/gaussians/refinement.py`, `src/reliable_endo_gs/training/student_config.py`

## ablations A/B (insight-ablations-20261101/source-snapshot)
- tree hash: `6927b7407ffe9a911114d5036c3ba21251939923136da0660f7a579d6f4192cf`; files: 398; identical to the committed/working file: 395; differing: 3; not in the repository: 0
- differing files (the snapshot has the older version): `src/reliable_endo_gs/baseline/comparison/posed_ft.py`, `src/reliable_endo_gs/baseline/posed_gaussians.py`, `src/reliable_endo_gs/evaluation/comparison_runner.py`

## 2026-11-03 additions (training arms of the review response; outputs under `/mnt/d/3DGS_runs/reliableendo-gs/`)
- photometric-only student (`insight-ablations-20261102-reconly/source-snapshot`): tree hash `4570fb0ca280c616e5f7e8cf7276d44b83c23a8027e052e3e37793a1af69662c` = the ablations A/B snapshot plus `configs/training/student_rec_only.yaml`; protocol `PROTOCOL_C.txt`.
- seed replicates (`insight-ablations-20261103-seeds/source-snapshot`): tree hash `e7d8e8788c9ea53b046e0a192cdb55fc1f6a82213740f8017dd84ee058faefd3` = the ablations A/B snapshot plus three configs that differ from the matched arms only in the seed; the replicates start from a fresh initialisation of their seed; protocol `PROTOCOL_D.txt`.
- visibility-masked photometric loss (`insight-ablations-20261103-visible/source-snapshot`): tree hash `413b3bbc200ca36df2dc5f2f18caf9cec92294cdfaa360dd58830095f57ce08a` = the ablations A/B snapshot plus `geometry/visibility.py` and the versions of `training/interventions.py`, `training/student_config.py`, `training/student_learning.py` committed in `3cfcde7`, plus `configs/training/student_kd_visible.yaml`; protocol `PROTOCOL_E.txt`.
- the grid-density diagnostic and the qualitative figure were run from snapshots that contained the code committed in `47a19ec`.

Notes
- The files that differ between the E1 snapshot and the commits are the opt-in options added afterwards (zero photometric weight, `active_outputs`, `left_source_only`); their defaults reproduce the earlier behaviour and are covered by unit tests.
- Protocols, amendments, freezes and the revision policy are hashed files under `reports/comparison-20261101/plan/` and `reports/insight-ablations-20261101/plan/` (outside the repository).
- Formatting: three files are not reformatted by `ruff format` on purpose (byte identity with the snapshots).
