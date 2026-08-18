# Development Infrastructure

## Python and installation

ReliableEndo-GS supports Python 3.10 and newer. From the repository root, install the project and its development tools in editable mode:

```bash
python -m pip install -e ".[dev]"
```

The runtime dependency set currently contains only PyYAML. The `dev` extra adds pytest, pytest-cov, Ruff, mypy, pre-commit, and YAML typing support.

## Quality commands

```bash
ruff check .
ruff format .
mypy src
pytest -q
python -m compileall src
```

Install the local pre-commit hooks with `pre-commit install`. The hooks run Ruff check with safe fixes and Ruff formatting; they require no GPU or dataset.

## Infrastructure smoke command

```bash
reg --help
reg smoke --config configs/experiment/infrastructure_smoke.yaml
```

The smoke command loads and validates configuration, seeds Python's `random` module, computes a canonical configuration hash, reads Git metadata without changing Git state, and creates a unique run directory. It never imports a scientific model or accesses a dataset.

On Windows, activate the project virtual or Conda environment before invoking `reg`. Windows also ships a system executable named `reg.exe`, so the active environment's `Scripts` directory must precede `System32` on `PATH`.

## Configuration foundation

Configuration is intentionally small: frozen dataclasses describe the experiment name, seed, output root, and runtime device. YAML loading rejects missing, unknown, or invalid fields. There is no composition system, dataset section, model section, or scientific default.

Relative output paths remain relative in resolved configuration. Canonical JSON serialization with sorted keys and normalized path strings produces a stable SHA-256 configuration hash.

## Run manifests and outputs

Each smoke run writes:

```text
outputs/<run_id>/
|-- resolved_config.yaml
`-- run_manifest.json
```

The immutable manifest records the run ID, UTC timestamp, project commit when available, dirty status when available, configuration hash, seed, and Python version. It does not invent dataset, checkpoint, hardware, baseline, or scientific metadata.

Run directories are never overwritten. Runtime contents under `outputs/`, `artifacts/`, and `data/` are ignored except for documented placeholder files.

## Dependency policy

1. Add a dependency only when an implemented requirement needs it.
2. Prefer the standard library when it is sufficient.
3. Keep runtime and development dependencies separate.
4. Do not add large frameworks preemptively.
5. CUDA dependencies must never be required by CPU CI.

Hydra, Lightning, WandB, pandas, OpenCV, NumPy, PyTorch, SciPy, torchvision, LPIPS, and other scientific packages are intentionally absent. Scientific dependencies and CUDA environments will be introduced only by later, scoped implementation tasks.

## CPU environment and CI

`environments/cpu.yml` creates a lightweight Conda environment and installs `.[dev]`. CPU CI installs the same project extras, runs linting, formatting checks, type checking, tests, the smoke command, bytecode compilation, and a tracked-file cleanliness check. It neither downloads datasets nor requires a GPU.
