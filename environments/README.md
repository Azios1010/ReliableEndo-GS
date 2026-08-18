# CPU Development Environment

Create the lightweight CPU-only environment from the repository root:

```bash
conda env create -f environments/cpu.yml
conda activate reliable-endo-gs-cpu
```

The environment installs the project in editable mode with the `dev` dependency group. It contains no CUDA, dataset, or scientific-model dependencies.
