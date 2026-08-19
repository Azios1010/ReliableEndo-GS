# CPU Development Environment

Create the lightweight CPU-only environment from the repository root:

```bash
conda env create -f environments/cpu.yml
conda activate reliable-endo-gs-cpu
```

The environment installs a PyTorch 2.x build plus the project in editable mode
with the `dev` dependency group. It requests no CUDA toolkit and requires no GPU,
dataset, upstream model, or scientific-model dependency. A final CUDA environment
will be defined only after the baseline integration establishes real requirements.
