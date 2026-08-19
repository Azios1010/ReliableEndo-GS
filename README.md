# ReliableEndo-GS

Research repository for reliable feed-forward stereo endoscopic Gaussian Splatting.

The project is planned as a two-stage research program:

1. ProbStereo-EndoGS
2. RiskRoute-GS

Repository status: core PyTorch contract foundation implemented; baseline and
scientific methods remain staged behind the approved roadmap.

The upstream-independent `reliable_endo_gs.contracts` package provides typed
camera, stereo, Gaussian, render, and Phase I reconstruction-state containers.
Scientific methods, dataset tensor loaders, upstream integration, training
pipelines, and experiments are not implemented yet.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
mypy src
pytest
reg smoke --config configs/experiment/infrastructure_smoke.yaml
```

See `docs/development.md` for the infrastructure and dependency policy.
