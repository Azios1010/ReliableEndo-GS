# ReliableEndo-GS

Research repository for reliable feed-forward stereo endoscopic Gaussian Splatting.

The project is planned as a two-stage research program:

1. ProbStereo-EndoGS
2. RiskRoute-GS

Repository status: Plan 02 SCARED-C development data contracts and Plan 03A
baseline evaluation/profiling infrastructure are implemented. Official
baseline reproduction remains blocked by deferred Plan 01 parity, missing
checkpoint/runtime prerequisites, and the pending final grouped protocol.

The upstream-independent `reliable_endo_gs.contracts` package provides typed
camera, stereo, Gaussian, render, and Phase I reconstruction-state containers.
The external SCARED-C development protocol is kept distinct from original
SCARED. Training and later scientific methods remain staged behind the
approved roadmap.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
mypy src
pytest
reg smoke --config configs/experiment/infrastructure_smoke.yaml
```

See `docs/development.md` for the infrastructure and dependency policy.
