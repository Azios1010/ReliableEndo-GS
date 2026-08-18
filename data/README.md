# External Dataset Mount Point

Real datasets are external to this Git repository and must not be committed here. `RELIABLE_ENDO_DATA_ROOT` is the canonical mechanism for locating a mounted parent directory on a workstation or server.

This directory is only a protected runtime location. Local symlinks may be used when convenient, but they must not be committed. Portable dataset YAML files live under `configs/data/`, adapter contracts live under `src/reliable_endo_gs/data/`, and approved split manifests live under `splits/`.

No dataset is downloaded automatically, and no unverified download or internal-layout instructions are provided here.

