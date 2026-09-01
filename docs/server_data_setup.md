# Server Data Setup

## Environment-level root

Datasets are mounted externally and are not downloaded or copied by this
repository. A conceptual layout is:

```text
/data/endoscopy/
|-- SCARED/
|-- scared_c/
|-- EndoNeRF/
`-- C3VD/
```

Set the canonical parent root for the current shell:

```bash
export RELIABLE_ENDO_DATA_ROOT=/data/endoscopy
reg data validate --config configs/data/scared.yaml
reg data validate --config configs/data/scared_c.yaml
```

SCARED and SCARED-C are different dataset/protocol identities. The SCARED-C
development configuration selects the named
`scared_c_endoscope_stereo_calibration_v1` protocol and must not be used to
claim original-SCARED results.

SCARED-C validation reports `MISSING`, `INCOMPLETE`, `INVALID`, or `USABLE`,
plus logical sequence/sample counts and contract checks. The one-keyframe local
subset is usable for development integration but is insufficient for a final
grouped train/validation/test split.

## Explicit root override

For an alternate mounted location, pass an absolute parent root without
modifying committed configuration:

```bash
reg data validate --config configs/data/scared_c.yaml --data-root /scratch/project-data
```

The explicit root takes precedence over `RELIABLE_ENDO_DATA_ROOT`. Absolute
paths are operational inputs only and must not be written to configs, IDs,
manifests, reports, or ordinary logs.

## Windows PowerShell

```powershell
$env:RELIABLE_ENDO_DATA_ROOT = "D:\datasets\endoscopy"
reg data validate --config configs/data/scared_c.yaml
```

The local development subset in this workspace can be validated without
changing committed files:

```powershell
$env:RELIABLE_ENDO_DATA_ROOT = (Resolve-Path "data").Path
reg data validate --config configs/data/scared_c.yaml
```

The normal JSON report redacts the resolved root.

## Switching servers safely

Keep `configs/data/*.yaml` portable and change only the environment variable or
explicit CLI override. Do not store credentials, SSH configuration, cloud URLs,
mount commands, or medical data in the repository. This project does not
implement SSH, S3, network mounting, or dataset downloading.
