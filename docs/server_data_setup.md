# Server Data Setup

## Environment-level root

The repository assumes datasets are already mounted and readable. A conceptual Linux layout is:

```text
/data/endoscopy/
|-- SCARED/
|-- EndoNeRF/
`-- C3VD/
```

Set the canonical parent root for the current shell:

```bash
export RELIABLE_ENDO_DATA_ROOT=/data/endoscopy
reg data resolve --config configs/data/scared.yaml
reg data validate --config configs/data/scared.yaml
```

Resolution prints the selected external path. Validation returns structured JSON and exits nonzero when the dataset is absent or the root is not a directory. An existing directory passes only basic validation because internal layouts remain intentionally unimplemented.

## Explicit root override

For an alternate mounted location, pass an absolute parent root without modifying committed configuration:

```bash
reg data resolve --config configs/data/scared.yaml --data-root /scratch/project-data
reg data validate --config configs/data/scared.yaml --data-root /scratch/project-data
```

The explicit root takes precedence over `RELIABLE_ENDO_DATA_ROOT`. A dataset YAML may also contain an absolute `root` as an intentional machine-local override, but such paths should not be committed.

## Windows PowerShell

Activate the project environment so its `reg` command precedes the Windows Registry utility, then set:

```powershell
$env:RELIABLE_ENDO_DATA_ROOT = "D:\datasets\endoscopy"
reg data resolve --config configs/data/scared.yaml
reg data validate --config configs/data/scared.yaml
```

Windows paths are supported but are not canonical repository defaults.

## Switching servers safely

Keep `configs/data/*.yaml` portable and change only the environment variable or explicit CLI override. Do not store credentials, SSH configuration, cloud URLs, or mount commands in dataset YAML. This repository does not implement SSH, S3, Google Drive, network mounting, or dataset downloading.

