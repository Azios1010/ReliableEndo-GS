# Third-party source boundary

`endo_e2e_gs/` is the unmodified official
`Intelligent-Imaging-Center/Endo-E2E-GS` Git submodule pinned by the parent
repository. Initialize it with:

```bash
git submodule update --init --recursive
```

The selected revision is `186fa2b4a2159b28393492f6df1aa444b54391a8`.
Its MIT license and copyright notice remain in
`third_party/endo_e2e_gs/LICENSE`. The submodule must remain clean; project
compatibility code belongs under `src/reliable_endo_gs/baseline/`, and any
unavoidable future patch must follow `docs/third_party_integration.md`.

No dataset, checkpoint, compiled extension, or generated render is vendored
here. Source-code licensing does not establish dataset or checkpoint terms.
