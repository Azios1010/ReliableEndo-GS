# Continuous KD input pipeline

The optional `--stage-from ROOT --stage-inventory FILE` path keeps one student,
Adam state and frozen teacher alive for the entire training run. It prepares
future sample files in background threads while the main thread executes the
unchanged A00 tensor loader, teacher/cache provider and training step. It does
not change the manifest, sample order, resolution, precision, losses or epoch
budget. Current support is ordered student-only training (`joint_epochs: 0`),
without intervention configuration.

`--data-root` is the Linux staging destination. The pinned inventory lists
relative source members. `--staging-journal FILE` may be repeated to reuse
previous SHA256-verified copies. The required members are stereo RGB, frame and
camera metadata, and available protocol/provenance sidecars. GT disparity is
not copied or exposed to KD.

Defaults are 64 future samples and two I/O workers, configurable with
`--prefetch-samples` and `--staging-workers`. Workers use fixed 64 KiB streaming
buffers; they never construct CUDA tensors, decode images, modify RNG state or
run models. New files are checksummed, synced and atomically renamed. Reused
files are verified and warmed in the operating-system page cache. After the
canonical loader owns the decoded tensors, per-file `POSIX_FADV_DONTNEED` advice
releases consumed clean image-file pages. Files remain on disk for later
epochs. No global cache flush is used.

Transient `ENOMEM`, `EAGAIN` and `EINTR` file errors receive bounded backoff
(`--staging-retries`, default six). Other errors and checksum mismatches stop
training before consuming that sample. Canonical periodic checkpoints remain
the recovery authority; no update is silently skipped. A retry does not rebuild
the teacher cache. Device selection uses `CUDA_VISIBLE_DEVICES`; physical GPU1
remains logical `cuda:0` inside the process.

`OUTPUT.input.jsonl` records staging/retry events and cumulative input waiting
while the run is active. `OUTPUT/input_pipeline.json` summarizes a completed or
gracefully stopped run. Report GPU utilization and wall time separately from
per-step compute timing: storage throughput can still limit sustained training,
and a full GPU buffer cannot hide an indefinitely slower producer.

Checkpoint identities retain strict source and hardware checks. An I/O-only
source update or GPU move requires a new, explicitly traced checkpoint copy:
record both source hashes, the precise changed source files and GPU UUID, plus
both checkpoint hashes. Preserve the original checkpoint and verify every
weight, optimizer tensor, counter and RNG field against it. Do not rewrite old
metrics or disable normal compatibility validation.
