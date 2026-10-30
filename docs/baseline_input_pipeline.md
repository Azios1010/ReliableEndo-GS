# Supervised baseline input staging

The matched A00 baseline can stage a bounded window of compressed files on the
Linux filesystem while the current sample trains on the GPU. This changes input
I/O scheduling only. The original loader still decodes RGB, builds calibration,
resizes tensors and loads separate disparity supervision. Its model, loss,
precision, batch size, sample permutation, optimizer and update horizons are
unchanged.

`baseline/training/input_pipeline.py` provides `StagedInputPrefetch`. Byte-copy
workers call the existing `FileStager`, which verifies SHA-256 before exposing a
file through an atomic rename. A separate bounded pool calls the original tensor
loader. The consumer waits for futures in absolute update order. An initial
block can be prepared before training starts; subsequent staging overlaps tensor
loading and training without reloading the model at block boundaries.

`supervised_training_file_index` adds disparity TIFFs to the deployment file
index for supervised baseline training only. The deployment/KD index remains
GT-free. The input pipeline does not attach GT to deployment tensors or change
the manifest or excluded-sample list. A missing/corrupt member raises before that
sample's update; there is no silent source fallback or sample skip.

Runtime configuration specifies source and staging roots, the pinned file
inventory, checksum journal, byte lookahead, initial block size, I/O workers and
tensor lookahead/workers. `FileStager.copy_buffer_bytes` controls the bounded
source-read buffer; its 64 KiB default preserves the existing KD path. A baseline
continuation can explicitly select 1 MiB reads without changing copied bytes or
checksums. Data and runtime outputs stay outside the repository.
Copy buffers and tensor futures are bounded independently; compressed files
remain on Linux for reuse, and consumed clean pages receive per-file release
advice. No global cache flush is used.

An existing training artifact is continued in a separate run directory. Record
the old/new source and driver hashes, exact source-file changes, input settings,
parent checkpoint hash and new checkpoint hash. Verify that all model and
optimizer tensors, scheduler/scaler state, counters and RNG bytes are identical
across the transfer. Preserve the original run and its comparison protocol; do
not disable strict source/driver identity checks or rewrite old metrics.

Validation covers actual worker overlap, bounded queues, resume/epoch order,
checksum failures and GT separation. Before promoting a runtime continuation,
compare real deployment/supervision tensor bytes and metadata, then bounded
stage-1 and stage-2 updates from the same checkpoint. Include an unchanged
reference repeat if CUDA updates differ, so existing numerical nondeterminism
is distinguished from changed inputs or scientific code. Report measured
differences, rather than promising bitwise final training results from a bounded
check.

Measure whole-update wall time and input waiting over a sustained window, along
with initial fill, hardware and memory. Report cold staging separately from
Linux-file reuse. A producer slower than the GPU still limits the first copy
pass; lookahead cannot remove a sustained storage-bandwidth limit. Retained
Linux copies are reused on later epochs. These training timings do not
establish inference FPS.
