"""Bounded byte-copy staging; image decoding and all tensor operations stay canonical."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path


def deployment_file_index(sample_ids: Sequence[str], inventory: Path) -> list[tuple[str, ...]]:
    """Resolve only RGB, camera metadata and protocol; never stage GT for KD."""
    files = set(inventory.read_text().splitlines())
    if any(Path(f).is_absolute() or ".." in Path(f).parts for f in files):
        raise ValueError("Staging inventory must contain safe relative paths")
    result = []
    for sample_id in sample_ids:
        parts = sample_id.split("/")
        if len(parts) != 3 or parts[0] not in {f"dataset_{i}" for i in range(1, 8)}:
            raise ValueError("Staging accepts original TRAIN1-7 sample IDs only")
        if any(part in ("", ".", "..") for part in parts):
            raise ValueError("Invalid staging sample ID")
        dataset, keyframe, frame = parts
        sequence = f"{dataset}/{keyframe}"
        base = (
            sequence + "/data"
            if f"{sequence}/data/left_finalpass/{frame}.png" in files
            else sequence
        )
        required = [
            f"{base}/{folder}/{frame}.{ext}"
            for folder, ext in (
                ("left_finalpass", "png"),
                ("right_finalpass", "png"),
                ("frame_data", "json"),
                ("newpram_data", "json"),
            )
        ]
        for parent in dict.fromkeys((sequence, base)):
            for name in ("a00_protocol.json", "a00_provenance.json"):
                relative = f"{parent}/{name}"
                if relative in files:
                    required.append(relative)
        if not set(required).issubset(files):
            raise ValueError(f"Missing sample members in pinned inventory: {sample_id}")
        result.append(tuple(required))
    return result


class FileStager:
    """Stream byte-identical files to Linux with bounded buffers and retry telemetry.

    No tensors are allocated here. Each worker uses one bounded copy buffer
    (64 KiB by default). Files become visible only after SHA256 verification and
    atomic rename. Journalled files are re-hashed on reuse; mismatches fail closed.
    """

    def __init__(
        self,
        source: Path,
        destination: Path,
        *,
        journal: Path,
        previous_journals: Sequence[Path] = (),
        retries: int = 6,
        reserve_bytes: int = 8 * 1024**3,
        copy_buffer_bytes: int = 64 * 1024,
        event: Callable[..., None] | None = None,
    ) -> None:
        if retries < 0 or reserve_bytes < 0 or copy_buffer_bytes < 1:
            raise ValueError("Invalid staging retry/reserve limit")
        self.source, self.destination = source.resolve(), destination.resolve()
        if self.source == self.destination:
            raise ValueError("Staging source and destination must differ")
        self.journal, self.retries, self.reserve_bytes, self.event = (
            journal,
            retries,
            reserve_bytes,
            event,
        )
        self.copy_buffer_bytes = copy_buffer_bytes
        self._known: dict[str, tuple[int, str]] = {}
        self._checked: set[str] = set()
        self._locks = [threading.Lock() for _ in range(64)]
        self._journal_lock = threading.Lock()
        for path in (*previous_journals, journal):
            if path.exists():
                with path.open() as handle:
                    for line in handle:
                        record = json.loads(line)
                        self._known[record["relative_path"]] = (record["bytes"], record["sha256"])

    @staticmethod
    def digest(path: Path) -> str:
        """SHA256 using a fixed 64 KiB host buffer."""
        result = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(64 * 1024), b""):
                result.update(block)
        return result.hexdigest()

    def prepare(self, relatives: Sequence[str]) -> None:
        """Materialize one sample's files; concurrent calls may share protocol files."""
        for relative in relatives:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe staging path")
            # Striped locks bound synchronization state independently of dataset size.
            with self._locks[hash(relative) % len(self._locks)]:
                if relative in self._checked:
                    # Warm future samples again on later epochs after consumed pages
                    # were released; no unbounded tensor/RAM cache is retained.
                    if self.digest(self.destination / relative) != self._known[relative][1]:
                        raise ValueError(f"Previously verified Linux file changed: {relative}")
                    continue
                for attempt in range(self.retries + 1):
                    try:
                        self._prepare_one(relative)
                        self._checked.add(relative)
                        break
                    except OSError as error:
                        transient = error.errno in (errno.ENOMEM, errno.EAGAIN, errno.EINTR)
                        if not transient or attempt == self.retries:
                            raise
                        delay = min(2**attempt, 16)
                        if self.event is not None:
                            self.event(
                                "staging_retry",
                                relative_path=relative,
                                errno=error.errno,
                                attempt=attempt + 1,
                                delay_seconds=delay,
                            )
                        time.sleep(delay)

    def release(self, relatives: Sequence[str]) -> None:
        """Release this job's consumed clean file pages, retaining bytes on disk.

        This is per-file advice, never a system-wide cache flush. Decoded tensors
        own their storage and do not depend on compressed-image file mappings.
        """
        if not hasattr(os, "posix_fadvise"):
            return
        for relative in relatives:
            with (self.destination / relative).open("rb", buffering=0) as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    def _prepare_one(self, relative: str) -> None:
        source, target = self.source / relative, self.destination / relative
        known = self._known.get(relative)
        if target.is_file():
            size = target.stat().st_size
            digest = self.digest(target)
            if known is not None:
                if (size, digest) != known:
                    raise ValueError(f"Previously verified Linux file changed: {relative}")
                return
            if size != source.stat().st_size or digest != self.digest(source):
                raise ValueError(f"Existing destination differs: {relative}")
            self._record(relative, size, digest)
            return
        before = source.stat()
        target.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(target.parent).free < before.st_size + self.reserve_bytes:
            raise OSError(errno.ENOSPC, "Linux staging reserve would be exhausted")
        partial = target.with_name(target.name + ".prefetch-partial")
        result = hashlib.sha256()
        with source.open("rb") as reader, partial.open("wb") as writer:
            for block in iter(lambda: reader.read(self.copy_buffer_bytes), b""):
                result.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        after = source.stat()
        digest = result.hexdigest()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changed during staging: {relative}")
        if partial.stat().st_size != before.st_size or self.digest(partial) != digest:
            raise ValueError(f"Staging checksum mismatch: {relative}")
        shutil.copystat(source, partial)
        os.replace(partial, target)
        self._record(relative, before.st_size, digest)
        if self.event is not None:
            self.event("file_staged", relative_path=relative, bytes=before.st_size)

    def _record(self, relative: str, size: int, digest: str) -> None:
        record = {"relative_path": relative, "bytes": size, "sha256": digest}
        with self._journal_lock:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            self._known[relative] = (size, digest)
