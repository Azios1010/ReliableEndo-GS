"""Staging integrity, bounded retry and KD/GT separation contracts."""

import errno
import json

import pytest

from reliable_endo_gs.data.staging import FileStager, deployment_file_index


def test_byte_copy_reuse_and_corruption_rejection(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "linux"
    source.mkdir()
    payload = bytes(range(256)) * 2000
    (source / "image.png").write_bytes(payload)
    journal = tmp_path / "journal.jsonl"
    stage = FileStager(source, destination, journal=journal, reserve_bytes=0)
    stage.prepare(["image.png"])
    assert (destination / "image.png").read_bytes() == payload
    assert not (destination / "image.png.prefetch-partial").exists()
    record = json.loads(journal.read_text())
    assert record["sha256"] == FileStager.digest(source / "image.png")
    (source / "image.png").unlink()
    restarted = FileStager(source, destination, journal=journal, reserve_bytes=0)
    restarted.prepare(["image.png"])  # verified Linux copies require no Windows re-read
    (destination / "image.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        FileStager(source, destination, journal=journal, reserve_bytes=0).prepare(["image.png"])
    assert (destination / "image.png").read_bytes() == b"corrupt"


def test_transient_memory_error_has_bounded_retry(tmp_path, monkeypatch):
    stage = FileStager(
        tmp_path / "source", tmp_path / "dest", journal=tmp_path / "journal", retries=2
    )
    calls = []
    sleeps = []
    events = []

    def prepare(relative):
        calls.append(relative)
        if len(calls) <= 2:
            raise OSError(errno.ENOMEM, "host pressure")

    monkeypatch.setattr(stage, "_prepare_one", prepare)
    monkeypatch.setattr("reliable_endo_gs.data.staging.time.sleep", sleeps.append)
    stage.event = lambda event, **values: events.append((event, values))
    stage.prepare(["image.png"])
    assert len(calls) == 3 and sleeps == [1, 2]
    assert len(events) == 2

    def permanent(relative):
        raise OSError(errno.ENOMEM, "still full")

    monkeypatch.setattr(stage, "_prepare_one", permanent)
    with pytest.raises(OSError, match="still full"):
        stage.prepare(["other.png"])
    assert sleeps == [1, 2, 1, 2]


def test_nontransient_error_is_not_hidden(tmp_path, monkeypatch):
    stage = FileStager(tmp_path / "a", tmp_path / "b", journal=tmp_path / "j")

    def fail(relative):
        raise OSError(errno.EIO, "bad disk")

    monkeypatch.setattr(stage, "_prepare_one", fail)
    with pytest.raises(OSError, match="bad disk"):
        stage.prepare(["a"])
    with pytest.raises(ValueError, match="Unsafe"):
        stage.prepare(["../escape"])


def test_deployment_index_excludes_gt_and_test_data(tmp_path):
    sample = "dataset_3/keyframe_3/frame_data000104"
    base = "dataset_3/keyframe_3/data"
    files = [
        f"{base}/{folder}/frame_data000104.{ext}"
        for folder, ext in (
            ("left_finalpass", "png"),
            ("right_finalpass", "png"),
            ("frame_data", "json"),
            ("newpram_data", "json"),
            ("disparity", "tiff"),
        )
    ]
    files.append(f"{base}/a00_protocol.json")
    inventory = tmp_path / "files.txt"
    inventory.write_text("\n".join(files))
    result = deployment_file_index([sample], inventory)[0]
    assert len(result) == 5 and not any("disparity" in path for path in result)
    with pytest.raises(ValueError, match="TRAIN1-7"):
        deployment_file_index([sample.replace("dataset_3", "dataset_8")], inventory)
    inventory.write_text("../escape")
    with pytest.raises(ValueError, match="safe relative"):
        deployment_file_index([sample], inventory)


def test_release_advises_only_own_file_without_deleting_bytes(tmp_path, monkeypatch):
    import os

    source, destination = tmp_path / "source", tmp_path / "linux"
    destination.mkdir()
    (destination / "a").write_bytes(b"unchanged")
    stage = FileStager(source, destination, journal=tmp_path / "j")
    calls = []
    monkeypatch.setattr(os, "posix_fadvise", lambda *args: calls.append(args))
    stage.release(["a"])
    assert len(calls) == 1 and (destination / "a").read_bytes() == b"unchanged"
