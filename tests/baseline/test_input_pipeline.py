"""Real worker overlap, byte equality, resume order and failure isolation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path

from reliable_endo_gs.baseline.training.input_pipeline import (
    StagedInputPrefetch,
    supervised_training_file_index,
)
from reliable_endo_gs.data.staging import FileStager, deployment_file_index


class InputPipelineTests(unittest.TestCase):
    def test_copy_buffer_size_cannot_change_staged_bytes_or_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            payload = bytes(range(256)) * 8193
            (source / "frame.tiff").write_bytes(payload)
            for size in (64 * 1024, 1024 * 1024):
                destination = root / f"buffer-{size}"
                journal = root / f"journal-{size}"
                stager = FileStager(
                    source, destination, journal=journal, reserve_bytes=0, copy_buffer_bytes=size
                )
                stager.prepare(["frame.tiff"])
                self.assertEqual((destination / "frame.tiff").read_bytes(), payload)
                self.assertEqual(
                    json.loads(journal.read_text())["sha256"], hashlib.sha256(payload).hexdigest()
                )
            with self.assertRaisesRegex(ValueError, "Invalid staging"):
                FileStager(source, root / "invalid", journal=root / "j", copy_buffer_bytes=0)

    def test_supervision_is_separate_and_missing_gt_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = "dataset_3/keyframe_1/frame_data000001"
            base = "dataset_3/keyframe_1/data"
            files = [
                f"{base}/{folder}/frame_data000001.{extension}"
                for folder, extension in (
                    ("left_finalpass", "png"),
                    ("right_finalpass", "png"),
                    ("frame_data", "json"),
                    ("newpram_data", "json"),
                    ("disparity", "tiff"),
                )
            ]
            inventory = root / "files.txt"
            inventory.write_text("\n".join(files))
            supervised = supervised_training_file_index([sample], inventory)[sample]
            self.assertEqual(supervised[:-1], deployment_file_index([sample], inventory)[0])
            self.assertEqual(supervised[-1], files[-1])
            inventory.write_text("\n".join(files[:-1]))
            with self.assertRaisesRegex(ValueError, "Missing supervised"):
                supervised_training_file_index([sample], inventory)
            with self.assertRaisesRegex(ValueError, "TRAIN1-7"):
                supervised_training_file_index(
                    [sample.replace("dataset_3", "dataset_8")], inventory
                )

    def test_real_copies_keep_bytes_order_and_bounded_resume_across_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source", root / "linux"
            source.mkdir()
            values = {f"sample-{i}": bytes([i]) * 90000 for i in range(5)}
            for name, value in values.items():
                (source / name).write_bytes(value)
            stager = FileStager(source, destination, journal=root / "journal", reserve_bytes=0)
            order = ["sample-4", "sample-1", "sample-3", "sample-0", "sample-2"]
            pipeline = StagedInputPrefetch(
                sample_id_at=lambda k: order[k % len(order)],
                load=lambda sid: (sid, (destination / sid).read_bytes()),
                members={sid: (sid,) for sid in order},
                stager=stager,
                start=3,
                tensor_ahead=2,
                tensor_workers=2,
                file_ahead=4,
                io_workers=2,
                initial_samples=2,
            )
            try:
                for k in range(3, 15):
                    sid, value = pipeline.get()
                    self.assertEqual(sid, order[k % len(order)])
                    self.assertEqual(value, values[sid])
                self.assertEqual(pipeline.expected, 15)
                self.assertEqual(pipeline.consumed, 12)
                self.assertLessEqual(pipeline.peak_files_pending, 4)
                self.assertLessEqual(pipeline.peak_tensors_pending, 2)
            finally:
                pipeline.close()
            pipeline.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                pipeline.get()

    def test_next_io_overlaps_consumer_without_reordering(self) -> None:
        started, release = threading.Event(), threading.Event()

        class Stager:
            def prepare(self, members):
                if members == ("1",):
                    started.set()
                    if not release.wait(5):
                        raise TimeoutError("consumer did not release next I/O")

            def release(self, members):
                pass

        pipeline = StagedInputPrefetch(
            sample_id_at=lambda k: str(k % 3),
            load=lambda sid: sid,
            members={str(k): (str(k),) for k in range(3)},
            stager=Stager(),
            start=0,
            tensor_ahead=1,
            tensor_workers=1,
            file_ahead=3,
            io_workers=2,
            initial_samples=1,
        )
        try:
            self.assertEqual(pipeline.get(), "0")
            self.assertTrue(started.wait(5))
            release.set()
            self.assertEqual(pipeline.get(), "1")
        finally:
            release.set()
            pipeline.close()

    def test_staging_error_never_loads_or_skips_the_failed_sample(self) -> None:
        loaded = []

        class Stager:
            def prepare(self, members):
                if members == ("1",):
                    raise OSError("checksum failed")

            def release(self, members):
                pass

        pipeline = StagedInputPrefetch(
            sample_id_at=lambda k: str(k % 2),
            load=lambda sid: loaded.append(sid),
            members={"0": ("0",), "1": ("1",)},
            stager=Stager(),
            start=0,
            tensor_ahead=1,
            tensor_workers=1,
            file_ahead=2,
            initial_samples=0,
        )
        try:
            pipeline.get()
            for _ in range(2):
                with self.assertRaisesRegex(OSError, "checksum failed"):
                    pipeline.get()
                self.assertEqual(pipeline.expected, 1)
            self.assertEqual(loaded, ["0"])
        finally:
            pipeline.close()

    def test_invalid_limits_fail_before_creating_workers(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid staging"):
            StagedInputPrefetch(
                sample_id_at=str,
                load=str,
                members={},
                stager=None,
                start=0,
                tensor_ahead=4,
                file_ahead=2,
            )


if __name__ == "__main__":
    unittest.main()
