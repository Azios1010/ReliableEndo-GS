"""Actual worker overlap, resume order and bounded queue tests without CUDA."""

import threading

import pytest

from reliable_endo_gs.training.input_prefetch import OrderedInputPrefetch


def test_resume_wrap_preserves_values_and_bounded_queue():
    values = [object() for _ in range(5)]
    prepared = []
    with OrderedInputPrefetch(values, prepared.append, start=3, ahead=2, workers=1) as samples:
        assert len(samples) == 5
        for index in [3, 4, 0, 1, 2, 3, 4]:
            assert samples[index] is values[index]
        assert samples.consumed == 7 and samples.peak_pending == 2
    assert prepared[:7] == [3, 4, 0, 1, 2, 3, 4]


def test_io_of_next_sample_overlaps_current_consumer():
    entered = threading.Event()
    release = threading.Event()

    def prepare(index):
        if index == 1:
            entered.set()
            assert release.wait(5)

    with OrderedInputPrefetch(["current", "next", "last"], prepare, ahead=2) as samples:
        try:
            assert samples[0] == "current"
            assert entered.wait(5)  # sample 1 I/O is active while consumer owns sample 0
        finally:
            release.set()
        assert samples[1] == "next"


def test_prefetch_error_prevents_loading_or_skipping_sample():
    loaded = []

    class Values:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            loaded.append(index)
            return index

    def prepare(index):
        if index == 2:
            raise OSError("copy failed")

    with OrderedInputPrefetch(Values(), prepare, start=1, ahead=2) as samples:
        assert samples[1] == 1
        with pytest.raises(OSError, match="copy failed"):
            samples[2]
        assert loaded == [1] and samples.expected == 2
        with pytest.raises(ValueError, match="sequential"):
            samples[3]


def test_input_limits_and_context_are_explicit():
    with pytest.raises(ValueError):
        OrderedInputPrefetch([], lambda i: None)
    samples = OrderedInputPrefetch([0], lambda i: None, ahead=10)
    assert samples.ahead == 1
    with pytest.raises(RuntimeError, match="context"):
        samples[0]


def test_consumed_file_release_and_metrics_preserve_order():
    released, events = [], []
    with OrderedInputPrefetch(
        [1, 2],
        lambda i: None,
        release=released.append,
        event=lambda event, **fields: events.append((event, fields)),
    ) as samples:
        assert samples[0] == 1
        assert released == [0]
        assert events[0][0] == "sample_input_ready"
        assert events[0][1]["consumed"] == 1
