"""Synthetic callable tests for the synchronized profiling protocol."""

import pytest

from reliable_endo_gs.profiling import ProfilingError, profile_callable


def test_profile_callable_discards_warmup_and_synchronizes() -> None:
    work_calls = 0
    sync_calls = 0

    def work() -> int:
        nonlocal work_calls
        work_calls += 1
        return work_calls

    def sync() -> None:
        nonlocal sync_calls
        sync_calls += 1

    profile, output = profile_callable(work, warmup=2, repetitions=3, synchronize=sync)
    assert output == 5
    assert profile.repetitions == 3
    assert len(profile.latencies_ms) == 3
    assert work_calls == 5
    assert sync_calls == 8
    assert profile.p90_ms >= profile.median_ms
    assert profile.fps > 0


def test_profile_callable_rejects_invalid_protocol() -> None:
    with pytest.raises(ProfilingError):
        profile_callable(lambda: None, repetitions=0)


def test_profile_callable_reports_injected_memory_protocol() -> None:
    events: list[str] = []

    def reset() -> None:
        events.append("reset")

    def peak() -> int:
        events.append("peak")
        return 123

    profile, _ = profile_callable(
        lambda: None,
        warmup=1,
        repetitions=2,
        reset_peak_memory=reset,
        peak_memory=peak,
    )
    assert events == ["reset", "peak"]
    assert profile.peak_memory_bytes == 123
