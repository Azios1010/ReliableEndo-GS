"""Batched scalar readback equals per-scalar reads and rejects non-scalars."""

import pytest
import torch

from reliable_endo_gs.utils.readback import read_bools, read_ints, read_scalars


def test_counts_and_flags_match_individual_reads():
    mask = torch.rand(7, 11) > 0.4
    finite = torch.isfinite(torch.randn(5, 5))
    counts = [mask.sum(), (~mask).sum(), (mask & mask.roll(1, 0)).sum()]
    assert read_ints(*counts) == [int(c.item()) for c in counts]
    assert read_bools(finite.all(), (~finite).any()) == [bool(finite.all()), bool((~finite).any())]
    assert read_scalars([]) == [] and read_ints() == []


def test_large_exact_counts_and_dtype_mixing():
    big = torch.tensor(2**40 + 1, dtype=torch.int64)
    assert read_ints(big, torch.tensor(3), torch.tensor(True)) == [2**40 + 1, 3, 1]
    assert read_scalars([torch.tensor(0.5), torch.tensor([2])]) == [0.5, 2.0]


def test_non_scalar_is_rejected():
    with pytest.raises(ValueError, match="single-element"):
        read_scalars([torch.ones(2)])


def test_raise_first_failure_orders_checks_and_builds_messages_lazily():
    from reliable_endo_gs.utils.readback import raise_first_failure

    built: list[str] = []

    def make(text: str):
        def factory():
            built.append(text)
            return ValueError(text)

        return factory

    raise_first_failure([])
    raise_first_failure([(torch.tensor(True), make("a")), (torch.tensor(True), make("b"))])
    assert built == []  # nothing is built while every check passes
    with pytest.raises(ValueError, match="second"):
        raise_first_failure(
            [
                (torch.tensor(True), make("first")),
                (torch.tensor(False), make("second")),
                (torch.tensor(False), make("third")),
            ]
        )
    assert built == ["second"]


def test_deferred_validation_collects_and_raises_once_at_scope_exit_in_order():
    from reliable_endo_gs.utils.readback import deferred_validation, raise_first_failure

    reached: list[str] = []
    with pytest.raises(ValueError, match="second"):
        with deferred_validation():
            raise_first_failure([(torch.tensor(False), lambda: ValueError("second"))])
            reached.append("body continues after the failed check")  # execution is not interrupted
            raise_first_failure([(torch.tensor(False), lambda: ValueError("third"))])
    assert reached == ["body continues after the failed check"]
    with deferred_validation():  # passing checks leave no trace and the scope is reusable
        raise_first_failure([(torch.tensor(True), lambda: ValueError("never"))])
    with pytest.raises(ValueError, match="immediate"):  # outside a scope checks stay immediate
        raise_first_failure([(torch.tensor(False), lambda: ValueError("immediate"))])


def test_deferred_validation_prefers_the_validation_error_over_a_downstream_symptom():
    from reliable_endo_gs.utils.readback import deferred_validation, raise_first_failure

    with pytest.raises(ValueError, match="bad input") as caught:
        with deferred_validation():
            raise_first_failure([(torch.tensor(False), lambda: ValueError("bad input"))])
            raise RuntimeError("downstream symptom")
    assert isinstance(caught.value.__cause__, RuntimeError)
    with pytest.raises(
        RuntimeError, match="unrelated"
    ):  # no failed check: the body's error is untouched
        with deferred_validation():
            raise_first_failure([(torch.tensor(True), lambda: ValueError("fine"))])
            raise RuntimeError("unrelated")


def test_nested_deferred_scopes_join_the_outermost():
    from reliable_endo_gs.utils.readback import deferred_validation, raise_first_failure

    with pytest.raises(ValueError, match="late"):
        with deferred_validation():
            with deferred_validation():
                raise_first_failure([(torch.tensor(False), lambda: ValueError("late"))])
            # the inner exit must not have evaluated anything yet


def test_packed_checks_keep_registration_order_and_the_first_failure():
    from reliable_endo_gs.utils.readback import PackedChecks, pack_checks, raise_first_failure

    def make(text: str):
        return lambda: ValueError(text)

    checks = [
        (torch.tensor(True), make("a")),
        (torch.tensor(False), make("b")),
        (torch.tensor(False), make("c")),
    ]
    packed = pack_checks(checks)
    assert isinstance(packed, PackedChecks) and packed.flags.shape == (3,)
    with pytest.raises(ValueError, match="^b$"):
        raise_first_failure([packed])
    raise_first_failure([pack_checks(checks[:1])])
    # a packed entry mixes with plain ones and keeps the global order
    with pytest.raises(ValueError, match="^plain$"):
        raise_first_failure([(torch.tensor(False), make("plain")), packed])


def test_collected_checks_are_not_evaluated_and_cannot_nest_in_a_deferred_scope():
    from reliable_endo_gs.utils.readback import (
        collected_checks,
        deferred_validation,
        raise_first_failure,
    )

    with collected_checks() as pending:
        raise_first_failure([(torch.tensor(False), lambda: ValueError("never raised here"))])
    assert len(pending) == 1  # collected for a later re-registration, not evaluated
    with deferred_validation():
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with collected_checks():
                pass


def test_capture_placeholders_are_refused_outside_a_graph_capture():
    from reliable_endo_gs.utils.readback import graph_capture_diagnostics

    with graph_capture_diagnostics():
        with pytest.raises(RuntimeError, match="only valid while a CUDA graph"):
            read_ints(torch.tensor(3))
    assert read_ints(torch.tensor(3)) == [3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_counts_are_placeholders_inside_a_capture_and_real_outside():
    from reliable_endo_gs.utils.readback import graph_capture_diagnostics

    x = torch.ones(8, device="cuda", dtype=torch.bool)
    read_ints(x.sum())  # warm up outside the capture
    graph = torch.cuda.CUDAGraph()
    with graph_capture_diagnostics(), torch.cuda.graph(graph):
        assert read_ints(x.sum(), x.sum()) == [0, 0]
    assert read_ints(x.sum()) == [8]
