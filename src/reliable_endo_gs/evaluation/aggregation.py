"""Explicit, valid-count-weighted metric aggregation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reliable_endo_gs.evaluation.stereo import MetricResult


class AggregationError(ValueError):
    """Raised for incompatible metric records or ambiguous aggregation."""


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """A metric summary retaining its denominator and record count."""

    name: str
    value: float | None
    valid_count: int
    total_count: int
    record_count: int
    definition: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "valid_count": self.valid_count,
            "total_count": self.total_count,
            "record_count": self.record_count,
            "definition": self.definition,
        }


def aggregate_metric_results(results: Iterable[MetricResult]) -> MetricSummary:
    """Aggregate scalar metrics by valid-pixel count, preserving empty cases."""

    records = tuple(results)
    if not records:
        raise AggregationError("at least one metric result is required")
    first = records[0]
    if any(record.name != first.name for record in records):
        raise AggregationError("all metric results must have the same name")
    if any(record.definition != first.definition for record in records):
        raise AggregationError("all metric results must have the same definition")
    valid_count = sum(record.valid_count for record in records)
    total_count = sum(record.total_count for record in records)
    weighted = sum(
        record.value * record.valid_count
        for record in records
        if record.value is not None and record.valid_count
    )
    value = weighted / valid_count if valid_count else None
    return MetricSummary(
        first.name, value, valid_count, total_count, len(records), first.definition
    )


def aggregate_metrics(results: Mapping[str, Sequence[MetricResult]]) -> dict[str, MetricSummary]:
    """Aggregate each named metric independently."""

    return {name: aggregate_metric_results(values) for name, values in results.items()}


def aggregate_by_sequence(
    sequence_ids: Sequence[str], metrics: Sequence[Mapping[str, MetricResult]]
) -> dict[str, dict[str, MetricSummary]]:
    """Aggregate sample-level records per sequence, then callers may aggregate globally.

    ``sequence_ids`` and ``metrics`` are parallel and retain input order in the
    returned mapping. No sequence weighting is inferred; downstream aggregation
    remains explicit and valid-count weighted.
    """

    if len(sequence_ids) != len(metrics):
        raise AggregationError("sequence_ids and metrics must have equal lengths")
    grouped: dict[str, dict[str, list[MetricResult]]] = {}
    for sequence_id, record in zip(sequence_ids, metrics, strict=True):
        if not isinstance(sequence_id, str) or not sequence_id:
            raise AggregationError("sequence IDs must be non-empty strings")
        target = grouped.setdefault(sequence_id, {})
        for name, result in record.items():
            if result.name != name:
                raise AggregationError(
                    f"metric key {name!r} does not match result name {result.name!r}"
                )
            target.setdefault(name, []).append(result)
    return {
        sequence_id: {name: aggregate_metric_results(values) for name, values in records.items()}
        for sequence_id, records in grouped.items()
    }


aggregate = aggregate_metric_results
aggregate_sequence_metrics = aggregate_by_sequence
