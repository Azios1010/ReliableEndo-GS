"""Bounded-memory CPU validation of an A04 role's complete deployment manifest."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

from reliable_endo_gs.data.scared_a00 import A01Manifest, ScaredA00Adapter
from reliable_endo_gs.teachers.roles import A01ScaredDataRoles


@dataclass(frozen=True)
class _PreflightChunk:
    root: Path
    manifest: A01Manifest
    roles: A01ScaredDataRoles


def _check_chunk(chunk: _PreflightChunk) -> list[dict[str, object]]:
    # Each spawned CPU worker owns its adapter and tensor runtime. No CUDA
    # context is inherited, and no mutable scientific globals are introduced.
    import torch

    torch.set_num_threads(1)
    adapter = ScaredA00Adapter(chunk.root, chunk.manifest, roles=chunk.roles)
    results: list[dict[str, object]] = []
    for entry in chunk.manifest.entries:
        try:
            result = adapter.validate_deployment_files(entry.sample_id, data_role=entry.data_role)
        except (OSError, ValueError) as error:
            result = {
                "sample_id": entry.sample_id,
                "error": str(error),
                "error_type": type(error).__name__,
            }
        results.append(result)
    return results


def preflight_a04_data(
    adapter: ScaredA00Adapter,
    *,
    data_role: str = "final_train",
    workers: int = 4,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Validate every selected entry, keeping GT separate and split/order unchanged.

    Each worker decodes at most one stereo pair at a time on CPU. No teacher,
    model, GPU, target cache or supervision file is accessed.
    """

    adapter.roles.datasets_for_role(data_role)
    if type(workers) is not int or workers < 1:
        raise ValueError("preflight workers must be a positive integer")
    entries = tuple(e for e in adapter.manifest.entries if e.data_role == data_role)
    if not entries:
        raise ValueError(f"no samples in role {data_role}")

    errors: list[dict[str, object]] = []
    datasets: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    # Small immutable metadata chunks bound IPC payloads and retained results.
    # These are in-memory views of the manifest, never dataset copies/splits.
    chunks = tuple(
        _PreflightChunk(
            adapter.root,
            A01Manifest(adapter.manifest.role_contract_hash, entries[start : start + 64]),
            adapter.roles,
        )
        for start in range(0, len(entries), 64)
    )
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        # Flatten results in manifest order. No optimizer/data cursor is mutated.
        results = (r for chunk in executor.map(_check_chunk, chunks) for r in chunk)
        for index, result in enumerate(results, start=1):
            datasets[str(result["sample_id"]).split("/", 1)[0]] += 1
            if "error" in result:
                errors.append(result)
            else:
                shapes[str(result["image_hw"])] += 1
            if progress is not None and (index % 250 == 0 or index == len(entries)):
                progress(
                    {
                        "event": "preflight_progress",
                        "checked": index,
                        "total": len(entries),
                        "errors": len(errors),
                    }
                )
    return {
        "status": "PASS" if not errors else "FAIL",
        "data_role": data_role,
        "data_root": str(adapter.root.resolve()),
        "manifest_hash": adapter.manifest.manifest_hash,
        "sample_count": len(entries),
        "dataset_counts": dict(datasets),
        "image_shapes": dict(shapes),
        "errors": errors,
        "ground_truth_loaded": False,
        "teacher_inference_executed": False,
        "training_executed": False,
    }
