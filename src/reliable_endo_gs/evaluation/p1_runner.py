"""Development-only P1 proxy measurement runner.

The runner loads the frozen deterministic Stage-2 network, retains the pinned
RAFT iteration stack, writes proxy maps and oracle disparity targets, and
never touches the final evaluation sequences.  It does not train, calibrate,
render, or initialize a W&B scientific run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from reliable_endo_gs.evaluation.p1 import evaluate_p1_proxy, oracle_disparity_error
from reliable_endo_gs.uncertainty.extraction import (
    availability_table,
    extract_model_iterations,
    extract_proxy_outputs,
)


FINAL_SEQUENCES = frozenset({"dataset_5/keyframe_1", "dataset_8/keyframe_2"})
DEVELOPMENT_SEQUENCES = (
    "dataset_1/keyframe_1",
    "dataset_2/keyframe_1",
    "dataset_3/keyframe_1",
    "dataset_7/keyframe_2",
    "dataset_4/keyframe_4",
)
PINNED_UPSTREAM = "186fa2b4a2159b28393492f6df1aa444b54391a8"
PHASE1_PROTOCOL_SHA = "E4FCAF8A15F0DBA00294DE02AAA74ED12BEC095AA797D97022079C216D663032"
DEVELOPMENT_MANIFEST_SHA = "67C4B16BDAEC11B653A5AD757897F4699D37F8C74F4B320E084E5A5492665786"
DEVELOPMENT_SPLIT_SHA = "A03495E8C345C3396463D6E7F9A85F625699A3504804B71075929E67B2605590"
DEVELOPMENT_CONFIG_SHA = "933E9A2777B9E0080CC00ACAC3E476DA2B8EA2F1142CC036E671195E24940A28"
BASELINE_SHA = "32D42DE10A4C683C31FB4B3E0DE831F900E11197CDF1769B18CF3BBAFA5EE16F"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def git_output(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_start(device: torch.device) -> tuple[int, int] | None:
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(device)
    return int(torch.cuda.memory_allocated(device)), int(torch.cuda.memory_reserved(device))


def _memory_finish(
    device: torch.device, baseline_memory: tuple[int, int] | None
) -> dict[str, int | None]:
    if device.type != "cuda" or baseline_memory is None:
        return {"peak_allocated_delta_bytes": None, "peak_reserved_delta_bytes": None}
    baseline_allocated, baseline_reserved = baseline_memory
    return {
        "peak_allocated_delta_bytes": max(
            0, int(torch.cuda.max_memory_allocated(device)) - baseline_allocated
        ),
        "peak_reserved_delta_bytes": max(
            0, int(torch.cuda.max_memory_reserved(device)) - baseline_reserved
        ),
    }


def _scalar(value: object) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return value
    raise TypeError(f"expected scalar, got {type(value).__name__}")


def _sequence_from_name(value: object) -> str:
    parts = str(value).split("/")
    if len(parts) < 2:
        raise RuntimeError(f"cannot recover sequence identity from sample name {value!r}")
    return "/".join(parts[:2])


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for view in ("lmain", "rmain"):
        for key, value in batch[view].items():
            if isinstance(value, torch.Tensor):
                batch[view][key] = value.to(device, non_blocking=True)
    return batch


def _verify_no_final_access(
    config_path: Path,
    split_path: Path,
    manifest_path: Path,
    sequences: tuple[str, ...],
) -> None:
    if (
        config_path.name == "scared_phase1_final_v1.yaml"
        or split_path.name == "phase1_final_untouched_v1.json"
        or manifest_path.name == "scared_phase1_final_untouched_v1.json"
    ):
        raise RuntimeError("P1 development runner refuses the frozen final-evaluation config")
    forbidden = FINAL_SEQUENCES.intersection(sequences)
    if forbidden:
        raise RuntimeError(f"P1 development runner refuses final sequence(s): {sorted(forbidden)}")


def _verify_identities(
    repo: Path,
    config_path: Path,
    checkpoint_path: Path,
    data_root: Path,
    manifest_path: Path,
    split_path: Path,
    data_config_path: Path,
    raw: dict[str, Any],
) -> dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    for path in (manifest_path, split_path, data_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_sha = git_output("rev-parse", "HEAD", cwd=repo)
    expected_source = os.environ.get(str(raw["identity"].get("source_sha_env", "")), "")
    if not expected_source or source_sha != expected_source:
        raise RuntimeError(f"source identity mismatch: actual={source_sha}, expected={expected_source!r}")
    if git_output("status", "--short", "--untracked-files=all", cwd=repo):
        raise RuntimeError("ReliableEndo-GS source must be clean before P1 execution")
    upstream = repo / "third_party" / "endo_e2e_gs"
    upstream_sha = git_output("rev-parse", "HEAD", cwd=upstream)
    if upstream_sha != PINNED_UPSTREAM or git_output("status", "--short", cwd=upstream):
        raise RuntimeError("pinned third_party identity/status check failed")
    hashes = {
        "baseline_checkpoint_sha256": sha256_file(checkpoint_path),
        "development_manifest_sha256": sha256_file(manifest_path),
        "development_split_sha256": sha256_file(split_path),
        "development_config_sha256": sha256_file(data_config_path),
        "phase1_protocol_sha256": sha256_file(repo / "configs/research/phase1_revised.yaml"),
    }
    expected = {
        "baseline_checkpoint_sha256": BASELINE_SHA,
        "development_manifest_sha256": DEVELOPMENT_MANIFEST_SHA,
        "development_split_sha256": DEVELOPMENT_SPLIT_SHA,
        "development_config_sha256": DEVELOPMENT_CONFIG_SHA,
        "phase1_protocol_sha256": PHASE1_PROTOCOL_SHA,
    }
    if hashes != expected:
        raise RuntimeError(f"P1 frozen identity mismatch: actual={hashes}, expected={expected}")
    return {
        **hashes,
        "source_sha": source_sha,
        "upstream_sha": upstream_sha,
        "data_root": str(data_root),
        "checkpoint_step": int(raw["baseline"]["step"]),
    }


def _load_upstream_model(repo: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    from reliable_endo_gs.baseline.upstream import import_upstream_module

    upstream = repo / "third_party" / "endo_e2e_gs"
    config_module = import_upstream_module("config.stereo_config", upstream)
    loader = config_module.ConfigStereo()
    loader.load(str(upstream / "config/stage2.yaml"))
    cfg = loader.get_cfg()
    cfg.defrost()
    cfg.batch_size = 1
    cfg.raft.mixed_precision = False
    if int(cfg.raft.val_iters) != 3:
        raise RuntimeError(f"expected pinned val_iters=3, got {cfg.raft.val_iters}")
    cfg.freeze()
    network_module = import_upstream_module("lib.network", upstream)
    model = network_module.StereoEndoModel(cfg, with_gs_render=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "network" not in checkpoint:
        raise RuntimeError("baseline checkpoint lacks network state")
    if int(checkpoint.get("step", -1)) != 3000:
        raise RuntimeError(f"baseline checkpoint step mismatch: {checkpoint.get('step')!r}")
    result = model.load_state_dict(checkpoint["network"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict baseline network load returned unexpected key differences")
    return model.to(device).eval()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_p1(
    *,
    repo: Path,
    config_path: Path,
    output_path: Path,
    device: torch.device,
    max_samples: int | None = None,
    include_left_right: bool = True,
) -> dict[str, object]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("protocol_status") != "development_only_measurement":
        raise RuntimeError("P1 runner requires the development-only P1 config")
    data_cfg = raw["data"]
    data_root = Path(os.environ[str(data_cfg["root_env"])])
    manifest_path = Path(os.environ[str(data_cfg["manifest_env"])])
    split_path = repo / str(data_cfg["split"])
    data_config_path = repo / str(data_cfg["config"])
    checkpoint_path = Path(os.environ[str(raw["baseline"]["checkpoint_env"])])
    from reliable_endo_gs.data.scared_manifest import load_scared_split_manifest

    split = load_scared_split_manifest(split_path)
    sequences = tuple(split.train) + tuple(split.validation)
    _verify_no_final_access(config_path, split_path, manifest_path, sequences)
    if sequences != tuple(DEVELOPMENT_SEQUENCES):
        raise RuntimeError(f"development sequence contract mismatch: {sequences}")
    identities = _verify_identities(
        repo,
        config_path,
        checkpoint_path,
        data_root,
        manifest_path,
        split_path,
        data_config_path,
        raw,
    )
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite P1 output directory: {output_path}")

    from reliable_endo_gs.data.stage2 import ScaredStage2Dataset, stage2_collate_fn
    from torch.utils.data import ConcatDataset, DataLoader

    train_dataset = ScaredStage2Dataset(
        scared_root=data_root,
        split_manifest=split_path,
        split="train",
        phase=str(data_cfg["train_phase"]),
        test_every=int(data_cfg["test_every"]),
    )
    validation_dataset = ScaredStage2Dataset(
        scared_root=data_root,
        split_manifest=split_path,
        split="validation",
        phase=str(data_cfg["validation_phase"]),
        test_every=int(data_cfg["test_every"]),
    )
    if tuple(train_dataset.keyframe_entries) != tuple(data_cfg["train_sequences"]):
        raise RuntimeError("P1 training sequence contract mismatch")
    if tuple(validation_dataset.keyframe_entries) != tuple(data_cfg["development_holdouts"]):
        raise RuntimeError("P1 holdout sequence contract mismatch")
    dataset = ConcatDataset((train_dataset, validation_dataset))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=stage2_collate_fn)

    output_path.mkdir(parents=True)
    (output_path / "samples").mkdir()
    (output_path / "metrics").mkdir()
    _write_json(output_path / "protocol.json", raw)
    _write_json(output_path / "source_identity.json", identities)
    _write_json(output_path / "availability.json", {"proxies": list(availability_table(include_left_right=include_left_right))})

    model = _load_upstream_model(repo, checkpoint_path, device)
    rows: list[dict[str, object]] = []
    sequence_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    sample_count = 0
    with (output_path / "metrics/per_frame.jsonl").open("w", encoding="utf-8") as metrics_file:
        with torch.no_grad():
            for batch in loader:
                sequence = _sequence_from_name(batch["name"][0])
                batch = _move_batch(batch, device)
                left = batch["lmain"]["img"]
                right = batch["rmain"]["img"]
                _sync(device)
                left_memory_start = _memory_start(device)
                left_start = time.perf_counter()
                left_iterations = extract_model_iterations(model, left, right, iterations=3)
                _sync(device)
                left_seconds = time.perf_counter() - left_start
                left_memory = _memory_finish(device, left_memory_start)
                right_iterations = None
                right_seconds = 0.0
                right_memory = {"peak_allocated_delta_bytes": None, "peak_reserved_delta_bytes": None}
                if include_left_right:
                    _sync(device)
                    right_memory_start = _memory_start(device)
                    right_start = time.perf_counter()
                    right_iterations = extract_model_iterations(model, right, left, iterations=3)
                    _sync(device)
                    right_seconds = time.perf_counter() - right_start
                    right_memory = _memory_finish(device, right_memory_start)
                _sync(device)
                post_memory_start = _memory_start(device)
                post_start = time.perf_counter()
                outputs = extract_proxy_outputs(
                    left_iterations,
                    left,
                    right,
                    right_disparity_iterations=right_iterations,
                    iteration_window=3,
                )
                _sync(device)
                post_seconds = time.perf_counter() - post_start
                post_memory = _memory_finish(device, post_memory_start)
                error, target_mask = oracle_disparity_error(
                    outputs.disparity,
                    batch["lmain"]["disp"],
                    batch["lmain"]["mask"] >= 0.5,
                )
                sample_id = str(batch["name"][0])
                sample_dir = output_path / "samples" / sequence
                sample_dir.mkdir(parents=True, exist_ok=True)
                sample_payload: dict[str, np.ndarray] = {
                    "disparity": outputs.disparity[0].cpu().numpy(),
                    "disparity_iterations": outputs.disparity_iterations[0].cpu().numpy(),
                    "inference_valid_mask": outputs.inference_valid_mask[0].cpu().numpy(),
                    "ground_truth_disparity": batch["lmain"]["disp"][0].cpu().numpy(),
                    "ground_truth_valid_mask": target_mask[0].cpu().numpy(),
                    "absolute_disparity_error": error[0].cpu().numpy(),
                }
                for proxy_name, score in outputs.scores().items():
                    sample_payload[f"{proxy_name}"] = score.score[0].cpu().numpy()
                    sample_payload[f"{proxy_name}_valid_mask"] = score.valid_mask[0].cpu().numpy()
                np.savez_compressed(sample_dir / f"{sample_id.rsplit('/', 1)[-1]}.npz", **sample_payload)

                metric_row: dict[str, object] = {
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "left_forward_seconds": left_seconds,
                    "right_forward_seconds": right_seconds,
                    "proxy_postprocess_seconds": post_seconds,
                    "left_forward_vram": left_memory,
                    "right_forward_vram": right_memory,
                    "proxy_postprocess_vram": post_memory,
                    "target_valid_count": int(target_mask.sum().item()),
                    "metrics": {},
                }
                for proxy_name, score in outputs.scores().items():
                    summary = evaluate_p1_proxy(
                        proxy_name,
                        error,
                        score.score,
                        target_mask & score.valid_mask,
                        high_error_threshold=raw["evaluation"]["high_error_threshold"],
                        coverage_levels=tuple(raw["evaluation"]["coverage_levels"]),
                    )
                    metric_row["metrics"][proxy_name] = summary.to_dict()
                metrics_file.write(json.dumps(metric_row, sort_keys=True) + "\n")
                rows.append(metric_row)
                sequence_rows[sequence].append(metric_row)
                sample_count += 1
                if max_samples is not None and sample_count >= max_samples:
                    break

    def mean_metric(sequence_rows_: list[dict[str, object]], proxy_name: str, metric_name: str) -> float | None:
        values = [row["metrics"][proxy_name][metric_name] for row in sequence_rows_]
        numeric = [float(value) for value in values if value is not None]
        return sum(numeric) / len(numeric) if numeric else None

    per_sequence: dict[str, dict[str, object]] = {}
    for sequence, sequence_records in sequence_rows.items():
        available_proxy_names = tuple(
            str(proxy["proxy"])
            for proxy in availability_table(include_left_right=include_left_right)
            if proxy["available"]
            and any(str(proxy["proxy"]) in row["metrics"] for row in sequence_records)
        )
        per_sequence[sequence] = {
            "samples": len(sequence_records),
            "proxies": {
                proxy: {
                    "spearman": mean_metric(sequence_records, proxy, "spearman"),
                    "auroc": mean_metric(sequence_records, proxy, "auroc"),
                    "auprc": mean_metric(sequence_records, proxy, "auprc"),
                }
                for proxy in available_proxy_names
            },
        }
    macro: dict[str, object] = {"sequences": len(per_sequence), "proxies": {}}
    for proxy_name in {proxy for item in per_sequence.values() for proxy in item["proxies"]}:
        macro["proxies"][proxy_name] = {
            metric: _macro_optional(
                [item["proxies"][proxy_name].get(metric) for item in per_sequence.values() if proxy_name in item["proxies"]]
            )
            for metric in ("spearman", "auroc", "auprc")
        }
    _write_json(output_path / "metrics/per_sequence.json", per_sequence)
    _write_json(output_path / "metrics/macro.json", macro)
    return {"samples": sample_count, "sequences": list(sequence_rows), "output": str(output_path)}


def _macro_optional(values: list[object]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run development-only P1 uncertainty proxy measurement")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--disable-left-right", action="store_true")
    args = parser.parse_args(argv)
    result = run_p1(
        repo=args.repo.resolve(),
        config_path=args.config.resolve(),
        output_path=args.output.resolve(),
        device=torch.device(args.device),
        max_samples=args.max_samples,
        include_left_right=not args.disable_left_right,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
