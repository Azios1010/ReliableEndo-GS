"""Thin entry point: fine-tune MVSplat or DepthSplat on train_1236 (see baseline/finetune/runtime.py and docs/experiments_needed).

usage (from the upstream repository directory, GPU0 pinned by UUID in the environment):
  finetune_posed_baseline.py KIND OUT_DIR [--exposures N] [--memory-fraction F] [--snapshot-sha256 SHA]
Defaults: 39,450 exposures (3 passes over the 13,150 manifest ids, manifest order, no shuffle), AdamW 2e-5 (DepthSplat
pretrained monocular group 2e-7), weight decay 0.01, gradient clip 0.5, encoder autocast fp16 with fp32 geometry islands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path

import torch

from reliable_endo_gs.baseline.finetune.memory import apply_memory_policy
from reliable_endo_gs.baseline.finetune.projection import apply_principal_point_projection
from reliable_endo_gs.baseline.finetune.runtime import FinetuneConfig, replay_order, run_finetune
from reliable_endo_gs.baseline.finetune_feasibility import both_view_mse, posed_training_rgb
from reliable_endo_gs.baseline.posed_gaussians_native import build_posed_predictor
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, load_a01_manifest
from reliable_endo_gs.teachers import load_a01_roles

BASE = Path("/mnt/d/3DGS_runs/reliableendo-gs/baselines-20261031")
FEASIBILITY = BASE / "finetune-plan-20261031T154040Z"
DATA_ROOT = Path("/mnt/d/3DGS_datasets/preprocessing/SCARED_A00")
TRAIN_MANIFEST = Path(
    "/home/bkai/.local/share/reliableendo-gs/preparations/kd-train-1236-20261024T192708Z/train_1236.json"
)
GPU0 = "GPU-b9607686-60e0-2545-b328-ca3f0e79e24c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("mvsplat", "depthsplat"))
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--exposures", type=int, default=39450)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--snapshot-sha256", default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU0:
        raise SystemExit("REFUSED: this run is pinned to GPU0 by UUID")
    if not 0 < args.memory_fraction <= 0.90:
        raise SystemExit("REFUSED: memory fraction above the approved 0.90")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    config_json = json.loads(
        (BASE / "preparation-20261031T065247Z" / f"{args.kind}.json").read_text()
    )
    checkpoint = Path(config_json["checkpoint"])
    if _sha256(checkpoint) != config_json["checkpoint_sha256"]:
        raise SystemExit("REFUSED: official checkpoint hash mismatch")
    bounds_path = FEASIBILITY / "train-depth-range-metadata.json"
    bounds = json.loads(bounds_path.read_text())
    if bounds["status"] != "passed" or bounds["frames"] != 13150:
        raise SystemExit("REFUSED: train depth range is not the frozen one")
    predictor, identity = build_posed_predictor(
        args.kind,
        Path(config_json["upstream"]),
        config_json["upstream_commit"],
        checkpoint,
        near_m=bounds["near_m"],
        far_m=bounds["far_m"],
    )
    predictor.train().requires_grad_(True)
    policy = apply_memory_policy(args.kind, predictor)
    projection_patch = apply_principal_point_projection()
    roles = load_a01_roles(Path(__file__).resolve().parents[1] / "configs/teachers/a01_scared.yaml")
    manifest = load_a01_manifest(TRAIN_MANIFEST, roles=roles)
    manifest_ids = [entry.sample_id for entry in manifest.entries]
    if len(manifest_ids) != 13150:
        raise SystemExit("REFUSED: unexpected train manifest size")
    full = replay_order(manifest_ids, 3)
    ordered = full[: args.exposures]
    config = FinetuneConfig(
        kind=args.kind,
        exposures=args.exposures,
        seed=20261006,
        lr_main=2e-5,
        lr_pretrained=2e-7 if args.kind == "depthsplat" else None,
        weight_decay=0.01,
        grad_clip=0.5,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
    )
    local = threading.local()

    def load(sample_id: str):
        adapter = getattr(local, "adapter", None)
        if adapter is None:
            adapter = local.adapter = ScaredA00Adapter(DATA_ROOT, manifest, roles=roles)
        sample = adapter.load(
            sample_id,
            data_role="final_train",
            include_supervision=False,
            output_hw=(512, 640),
            device="cpu",
        )
        return sample.batch

    provenance = {
        "project_head": subprocess.check_output(
            ["git", "-C", "/mnt/d/workspace/huann/reliableendo-gs", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_snapshot_sha256": args.snapshot_sha256,
        "checkpoint_sha256": config_json["checkpoint_sha256"],
        "upstream_commit": config_json["upstream_commit"],
        "near_m": bounds["near_m"],
        "far_m": bounds["far_m"],
        "depth_range_sha256": _sha256(bounds_path),
        "train_manifest_sha256": _sha256(TRAIN_MANIFEST),
        "memory_policy": policy,
        "principal_point_projection": projection_patch,
        "native_identity": identity,
        "gpu_uuid": GPU0,
        "memory_fraction": args.memory_fraction,
        "torch": torch.__version__,
        "precision": "encoder autocast fp16 + GradScaler; fp32 geometry islands and rasterizer",
        "loss": "mean supported RGB MSE over left and right input cameras (deployment support)",
        "order": "manifest order replayed, no shuffle",
    }
    summary = run_finetune(
        predictor,
        config,
        ordered,
        load,
        posed_training_rgb,
        both_view_mse,
        args.out_dir,
        device=torch.device("cuda:0"),
        provenance=provenance,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, default=str))


if __name__ == "__main__":
    main()
