"""Explicit larger selector cohorts and RGB-only operator ROI preparation."""

from __future__ import annotations

import hashlib
import io
import json
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from reliable_endo_gs.config.selector_refit import SelectorRefitConfig, partition_selector_training
from reliable_endo_gs.data.scared_a00 import load_a01_manifest
from reliable_endo_gs.teachers import load_a01_roles
from reliable_endo_gs.utils.io import write_json_atomic


def validate_selector_worklist(
    raw: dict[str, Any], train_ids: set[str], dev_ids: set[str], policy: SelectorRefitConfig
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Declared bounds, original manifests, all seven folds and no overlap."""
    if raw.get("schema") not in ("mask_validation_worklist.v1", "selector_cohort_worklist.v1"):
        raise ValueError("unsupported selector worklist schema")
    groups = []
    for name, known, maximum in (
        ("training", train_ids, policy.max_training_samples),
        ("development", dev_ids, policy.max_development_samples),
    ):
        ids = raw.get(name)
        if (
            not isinstance(ids, list)
            or not 7 <= len(ids) <= maximum
            or any(not isinstance(s, str) for s in ids)
            or len(ids) != len(set(ids))
            or not set(ids) <= known
            or {s.split("/")[0] for s in ids} != {f"dataset_{i}" for i in range(1, 8)}
        ):
            raise ValueError(f"invalid declared {name} cohort")
        groups.append(tuple(ids))
    if train_ids & set(groups[1]) or set(groups[0]) & set(groups[1]):
        raise ValueError("selector development overlaps training")
    return groups[0], groups[1]


def _external(path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    if path.resolve() == repo or repo in path.resolve().parents:
        raise ValueError("selector data assets must stay outside repository")


def _rgb_path(root: Path, sample_id: str) -> Path:
    parts = sample_id.split("/")
    if (
        len(parts) != 3
        or parts[0] not in {f"dataset_{i}" for i in range(1, 8)}
        or any(p in ("", ".", "..") for p in parts)
    ):
        raise ValueError("invalid original TRAIN sample identity")
    return root / parts[0] / parts[1] / "data/left_finalpass" / (parts[2] + ".png")


def prepare_selector_review(
    *,
    data_root: Path,
    training_manifest: Path,
    development_manifest: Path,
    roles_path: Path,
    policy: SelectorRefitConfig,
    previous_roi_policy: Path,
    output_dir: Path,
    review_frames: int = 16,
) -> dict[str, Any]:
    """Seal roles first; sampled RGB contact sheets, no GT/actions/teacher."""
    _external(output_dir)
    _external(data_root)
    if output_dir.exists() or not 8 <= review_frames <= 32:
        raise ValueError("review requires a new directory and 8..32 temporal frames")
    roles = load_a01_roles(roles_path)
    train, dev = (
        load_a01_manifest(p, roles=roles) for p in (training_manifest, development_manifest)
    )
    train_ids = [e.sample_id for e in train.entries if e.data_role == "final_train"]
    dev_ids = [e.sample_id for e in dev.entries if e.data_role == "final_train"]
    work = {
        "schema": "selector_cohort_worklist.v1",
        "training": train_ids,
        "development": dev_ids,
        "training_manifest_hash": train.manifest_hash,
        "development_manifest_hash": dev.manifest_hash,
        "role_hash": roles.role_hash,
        "selection": "reuse exact approved parent cohorts; no label selection",
    }
    validate_selector_worklist(work, set(train_ids), set(dev_ids), policy)
    partition = partition_selector_training(tuple(train_ids), policy)
    prior = json.loads(previous_roi_policy.read_text())
    hw = tuple(prior["output_hw"])
    if hw != (512, 640):
        raise ValueError("RGB review requires the declared checkpoint 512x640 grid")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "partition.json", partition)
    write_json_atomic(output_dir / "worklist-draft.json", work)
    pools: dict[str, list[str]] = defaultdict(list)
    for sid in train_ids + dev_ids:
        pools[sid.rsplit("/", 1)[0]].append(sid)
    reviews = {}
    bounds = {}
    for key, pool in sorted(pools.items()):
        ids = sorted(set(pool))
        n = min(review_frames, len(ids))
        chosen = [ids[i * (len(ids) - 1) // (n - 1)] for i in range(n)]
        box = prior.get("bounds_yxyx_by_keyframe", {}).get(key, [144, 192, 368, 432])
        bounds[key] = box
        canvas = Image.new("RGB", (960, 40 + ((n + 3) // 4) * 216), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 5), key + " RGB only; suggested ROI; temporal sample review", fill="black")
        hashes = {}
        for i, sid in enumerate(chosen):
            content = _rgb_path(data_root, sid).read_bytes()
            hashes[sid] = hashlib.sha256(content).hexdigest()
            with Image.open(io.BytesIO(content)) as image:
                thumb = image.convert("RGB").resize((240, 192), Image.Resampling.BILINEAR)
            y0, x0, y1, x1 = box
            ImageDraw.Draw(thumb).rectangle(
                (x0 * 240 / 640, y0 * 192 / 512, x1 * 240 / 640, y1 * 192 / 512),
                outline="lime",
                width=2,
            )
            x = (i % 4) * 240
            y = 40 + (i // 4) * 216
            canvas.paste(thumb, (x, y))
            draw.text((x + 3, y + 194), sid.split("/")[-1], fill="black")
        name = key.replace("/", "-") + ".png"
        canvas.save(output_dir / name)
        reviews[key] = {
            "sample_ids": chosen,
            "rgb_sha256": hashes,
            "sheet": name,
            "sheet_sha256": hashlib.sha256((output_dir / name).read_bytes()).hexdigest(),
        }
        print(
            json.dumps(
                {"event": "selector_roi_review_prepared", "keyframe": key, "review_frames": n}
            ),
            flush=True,
        )
    report = {
        "schema": "selector_rgb_roi_review.v1",
        "output_hw": list(hw),
        "bounds_yxyx_by_keyframe": bounds,
        "reviews": reviews,
        "reviewed_keyframes": [],
        "previous_roi_policy_sha256": hashlib.sha256(previous_roi_policy.read_bytes()).hexdigest(),
        "parent_roles_changed": False,
        "ground_truth_used": False,
        "action_outcomes_used": False,
        "review_scope": "sampled RGB frames across each complete cohort timeline; not per-frame segmentation",
        "limitations": [
            "Coarse operator tissue ROI; unreviewed intermediate frames may contain tools or specular tissue.",
            "No automatic tissue segmentation model is validated.",
        ],
    }
    write_json_atomic(output_dir / "roi-review.json", report)
    return {
        "status": "rgb_roi_review_pending",
        "training_samples": len(train_ids),
        "development_samples": len(dev_ids),
        "keyframes": len(reviews),
        "output_dir": str(output_dir),
        "ground_truth_loaded": False,
    }


def finalize_selector_rois(
    *, data_root: Path, output_dir: Path, resume: bool = False, workers: int = 2
) -> dict[str, Any]:
    """Require explicit RGB review; share bool HxW masks for repeated anchor boxes."""
    _external(output_dir)
    _external(data_root)
    policy = json.loads((output_dir / "roi-review.json").read_text())
    keys = set(policy["bounds_yxyx_by_keyframe"])
    if (
        policy.get("schema") != "selector_rgb_roi_review.v1"
        or set(policy.get("reviewed_keyframes", [])) != keys
        or policy.get("ground_truth_used") is not False
        or policy.get("action_outcomes_used") is not False
    ):
        raise ValueError("all keyframes require recorded RGB-only visual review")
    for row in policy["reviews"].values():
        if (
            hashlib.sha256((output_dir / row["sheet"]).read_bytes()).hexdigest()
            != row["sheet_sha256"]
        ):
            raise ValueError("RGB review sheet changed")
        for sid, digest in row["rgb_sha256"].items():
            if hashlib.sha256(_rgb_path(data_root, sid).read_bytes()).hexdigest() != digest:
                raise ValueError("reviewed RGB source changed")
    work = json.loads((output_dir / "worklist-draft.json").read_text())
    hw = policy["output_hw"]
    phash = hashlib.sha256((output_dir / "roi-review.json").read_bytes()).hexdigest()
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("ROI preparation permits one to four I/O workers")
    seal = {
        "schema": "selector_roi_finalization.v1",
        "policy_sha256": phash,
        "worklist_sha256": hashlib.sha256(
            (output_dir / "worklist-draft.json").read_bytes()
        ).hexdigest(),
        "data_root": str(data_root.resolve()),
        "producer_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    seal_path = output_dir / "finalization.json"
    if seal_path.exists():
        if not resume or json.loads(seal_path.read_text()) != seal:
            raise ValueError("ROI resume policy/cohort/source mismatch")
    else:
        write_json_atomic(seal_path, seal)
    mask_root = output_dir / "masks"
    if mask_root.exists() and not resume:
        raise ValueError("partial ROI assets require explicit resume")
    mask_root.mkdir(exist_ok=True)
    files: dict[tuple[str, tuple[int, ...]], tuple[str, str]] = {}
    ids = work["training"] + work["development"]
    boxes = {sid: reviewed_roi_box(policy, sid) for sid in ids}
    for key, box in sorted({(sid.rsplit("/", 1)[0], tuple(box)) for sid, box in boxes.items()}):
        if len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("invalid reviewed ROI bounds")
        y0, x0, y1, x1 = box
        if not (0 <= y0 < y1 <= hw[0] and 0 <= x0 < x1 <= hw[1]):
            raise ValueError("ROI outside declared output grid")
        mask = np.zeros(tuple(hw), dtype=np.bool_)
        mask[y0:y1, x0:x1] = True
        path = mask_root / (
            key.replace("/", "-")
            + "-"
            + hashlib.sha256(str(box).encode()).hexdigest()[:12]
            + ".npy"
        )
        if path.exists():
            existing = np.load(path, allow_pickle=False)
            if existing.dtype != np.bool_ or not np.array_equal(existing, mask):
                raise ValueError("existing ROI mask differs from approved policy")
        else:
            np.save(path, mask, allow_pickle=False)
        files[(key, box)] = (
            path.relative_to(output_dir).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    entries = {}
    sources = {}
    ledger = output_dir / "roi-members.jsonl"
    if ledger.exists():
        if not resume:
            raise ValueError("ROI member ledger requires resume")
        lines = ledger.read_text().splitlines(keepends=True)
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise ValueError("corrupt completed preparation ledger row") from None
                archive = output_dir / (
                    "roi-members-interrupted-" + uuid.uuid4().hex[:12] + ".jsonl"
                )
                ledger.rename(archive)
                ledger.write_text("".join(lines[:i]))
                print(
                    json.dumps({"event": "roi_ledger_partial_tail_archived", "path": str(archive)}),
                    flush=True,
                )
                break
            if row["sample_id"] in entries or row["sample_id"] not in ids:
                raise ValueError("invalid preparation ledger sample identity")
            for name, metadata in row["sources"].items():
                stat = (data_root / name).stat()
                if stat.st_size != metadata["size"] or stat.st_mtime_ns != metadata["mtime_ns"]:
                    raise ValueError("preparation source member changed")
            entries[row["sample_id"]] = row["entry"]
            sources.update(row["sources"])

    def member(sid: str) -> dict[str, Any]:
        rgb = _rgb_path(data_root, sid)
        digest = hashlib.sha256(rgb.read_bytes()).hexdigest()
        with Image.open(rgb) as image:
            if (image.height, image.width) != (1024, 1280):
                raise ValueError("processed RGB input grid differs from reviewed preprocessing")
        path, mask_hash = files[(sid.rsplit("/", 1)[0], tuple(boxes[sid]))]
        entry = {
            "path": path,
            "sha256": mask_hash,
            "output_hw": hw,
            "source": "input_roi",
            "source_identity": f"RGB_temporal_operator_ROI;policy_sha256={phash};rgb_sha256={digest}",
        }
        base = rgb.parent.parent
        members = [
            rgb,
            base / "right_finalpass" / rgb.name,
            base / "frame_data" / (sid.split("/")[-1] + ".json"),
            base / "newpram_data" / (sid.split("/")[-1] + ".json"),
            base / "disparity" / (sid.split("/")[-1] + ".tiff"),
            base / "a00_protocol.json",
        ]
        member_sources = {}
        for source_member in members:
            relative = source_member.relative_to(data_root).as_posix()
            stat = source_member.stat()
            h = hashlib.sha256()
            with source_member.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            if source_member.stat().st_mtime_ns != stat.st_mtime_ns:
                raise ValueError("source changed during hashing")
            member_sources[relative] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": h.hexdigest(),
            }
        return {"sample_id": sid, "entry": entry, "sources": member_sources}

    pending = [sid for sid in ids if sid not in entries]
    with ThreadPoolExecutor(max_workers=workers) as executor, ledger.open("a") as stream:
        for row in executor.map(member, pending):
            sid = row["sample_id"]
            entries[sid] = row["entry"]
            sources.update(row["sources"])
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if len(entries) == 1 or len(entries) % 250 == 0:
                print(
                    json.dumps(
                        {
                            "event": "selector_roi_index_progress",
                            "samples": len(entries),
                            "total": len(ids),
                            "io_workers": workers,
                        }
                    ),
                    flush=True,
                )
    entries = {sid: entries[sid] for sid in ids}
    write_json_atomic(
        output_dir / "target-mask-index.json",
        {"schema": "deployment_target_masks.v1", "samples": entries},
    )
    write_json_atomic(
        output_dir / "data-source-index.json",
        {"schema": "selector_data_source_snapshot.v1", "files": sources, "gt_decoded": False},
    )
    work["data_source_index_sha256"] = hashlib.sha256(
        (output_dir / "data-source-index.json").read_bytes()
    ).hexdigest()
    work["target_mask_index_sha256"] = hashlib.sha256(
        (output_dir / "target-mask-index.json").read_bytes()
    ).hexdigest()
    work["roi_policy_sha256"] = phash
    work["roi_review_scope"] = policy["review_scope"]
    write_json_atomic(output_dir / "worklist.json", work)
    report = {
        "status": "selector_cohort_rois_prepared",
        "training_samples": len(work["training"]),
        "development_samples": len(work["development"]),
        "mask_files": len(files),
        "target_mask_index_sha256": work["target_mask_index_sha256"],
        "roi_policy_sha256": phash,
        "ground_truth_loaded": False,
        "teacher_inference_executed": False,
        "training_executed": False,
    }
    write_json_atomic(output_dir / "preparation.json", report)
    return report


def reviewed_roi_box(policy: dict[str, Any], sample_id: str) -> list[int]:
    """Nearest reviewed temporal anchor; no image/GT/outcome-dependent reassignment."""
    key = sample_id.rsplit("/", 1)[0]
    overrides = policy.get("bounds_by_review_sample", {})
    anchors = policy["reviews"][key]["sample_ids"]
    frame = int(sample_id.split("/")[-1].removeprefix("frame_data"))
    nearest = min(
        anchors,
        key=lambda sid: (abs(int(sid.split("/")[-1].removeprefix("frame_data")) - frame), sid),
    )
    box = overrides.get(nearest, policy["bounds_yxyx_by_keyframe"][key])
    if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("invalid reviewed temporal ROI box")
    return [int(v) for v in box]


def render_selector_review(data_root: Path, output_dir: Path, keys: tuple[str, ...]) -> None:
    """Render updated ROI anchors for explicit RGB-only review."""
    policy = json.loads((output_dir / "roi-review.json").read_text())
    for key in keys:
        row = policy["reviews"][key]
        chosen = row["sample_ids"]
        n = len(chosen)
        canvas = Image.new("RGB", (960, 40 + ((n + 3) // 4) * 216), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 5), key + " RGB only; revised temporal ROI anchors", fill="black")
        for i, sid in enumerate(chosen):
            content = _rgb_path(data_root, sid).read_bytes()
            if hashlib.sha256(content).hexdigest() != row["rgb_sha256"][sid]:
                raise ValueError("RGB review source changed")
            with Image.open(io.BytesIO(content)) as im:
                thumb = im.convert("RGB").resize((240, 192), Image.Resampling.BILINEAR)
            y0, x0, y1, x1 = reviewed_roi_box(policy, sid)
            ImageDraw.Draw(thumb).rectangle(
                (x0 * 240 / 640, y0 * 192 / 512, x1 * 240 / 640, y1 * 192 / 512),
                outline="lime",
                width=2,
            )
            x = i % 4 * 240
            y = 40 + i // 4 * 216
            canvas.paste(thumb, (x, y))
            draw.text((x + 3, y + 194), sid.split("/")[-1], fill="black")
        canvas.save(output_dir / row["sheet"])
        row["sheet_sha256"] = hashlib.sha256((output_dir / row["sheet"]).read_bytes()).hexdigest()
    policy["reviewed_keyframes"] = [k for k in policy["reviewed_keyframes"] if k not in keys]
    write_json_atomic(output_dir / "roi-review.json", policy)


def verify_selector_source_snapshot(data_root: Path, index: Path, expected_sha256: str) -> None:
    """Content hashes are sealed at preparation; reject any changed file stat on resume."""
    content = index.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("source snapshot identity mismatch")
    raw = json.loads(content)
    if raw.get("schema") != "selector_data_source_snapshot.v1":
        raise ValueError("unsupported source snapshot")
    resolved_root = data_root.resolve()

    def verify_member(item: tuple[str, dict[str, Any]]) -> None:
        name, row = item
        path = (resolved_root / name).resolve()
        if resolved_root not in path.parents:
            raise ValueError("source path escapes external data root")
        stat = path.stat()
        if stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise ValueError("source member changed; prepare new verified cohort: " + name)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for verified_count, _ in enumerate(executor.map(verify_member, raw["files"].items()), 1):
            if verified_count % 5000 == 0:
                print(
                    json.dumps(
                        {
                            "event": "selector_source_snapshot_progress",
                            "files": verified_count,
                            "total": len(raw["files"]),
                            "io_workers": 4,
                        }
                    ),
                    flush=True,
                )
    if data_root.resolve() != resolved_root:
        raise ValueError("external data root changed during source verification")
