"""Package the bounded Stage2 patch, including untracked modules, for transfer."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from reliable_endo_gs.baseline.upstream import PINNED_COMMIT

PATCH_FILES = (
    "config/stage2.yaml",
    "config/stereo_config.py",
    "gaussian_renderer/__init__.py",
    "lib/GaussianRender.py",
    "lib/gs_parm_network.py",
    "lib/network.py",
    "lib/scale_parameterization.py",
    "lib/stage2_diagnostics.py",
    "lib/stage2_run.py",
    "train_stage2.py",
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout


def build_stage2_bundle(repo_root: Path, output: Path) -> dict[str, object]:
    """Create a new ZIP containing a git-applicable patch and byte-identifiable sources.

    The bundle contains code and a runbook only: no data, checkpoints or outputs.
    Existing artifacts are never overwritten. Text hashes normalize newlines.
    """
    root = repo_root / "third_party" / "endo_e2e_gs"
    revision = _git(root, "rev-parse", "HEAD").strip()
    if revision != PINNED_COMMIT:
        raise ValueError(f"Unexpected upstream base {revision}; expected {PINNED_COMMIT}")
    changed = set(_git(root, "diff", "--name-only", "HEAD").splitlines())
    if changed - set(PATCH_FILES):
        raise ValueError(f"Out-of-scope upstream edits: {sorted(changed - set(PATCH_FILES))}")
    tracked = set(_git(root, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
    patch_parts: list[str] = []
    hashes: dict[str, str] = {}
    for relative in PATCH_FILES:
        current = (root / relative).read_text(encoding="utf-8")
        hashes[relative] = hashlib.sha256(current.encode("utf-8")).hexdigest()
        original = _git(root, "show", f"HEAD:{relative}") if relative in tracked else ""
        if original == current:
            continue
        patch_parts.append(f"diff --git a/{relative} b/{relative}\n")
        if relative not in tracked:
            patch_parts.append("new file mode 100644\n")
        for line in difflib.unified_diff(
            original.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile=f"a/{relative}" if relative in tracked else "/dev/null",
            tofile=f"b/{relative}",
        ):
            patch_parts.append(
                line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
            )
    patch = "".join(patch_parts).encode("utf-8")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "upstream_base_commit": revision,
        "project_commit": _git(repo_root, "rev-parse", "HEAD").strip(),
        "project_dirty": bool(_git(repo_root, "status", "--porcelain")),
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "target_source_sha256": hashes,
        "source_hash_encoding": "UTF-8, universal-newline normalization",
        "scientific_status": "local_checks_only; server_data_and_cuda_not_validated",
    }
    runbook = repo_root / "docs" / "stage2_fixed10k_server.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("stage2.patch", patch)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        archive.writestr("README.md", runbook.read_text(encoding="utf-8"))
    return manifest


def main() -> None:
    """Build the transfer artifact without changing either Git index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_stage2_bundle(args.repo_root.resolve(), args.output.resolve())
    print(json.dumps({"bundle": str(args.output.resolve()), **manifest}, indent=2))


if __name__ == "__main__":
    main()
