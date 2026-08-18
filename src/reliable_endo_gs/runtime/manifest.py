"""Immutable, infrastructure-only run manifests."""

import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from reliable_endo_gs.utils.io import write_json_atomic

_NON_SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class RunManifest:
    """Minimal provenance captured for an infrastructure run."""

    run_id: str
    timestamp_utc: str
    project_git_commit: str | None
    git_dirty: bool | None
    config_hash: str
    seed: int
    python_version: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of the manifest."""

        return {
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp_utc,
            "project_git_commit": self.project_git_commit,
            "git_dirty": self.git_dirty,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "python_version": self.python_version,
        }


def utc_timestamp(now: datetime | None = None) -> str:
    """Return an ISO-8601 timestamp in UTC with second precision."""

    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise ValueError("UTC timestamp input must be timezone-aware")
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sanitize_experiment_name(name: str) -> str:
    """Convert an experiment name into a conservative path component."""

    sanitized = _NON_SLUG.sub("-", name.strip().lower()).strip("-")
    return sanitized or "run"


def create_run_id(experiment_name: str, config_hash: str, timestamp_utc: str) -> str:
    """Create a readable run identifier from time, experiment, and config."""

    parsed_time = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    compact_time = parsed_time.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{compact_time}_{sanitize_experiment_name(experiment_name)}_{config_hash[:8]}"


def create_run_manifest(
    *,
    run_id: str,
    project_git_commit: str | None,
    git_dirty: bool | None,
    config_hash: str,
    seed: int,
    timestamp_utc: str | None = None,
    python_version: str | None = None,
) -> RunManifest:
    """Create an immutable manifest without scientific metadata."""

    return RunManifest(
        run_id=run_id,
        timestamp_utc=timestamp_utc if timestamp_utc is not None else utc_timestamp(),
        project_git_commit=project_git_commit,
        git_dirty=git_dirty,
        config_hash=config_hash,
        seed=seed,
        python_version=python_version if python_version is not None else platform.python_version(),
    )


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    """Write a run manifest atomically as JSON."""

    write_json_atomic(path, manifest.to_dict())
