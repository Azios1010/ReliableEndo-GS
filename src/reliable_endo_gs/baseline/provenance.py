"""Serializable identity of the pinned, unmodified baseline integration."""

from dataclasses import dataclass, field


def _validate_sha(value: str, length: int, name: str) -> None:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase {length}-character hexadecimal digest")


@dataclass(frozen=True, slots=True)
class BaselineProvenance:
    """Minimum provenance required before an output can claim baseline identity."""

    upstream_project: str
    upstream_repository: str
    upstream_commit: str
    integration_strategy: str
    patched: bool
    upstream_dirty: bool | None
    reliable_endo_gs_commit: str | None
    patch_ids: tuple[str, ...] = field(default_factory=tuple)
    checkpoint_id: str | None = None
    checkpoint_sha256: str | None = None

    SCHEMA_VERSION = "1.0"

    def __post_init__(self) -> None:
        for name, value in (
            ("upstream_project", self.upstream_project),
            ("upstream_repository", self.upstream_repository),
            ("integration_strategy", self.integration_strategy),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        _validate_sha(self.upstream_commit, 40, "upstream_commit")
        if self.reliable_endo_gs_commit is not None:
            _validate_sha(self.reliable_endo_gs_commit, 40, "reliable_endo_gs_commit")
        if self.checkpoint_sha256 is not None:
            _validate_sha(self.checkpoint_sha256, 64, "checkpoint_sha256")
        if (self.checkpoint_id is None) != (self.checkpoint_sha256 is None):
            raise ValueError("checkpoint_id and checkpoint_sha256 must be present together")
        if self.patched != bool(self.patch_ids):
            raise ValueError("patched must agree with whether patch_ids are present")

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON/YAML-compatible representation."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "upstream_project": self.upstream_project,
            "upstream_repository": self.upstream_repository,
            "upstream_commit": self.upstream_commit,
            "integration_strategy": self.integration_strategy,
            "patched": self.patched,
            "patch_ids": list(self.patch_ids),
            "upstream_dirty": self.upstream_dirty,
            "reliable_endo_gs_commit": self.reliable_endo_gs_commit,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_sha256": self.checkpoint_sha256,
        }
