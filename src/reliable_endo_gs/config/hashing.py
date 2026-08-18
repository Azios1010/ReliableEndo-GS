"""Stable hashing for resolved project configurations."""

import hashlib
import json

from reliable_endo_gs.config.schema import ProjectConfig, config_to_dict


def canonical_config_json(config: ProjectConfig) -> str:
    """Serialize a resolved config canonically for comparison and hashing."""

    return json.dumps(
        config_to_dict(config),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def hash_config(config: ProjectConfig) -> str:
    """Return a stable SHA-256 digest for a resolved configuration."""

    canonical = canonical_config_json(config).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
