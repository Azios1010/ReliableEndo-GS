"""Baseline config and explicit checkpoint identity tests."""

from pathlib import Path

import pytest
import torch

from reliable_endo_gs.baseline.checkpoint import (
    CheckpointError,
    load_upstream_checkpoint,
    sha256_file,
)
from reliable_endo_gs.baseline.config import BaselineConfigError, load_baseline_config
from reliable_endo_gs.baseline.upstream import PINNED_COMMIT


def test_committed_baseline_config_is_machine_portable_and_unbound() -> None:
    config = load_baseline_config(Path("configs/baseline/endo_e2e_gs.yaml"))

    assert config.upstream_commit == PINNED_COMMIT
    assert config.upstream_root == Path("third_party/endo_e2e_gs")
    assert not config.upstream_root.is_absolute()
    assert config.checkpoint_id is None
    assert config.checkpoint_path is None
    assert config.checkpoint_sha256 is None


def test_baseline_config_rejects_partial_checkpoint_identity(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    source = Path("configs/baseline/endo_e2e_gs.yaml").read_text(encoding="utf-8")
    path.write_text(
        source.replace("checkpoint_id: null", "checkpoint_id: official"), encoding="utf-8"
    )

    with pytest.raises(BaselineConfigError, match="must be set together"):
        load_baseline_config(path)


def test_checkpoint_loader_verifies_hash_and_official_network_key(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    path = tmp_path / "baseline.pth"
    torch.save({"network": source.state_dict()}, path)
    digest = sha256_file(path)

    result = load_upstream_checkpoint(
        target,
        path,
        expected_sha256=digest,
        map_location="cpu",
    )

    assert result.sha256 == digest
    assert result.missing_keys == ()
    assert result.unexpected_keys == ()
    for source_parameter, target_parameter in zip(
        source.parameters(), target.parameters(), strict=True
    ):
        assert torch.equal(source_parameter, target_parameter)


def test_checkpoint_loader_rejects_changed_or_invalid_files(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "baseline.pth"
    torch.save({"wrong": model.state_dict()}, path)
    digest = sha256_file(path)

    with pytest.raises(CheckpointError, match="SHA-256 mismatch"):
        load_upstream_checkpoint(
            model,
            path,
            expected_sha256="0" * 64,
            map_location="cpu",
        )
    with pytest.raises(CheckpointError, match="containing 'network'"):
        load_upstream_checkpoint(
            model,
            path,
            expected_sha256=digest,
            map_location="cpu",
        )


def test_checkpoint_loader_matches_upstream_torch_load_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "official.pth"
    path.write_bytes(b"checkpoint identity only")
    observed: dict[str, object] = {}

    def load_like_upstream(
        path_argument: Path, *, map_location: str, weights_only: bool
    ) -> dict[str, object]:
        observed["path"] = path_argument
        observed["map_location"] = map_location
        observed["weights_only"] = weights_only
        return {"network": {}}

    monkeypatch.setattr(torch, "load", load_like_upstream)
    result = load_upstream_checkpoint(
        torch.nn.Identity(),
        path,
        expected_sha256=sha256_file(path),
        map_location="cuda",
    )

    assert observed == {"path": path, "map_location": "cuda", "weights_only": False}
    assert result.missing_keys == ()
    assert result.unexpected_keys == ()
