"""End-to-end smoke test for the infrastructure CLI."""

import json
from pathlib import Path

import yaml

from reliable_endo_gs.cli.main import main
from reliable_endo_gs.config.hashing import hash_config
from reliable_endo_gs.config.loader import load_config


def test_cli_smoke_creates_resolved_config_and_manifest(
    tmp_path: Path,
    monkeypatch: object,
    valid_config_data: dict[str, object],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(valid_config_data, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]

    exit_code = main(["smoke", "--config", str(config_path)])

    assert exit_code == 0
    run_directories = list((tmp_path / "outputs").iterdir())
    assert len(run_directories) == 1
    run_directory = run_directories[0]
    resolved_path = run_directory / "resolved_config.yaml"
    manifest_path = run_directory / "run_manifest.json"
    assert resolved_path.is_file()
    assert manifest_path.is_file()

    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert resolved["experiment"]["output_root"] == "outputs"
    assert manifest["config_hash"] == hash_config(load_config(config_path))
    assert manifest["run_id"] == run_directory.name
