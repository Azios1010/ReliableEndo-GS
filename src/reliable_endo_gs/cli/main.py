"""Minimal command-line interface for infrastructure validation."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from reliable_endo_gs.config import ConfigError, config_to_dict, hash_config, load_config
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.runtime.logging import configure_logging
from reliable_endo_gs.runtime.manifest import (
    create_run_id,
    create_run_manifest,
    utc_timestamp,
    write_run_manifest,
)
from reliable_endo_gs.utils.io import create_unique_directory, write_yaml_atomic
from reliable_endo_gs.utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the ReliableEndo-GS command-line parser."""

    parser = argparse.ArgumentParser(prog="reg", description="ReliableEndo-GS utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser(
        "smoke", description="Validate the development infrastructure"
    )
    smoke_parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config")
    return parser


def run_smoke(config_path: Path) -> Path:
    """Execute the infrastructure-only smoke workflow and return its run directory."""

    config = load_config(config_path)
    logger = configure_logging()
    logger.info("Loaded configuration from %s", config_path)

    set_seed(config.experiment.seed)
    config_hash = hash_config(config)
    repo_root = Path.cwd()
    project_git_commit = get_git_commit(repo_root)
    git_dirty = is_git_dirty(repo_root)

    timestamp = utc_timestamp()
    base_run_id = create_run_id(config.experiment.name, config_hash, timestamp)
    run_directory = create_unique_directory(config.experiment.output_root, base_run_id)

    manifest = create_run_manifest(
        run_id=run_directory.name,
        timestamp_utc=timestamp,
        project_git_commit=project_git_commit,
        git_dirty=git_dirty,
        config_hash=config_hash,
        seed=config.experiment.seed,
    )
    write_yaml_atomic(run_directory / "resolved_config.yaml", config_to_dict(config))
    write_run_manifest(run_directory / "run_manifest.json", manifest)

    logger.info("Created infrastructure smoke run at %s", run_directory)
    return run_directory


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ReliableEndo-GS CLI."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "smoke":
        try:
            run_smoke(arguments.config)
        except (ConfigError, OSError, ValueError) as error:
            parser.exit(status=2, message=f"error: {error}\n")
        return 0

    parser.error(f"Unsupported command: {arguments.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
