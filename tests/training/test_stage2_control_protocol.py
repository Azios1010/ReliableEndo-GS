from __future__ import annotations

from pathlib import Path

from reliable_endo_gs.training.stage2_control import Protocol, scheduler_probe


def test_final_stage2_scheduler_contract() -> None:
    protocol = Protocol(
        repo=Path("."),
        upstream=Path("."),
        config_path=Path("config.yaml"),
        run_dir=Path("run"),
        data_root=Path("data"),
        manifest_path=Path("manifest"),
        split_path=Path("split"),
        stage1_path=Path("stage1"),
        source_sha="source",
        upstream_sha="upstream",
        manifest_sha="manifest",
        split_sha="split",
        data_config_sha="config",
        stage1_sha="stage1",
    )
    info = scheduler_probe(protocol)
    assert info["class"] == "OneCycleLR"
    assert info["total_steps"] == 10100
    assert info["pct_start"] == 0.01
    assert info["anneal_strategy"] == "linear"
    assert info["peak_lr"] == 2e-4
    assert info["final_sampled_lr"] >= 0
    assert info["negative_lr_samples"] == 0
