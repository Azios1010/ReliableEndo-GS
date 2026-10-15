"""Diagnostic provenance hashes include behavior sources and resolved config inputs."""

from tests.diagnostics.synthetic_risk_probe import _source_hash


def test_probe_identity_tracks_sources_fixture_and_config_but_ignores_generated_files(tmp_path):
    for directory in ("src", "tests", "configs"):
        (tmp_path / directory).mkdir()
    source = tmp_path / "src" / "model.py"
    fixture = tmp_path / "tests" / "fixture.py"
    config = tmp_path / "configs" / "probe.yaml"
    source.write_text("width = 8\n", encoding="utf-8")
    fixture.write_text("seed = 3\n", encoding="utf-8")
    config.write_text("feature_clip: 5\n", encoding="utf-8")
    original = _source_hash(tmp_path)
    for path in (source, fixture, config):
        before = path.read_text(encoding="utf-8")
        path.write_text(before + "# changed\n", encoding="utf-8")
        assert _source_hash(tmp_path) != original
        path.write_text(before, encoding="utf-8")
        assert _source_hash(tmp_path) == original
    (tmp_path / "tests" / "generated.json").write_text("{}", encoding="utf-8")
    assert _source_hash(tmp_path) == original
