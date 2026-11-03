"""Pickle-global guard of the Fast-FoundationStereo adapter (synthetic names; no checkpoint, no upstream code is imported or executed)."""

from pathlib import Path

import pytest
import torch

from reliable_endo_gs.baseline.comparison import fast_foundation_stereo as ffs

# Representative of the globals that both research checkpoints (23-36-37 and 20-30-48) reference, as read from the pickle opcodes.
_OBSERVED = [
    "builtins.dict",
    "builtins.int",
    "builtins.list",
    "collections.defaultdict",
    "typing.Any",
    "numpy.dtype",
    "numpy.core.multiarray.scalar",
    "core.foundation_stereo.FastFoundationStereo",
    "core.distill_block.ForwardHelper",
    "timm.models.edgenext.EdgeNeXtStage",
    "timm.layers.norm.LayerNorm2d",
    "omegaconf.dictconfig.DictConfig",
    "omegaconf.base.ContainerMetadata",
    "torch.nn.modules.conv.Conv3d",
    "torch.nn.modules.batchnorm.SyncBatchNorm",
]


def _patch_globals(monkeypatch: pytest.MonkeyPatch, names: list[str]) -> None:
    monkeypatch.setattr(torch.serialization, "get_unsafe_globals_in_checkpoint", lambda _path: list(names))


def test_guard_accepts_the_globals_of_the_inspected_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_globals(monkeypatch, _OBSERVED)
    ffs._assert_allowed_globals(Path("unused.pth"))


@pytest.mark.parametrize(
    "hostile",
    [
        "builtins.eval",
        "builtins.exec",
        "builtins.getattr",
        "__builtin__.eval",
        "os.system",
        "posix.system",
        "subprocess.Popen",
        "torch.load",
        "torch.hub.load",
        "numpy.load",
        "omegaconf.omegaconf.OmegaConf",
        "coreutils.run",
    ],
)
def test_guard_refuses_anything_else_even_next_to_allowed_names(monkeypatch: pytest.MonkeyPatch, hostile: str) -> None:
    _patch_globals(monkeypatch, [*_OBSERVED, hostile])
    with pytest.raises(ffs.FastFoundationStereoError, match="unexpected globals") as raised:
        ffs._assert_allowed_globals(Path("unused.pth"))
    assert hostile in str(raised.value)


def test_checkpoint_hash_is_checked_before_the_pickle_is_inspected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(ffs, "_git_revision", lambda _root: "r" * 40)

    def unreachable(_path: Path) -> None:
        raise AssertionError("the pickle must not be inspected when the hash differs")

    monkeypatch.setattr(ffs, "_assert_allowed_globals", unreachable)
    with pytest.raises(ffs.FastFoundationStereoError, match="hash differs"):
        ffs.build_fast_foundation_stereo(tmp_path, "r" * 40, checkpoint, "0" * 64, torch.device("cpu"))


def test_upstream_revision_mismatch_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffs, "_git_revision", lambda _root: "a" * 40)
    with pytest.raises(ffs.FastFoundationStereoError, match="recorded revision"):
        ffs.build_fast_foundation_stereo(tmp_path, "b" * 40, tmp_path / "missing.pth", "0" * 64, torch.device("cpu"))
