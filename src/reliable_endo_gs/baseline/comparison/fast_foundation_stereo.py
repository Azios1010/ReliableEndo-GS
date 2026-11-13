"""Fast-FoundationStereo (NVlabs, CVPR 2026) behind the comparison interface: zero-shot disparity from the calibrated rectified pair, geometry only.

The research checkpoints are pickled model objects, so loading is guarded: the checkpoint hash must equal the recorded one, the upstream directory must
be the checked-out revision, and every global the pickle would import must be an allowed exact name (``builtins.dict``/``int``/``list``, ``defaultdict``,
``numpy.dtype``, ...) or belong to an allowed module (``torch.nn.modules``, timm, omegaconf, the upstream ``core`` package); any other name,
for example ``os.system``, ``subprocess.Popen``, ``torch.load`` or ``builtins.eval``, refuses the load. The names are matched as strings and
attribute chains are not resolved, so the recorded checkpoint hash, not this check, is what rules out a modified file. The model
is used as released: no fine-tuning, ``valid_iters`` and ``max_disp`` as recorded. Conventions follow the FoundationStereo teacher adapter: A00 RGB in
[0, 1] is scaled to the native [0, 255], padded to a multiple of 32, run under autocast, unpadded; the disparity has the sign ``u_left - u_right`` in
input pixels and is returned without clipping.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.evaluation.multi_model import Prediction

# Names the research checkpoints import, as listed by ``torch.serialization.get_unsafe_globals_in_checkpoint`` (it reads the pickle opcodes and
# executes nothing; torch tensors and storages are already outside that list). Plain containers and scalar types are matched by exact name:
# a ``builtins.`` prefix would also admit ``builtins.eval``. The checkpoint hash is the primary guard; this check refuses a pickle that
# imports more than the inspected one. It matches the joined ``module.name`` string and does not resolve attribute chains.
_ALLOWED_GLOBAL_NAMES = frozenset(
    {
        "builtins.dict",
        "builtins.int",
        "builtins.list",
        "collections.defaultdict",
        "typing.Any",
        "numpy.dtype",
        "numpy.core.multiarray.scalar",
    }
)
_ALLOWED_GLOBAL_PREFIXES = (
    "torch.nn.modules.",
    "timm.layers.",
    "timm.models.",
    "omegaconf.base.",
    "omegaconf.dictconfig.",
    "omegaconf.listconfig.",
    "omegaconf.nodes.",
    "core.",
)


class FastFoundationStereoError(RuntimeError):
    """A recorded artefact differs from the one that was frozen, or the checkpoint imports something unexpected."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root.resolve().as_posix()}", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@contextmanager
def _upstream_on_path(root: Path) -> Iterator[None]:
    """Scoped import path for the upstream ``core`` package (one process loads one upstream)."""
    entry = str(root.resolve())
    sys.path.insert(0, entry)
    try:
        yield
    finally:
        sys.path.remove(entry)


def _assert_allowed_globals(checkpoint: Path) -> None:
    names = torch.serialization.get_unsafe_globals_in_checkpoint(str(checkpoint))
    rejected = sorted(
        name
        for name in names
        if name not in _ALLOWED_GLOBAL_NAMES and not name.startswith(_ALLOWED_GLOBAL_PREFIXES)
    )
    if rejected:
        raise FastFoundationStereoError(
            f"the checkpoint would import unexpected globals: {rejected}"
        )


class FastFoundationStereoPredictor(nn.Module):
    """Geometry-only predictor: ``forward(sample)`` returns the disparity [1,1,H,W] of the left image."""

    def __init__(self, model: nn.Module, padder_class: type, valid_iters: int) -> None:
        super().__init__()
        self.model, self.padder_class, self.valid_iters = model, padder_class, valid_iters

    def forward(self, sample: ScaredA00Sample) -> Prediction:
        left, right = sample.batch.left.float(), sample.batch.right.float()
        padder = self.padder_class(left.shape, divis_by=32)
        padded_left, padded_right = padder.pad(left.contiguous() * 255, right.contiguous() * 255)
        with torch.autocast(device_type=left.device.type, dtype=torch.float16):
            disparity = self.model.forward(
                padded_left,
                padded_right,
                iters=self.valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )
        disparity = padder.unpad(disparity.float())
        if tuple(disparity.shape) != (left.shape[0], 1, *left.shape[-2:]):
            raise FastFoundationStereoError("unpadding changed the A00 output grid")
        return Prediction(disparity)


def build_fast_foundation_stereo(
    upstream: Path,
    revision: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    device: torch.device,
    *,
    valid_iters: int = 8,
    max_disp: int = 192,
) -> FastFoundationStereoPredictor:
    """Load a frozen research checkpoint of the pinned upstream revision (see the module docstring for the guards)."""
    if _git_revision(upstream) != revision:
        raise FastFoundationStereoError("the upstream checkout is not the recorded revision")
    if _sha256(checkpoint) != checkpoint_sha256:
        raise FastFoundationStereoError("the checkpoint hash differs from the recorded one")
    _assert_allowed_globals(checkpoint)
    with _upstream_on_path(upstream):
        padder_class = importlib.import_module("core.utils.utils").InputPadder
        model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model = model.to(device).eval().requires_grad_(False)
    return FastFoundationStereoPredictor(model, padder_class, valid_iters).to(device).eval()
