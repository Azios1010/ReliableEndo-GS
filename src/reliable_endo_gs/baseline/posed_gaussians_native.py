"""Lazy, strict native model loading for posed-Gaussian comparison adapters."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch

from reliable_endo_gs.baseline.posed_gaussians import (
    PosedGaussianConfig,
    PosedGaussianPredictor,
)


def build_posed_predictor(
    kind: Literal["mvsplat", "depthsplat"],
    upstream: Path,
    upstream_commit: str,
    checkpoint: Path,
    *,
    near_m: float,
    far_m: float,
) -> tuple[PosedGaussianPredictor, dict]:
    """Strict-load official weights on CPU; no dataset access or GPU allocation.

    Run each kind in a separate process with the upstream repository as cwd
    and our src directory on PYTHONPATH. Both upstreams own a package named
    ``src``. Its location and Git identity are verified before constructing
    any model. DepthSplat's native constructor may populate TORCH_HOME with
    DINOv2; the launch environment must direct that cache to D:.
    """
    from dacite import from_dict
    from omegaconf import OmegaConf

    PosedGaussianConfig(near_m, far_m, 0)
    upstream = upstream.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if commit != upstream_commit or dirty:
        raise ValueError("upstream commit changed or tracked source is modified")
    package = importlib.import_module("src")
    if upstream / "src" not in [Path(p).resolve() for p in package.__path__]:
        raise ValueError("wrong upstream src package; launch from its repository")
    if kind not in ("mvsplat", "depthsplat"):
        raise ValueError("unknown posed baseline")
    name = "costvolume" if kind == "mvsplat" else "depthsplat"
    module = importlib.import_module("src.model.encoder.encoder_" + name)
    cfg_type = getattr(
        module, "EncoderCostVolumeCfg" if kind == "mvsplat" else "EncoderDepthSplatCfg"
    )
    encoder_type = getattr(
        module, "EncoderCostVolume" if kind == "mvsplat" else "EncoderDepthSplat"
    )
    base = OmegaConf.load(upstream / "config/model/encoder" / (name + ".yaml"))
    experiment = OmegaConf.load(upstream / "config/experiment/re10k.yaml")
    resolved = OmegaConf.merge(base, experiment.model.encoder)
    if kind == "depthsplat":
        resolved.supervise_intermediate_depth = False
    global_cfg = importlib.import_module("src.global_cfg")
    global_cfg.set_cfg(
        OmegaConf.create(
            {"mode": "test", "seed": 111123, "dataset": {"view_sampler": {"num_context_views": 2}}}
        )
    )
    encoder = encoder_type(from_dict(cfg_type, OmegaConf.to_container(resolved)))
    decoder_module = importlib.import_module("src.model.decoder.decoder_splatting_cuda")
    decoder = decoder_module.DecoderSplattingCUDA(
        decoder_module.DecoderSplattingCUDACfg(name="splatting_cuda"),
        SimpleNamespace(background_color=[0.0, 0.0, 0.0]),
    )
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = raw["state_dict"] if "state_dict" in raw else raw["model"]
    if not all(k.startswith(("encoder.", "decoder.", "losses.")) for k in state):
        raise ValueError("unexpected native checkpoint parameter namespace")
    for prefix, component in (("encoder.", encoder), ("decoder.", decoder)):
        selected = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
        expected = component.state_dict()
        if selected.keys() != expected.keys():
            raise ValueError("checkpoint parameter keys differ: " + prefix)
        for key, tensor in selected.items():
            if tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype:
                raise ValueError("checkpoint shape/dtype differs: " + prefix + key)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError("checkpoint has nonfinite weights: " + prefix + key)
        component.load_state_dict(selected, strict=True)
    step = int(raw.get("global_step", 300001))
    config = PosedGaussianConfig(near_m, far_m, step)
    predictor = PosedGaussianPredictor(encoder, decoder, config).eval().requires_grad_(False)
    return predictor, {
        "kind": kind,
        "upstream_commit": commit,
        "encoder_config": OmegaConf.to_container(resolved),
        "near_m": near_m,
        "far_m": far_m,
        "global_step": step,
        "global_step_source": "checkpoint" if "global_step" in raw else "official_final_schedule",
        "precision": "fp32; native renderer",
        "training": "released RE10K GS checkpoint; zero-shot; no SCARED fitting",
    }
