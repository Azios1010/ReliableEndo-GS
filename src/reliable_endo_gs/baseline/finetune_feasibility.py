"""Disposable baseline probes and train-only depth statistics, not a trainer.

Native imports stay in this integration layer. No function reads validation or
test data, and no import initializes CUDA or accesses a dataset.
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from reliable_endo_gs.baseline.posed_gaussians import PosedGaussianPredictor, posed_context
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera
from reliable_endo_gs.data.scared_a00 import (
    ScaredA00Adapter,
    ScaredA00Sample,
    _extract_relative_transform,
    _load_json,
    _load_protocol,
    _load_supervision,
    _matrix,
)
from reliable_endo_gs.geometry.rectified import crop_resize_camera, reconstruct_rectified


def training_depth_sample(
    adapter: ScaredA00Adapter, sample_id: str, output_hw: tuple[int, int]
) -> ScaredA00Sample:
    """Metadata/GT-only CPU adapter for train statistics; RGB slots are placeholders.

    Calls A00's canonical protocol, TIFF and camera conversions without decoding
    RGB. Only image headers determine the source grid. Never pass this sample to
    a model or a photometric loss. Placeholders are [1,3,H,W], truth [1,1,H,W].
    """
    from PIL import Image

    if sample_id.split("/")[0] not in {"dataset_1", "dataset_2", "dataset_3", "dataset_6"}:
        raise ValueError("statistics accept train_1236 only")
    if sample_id not in adapter.manifest.sample_ids:
        raise ValueError("sample is absent from the pinned training manifest")
    adapter.roles.validate_sample_id(sample_id, role="final_train", cache=False)
    left, right, frame, calibration, disparity, _, keyframe = adapter._paths(
        sample_id, include_supervision=True
    )
    frame_data, newpram = _load_json(frame, "frame_data"), _load_json(calibration, "newpram_data")
    protocol = _load_protocol(keyframe, frame_data, newpram)
    with Image.open(left) as image:
        width, height = image.size
    with Image.open(right) as image:
        if image.size != (width, height):
            raise ValueError("left/right image grids differ")
    camera = RectifiedStereoCamera(
        _matrix(newpram.get("intr0"), "intr0", (3, 3)).float()[None],
        _matrix(newpram.get("intr1"), "intr1", (3, 3)).float()[None],
        _extract_relative_transform(protocol, frame_data, newpram).float()[None],
        True,
    )
    camera = crop_resize_camera(
        camera,
        left_crop_xy=(0, 0),
        right_crop_xy=(0, 0),
        crop_hw=(height, width),
        output_hw=output_hw,
    )
    truth = _load_supervision(
        disparity,
        input_hw=(height, width),
        crop_xy_left=(0, 0),
        crop_xy_right=(0, 0),
        crop_hw=(height, width),
        output_hw=output_hw,
        device="cpu",
    )
    image = torch.zeros(1, 3, *output_hw)
    batch = DeploymentStereoBatch(
        image, image, camera, torch.ones(1, 1, *output_hw, dtype=torch.bool)
    )
    return ScaredA00Sample(sample_id, "final_train", batch, {"statistics_only": True}, truth)


class TrainingDepthStatistics:
    """Collect a declared, frame-stratified sample of optical-z in metres.

    Each input is an A00 train sample with RGB [1,3,H,W] and separate disparity
    supervision [1,1,H,W]. At most ``pixels_per_frame`` valid pixels are selected
    at row-major midpoint quantiles. This is a sampled, not all-pixel percentile.
    """

    def __init__(self, pixels_per_frame: int = 4096) -> None:
        if type(pixels_per_frame) is not int or pixels_per_frame < 1:
            raise ValueError("pixels_per_frame must be positive")
        self.pixels_per_frame = pixels_per_frame
        self.values: list[np.ndarray] = []
        self.sample_ids: list[str] = []
        self.valid_pixels = 0
        self.empty_frames = 0

    def update(self, sample: ScaredA00Sample) -> None:
        """Accept only datasets 1/2/3/6 and final_train, with explicit GT."""
        if sample.data_role != "final_train" or sample.sample_id.split("/")[0] not in {
            "dataset_1",
            "dataset_2",
            "dataset_3",
            "dataset_6",
        }:
            raise ValueError("statistics accept train_1236 only")
        if sample.sample_id in self.sample_ids:
            raise ValueError("duplicate training sample")
        if sample.supervision is None:
            raise ValueError("training depth statistics require separate supervision")
        truth = sample.supervision
        geometry = reconstruct_rectified(
            truth.disparity,
            sample.batch.camera,
            truth.valid_mask & sample.batch.support,
            min_effective_disparity=1e-6,
        )
        depth = geometry.depth.depth
        mask = geometry.depth.valid_mask & torch.isfinite(depth) & (depth > 0)
        selected = depth[mask].detach().cpu().numpy()
        count = len(selected)
        self.sample_ids.append(sample.sample_id)
        self.valid_pixels += count
        if not count:
            self.empty_frames += 1
            return
        take = min(count, self.pixels_per_frame)
        indices = np.floor((np.arange(take) + 0.5) * count / take).astype(np.int64)
        self.values.append(selected[indices].copy())

    def result(self) -> dict[str, Any]:
        """Return q0.5/q99.5 and 10% padded bounds, without GT scale fitting."""
        if not self.values:
            raise ValueError("no positive valid training depth")
        values = np.concatenate(self.values)
        low, high = np.quantile(values, [0.005, 0.995], method="linear")
        near, far = float(0.9 * low), float(1.1 * high)
        if not np.isfinite([near, far]).all() or not 0 < near < far:
            raise ValueError("invalid train-derived depth bounds")
        return {
            "frames": len(self.sample_ids),
            "empty_frames": self.empty_frames,
            "valid_pixels": self.valid_pixels,
            "sampled_pixels": len(values),
            "quantile_population": "frame-stratified selected valid training pixels",
            "q005_m": float(low),
            "q995_m": float(high),
            "near_m": near,
            "far_m": far,
        }


def both_view_mse(rgb: torch.Tensor, batch: DeploymentStereoBatch) -> torch.Tensor:
    """Mean supported RGB MSE over both cameras; rgb is [1,2,3,H,W].

    Only deployment support is used. GT validity and predicted masks cannot
    remove difficult pixels from this training objective.
    """
    target = torch.stack((batch.left, batch.right), dim=1)
    if rgb.shape != target.shape or rgb.shape[0] != 1:
        raise ValueError("RGB must match both full input grids [1,2,3,H,W]")
    mask = batch.support[:, None].expand_as(target)
    if not bool(mask.any()):
        raise ValueError("empty deployment support")
    return (rgb.float() - target.float()).square()[mask].mean()


def posed_training_rgb(
    predictor: PosedGaussianPredictor, batch: DeploymentStereoBatch
) -> torch.Tensor:
    """Native encoder + two-camera FP32 renderer -> [1,2,3,H,W], with gradients.

    Caller controls trainability and encoder autocast. Native Gaussians are
    explicitly cast to FP32 for the unchanged CUDA rasterizer; casts retain
    gradients. This does not change the inference-only predictor interface.
    """
    context = posed_context(batch, predictor.config)
    output = predictor.encoder(context, predictor.config.global_step, deterministic=False)
    gaussians = output["gaussians"] if isinstance(output, dict) else output
    gaussians = replace(
        gaussians,
        means=gaussians.means.float(),
        covariances=gaussians.covariances.float(),
        harmonics=gaussians.harmonics.float(),
        opacities=gaussians.opacities.float(),
    )
    with torch.autocast(device_type=batch.left.device.type, enabled=False):
        decoded = predictor.decoder(
            gaussians,
            context["extrinsics"],
            context["intrinsics"],
            context["near"],
            context["far"],
            tuple(batch.left.shape[-2:]),
        )
    return decoded.color


def splatt3r_views(batch: DeploymentStereoBatch) -> tuple[dict, dict]:
    """Uncropped native views: img [-1,1] [1,3,H,W], true_shape [1,2]."""
    h, w = batch.left.shape[-2:]
    if h % 16 or w % 16:
        raise ValueError("Splatt3R requires whole 16-pixel patches; no silent crop")
    return tuple(
        {
            "img": image * 2 - 1,
            "original_img": image,
            "true_shape": torch.tensor([[h, w]], device=image.device),
            "idx": index,
            "instance": [str(index)],
        }
        for index, image in enumerate((batch.left, batch.right))
    )  # type: ignore[return-value]


def load_splatt3r_encoder(checkpoint: Path) -> nn.Module:
    """Strict-load the official native Gaussian-head encoder on CPU.

    Launch from the pinned Splatt3R repository with its vendored packages on
    PYTHONPATH. Constructor settings are those in its official main.py. Caller
    must verify source/checkpoint hashes first. Only the encoder is constructed:
    no Lightning trainer, LPIPS downloads or dataset loader is instantiated.
    """
    module = importlib.import_module("mast3r.model")
    encoder = module.AsymmetricMASt3R(
        pos_embed="RoPE100",
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 512),
        head_type="gaussian_head",
        output_mode="pts3d+gaussian+desc24",
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
        two_confs=True,
        use_offsets=True,
        sh_degree=1,
    )
    # Official Lightning checkpoint includes OmegaConf hyperparameters.
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        k.removeprefix("encoder."): v
        for k, v in raw["state_dict"].items()
        if k.startswith("encoder.")
    }
    encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False)
    encoder.downstream_head1.gaussian_dpt.dpt.requires_grad_(True)
    encoder.downstream_head2.gaussian_dpt.dpt.requires_grad_(True)
    return encoder


def splatt3r_training_rgb(
    encoder: nn.Module, batch: DeploymentStereoBatch, config: Any
) -> torch.Tensor:
    """Native Splatt3R head predictions + native CUDA RGB [1,2,3,H,W].

    Uses the official covariance/SH helpers and renderer, with the existing A00
    camera bridge. Native hardcoded 0.1/1000 bounds are replaced at the call
    boundary by explicit train-derived metre bounds. No scale/pose fitting.
    """
    geometry = importlib.import_module("utils.geometry")
    sh_utils = importlib.import_module("utils.sh_utils")
    raster = importlib.import_module("src.pixelsplat_src.cuda_splatting")
    views = splatt3r_views(batch)
    predictions = encoder(*views)
    h, w = batch.left.shape[-2:]
    means, covariances, harmonics, opacities = [], [], [], []
    for prediction, view in zip(predictions, views, strict=True):
        means.append(prediction["means"].float().reshape(1, h * w, 3))
        covariance = geometry.build_covariance(prediction["scales"], prediction["rotations"])
        covariances.append(covariance.float().reshape(1, h * w, 3, 3))
        residual = torch.zeros_like(prediction["sh"])
        residual[..., 0] = sh_utils.RGB2SH(view["original_img"].permute(0, 2, 3, 1))
        harmonics.append((prediction["sh"] + residual).float().reshape(1, h * w, 3, -1))
        opacities.append(prediction["opacities"].float().reshape(1, h * w))
    context = posed_context(batch, config)
    with torch.autocast(device_type=batch.left.device.type, enabled=False):
        rgb = raster.render_cuda(
            context["extrinsics"][0],
            context["intrinsics"][0],
            context["near"][0],
            context["far"][0],
            (h, w),
            batch.left.new_zeros(2, 3),
            torch.cat(means, dim=1).expand(2, -1, -1).contiguous(),
            torch.cat(covariances, dim=1).expand(2, -1, -1, -1).contiguous(),
            torch.cat(harmonics, dim=1).expand(2, -1, -1, -1).contiguous(),
            torch.cat(opacities, dim=1).expand(2, -1).contiguous(),
        )
    return rgb[None]


def load_pixelsplat_predictor(
    upstream: Path, checkpoint: Path, near_m: float, far_m: float
) -> PosedGaussianPredictor:
    """Strict native camera-ready two-view pixelSplat load; training probe only.

    Its native three-Gaussian sampling is retained, so the existing one-Gaussian
    inference geometry interface is intentionally not claimed to be supported.
    Native DINO constructor assets must be cached/hashed on D: by the caller.
    """
    from dacite import from_dict
    from omegaconf import OmegaConf

    from reliable_endo_gs.baseline.posed_gaussians import PosedGaussianConfig

    global_cfg = importlib.import_module("src.global_cfg")
    global_cfg.set_cfg(
        OmegaConf.create(
            {
                "mode": "test",
                "seed": 20261006,
                "dataset": {"view_sampler": {"num_context_views": 2}},
            }
        )
    )
    module = importlib.import_module("src.model.encoder.encoder_epipolar")
    cfg = OmegaConf.load(upstream / "config/model/encoder/epipolar.yaml")
    del cfg["defaults"]
    cfg.backbone = OmegaConf.load(upstream / "config/model/encoder/backbone/dino.yaml")
    encoder = module.EncoderEpipolar(
        from_dict(module.EncoderEpipolarCfg, OmegaConf.to_container(cfg))
    )
    decoder_module = importlib.import_module("src.model.decoder.decoder_splatting_cuda")
    decoder = decoder_module.DecoderSplattingCUDA(
        decoder_module.DecoderSplattingCUDACfg(name="splatting_cuda"),
        type("DatasetCfg", (), {"background_color": [0.0, 0.0, 0.0]})(),
    )
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        k.removeprefix("encoder."): v
        for k, v in raw["state_dict"].items()
        if k.startswith("encoder.")
    }
    encoder.load_state_dict(state, strict=True)
    return PosedGaussianPredictor(
        encoder, decoder, PosedGaussianConfig(near_m, far_m, int(raw.get("global_step", 300001)))
    )
