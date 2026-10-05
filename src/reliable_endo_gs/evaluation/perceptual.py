"""Offline LPIPS-AlexNet v0.1 from explicitly identified local weights.

Uses the existing torchvision dependency. The computation follows
richzhang/PerceptualSimilarity (LPIPS 0.1.4); parity is checked against that
package. No dependency installation, network access or implicit weight lookup.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from reliable_endo_gs.evaluation.stereo import MetricInputError, MetricResult


def file_sha256(path: Path) -> str:
    """Hash a local artifact without retaining it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LPIPSAlexNet(nn.Module):
    """Frozen LPIPS on float RGB ``[B,3,H,W]`` in [0,1], minimum side 32.

    The backbone file is torchvision's full AlexNet state_dict. Calibration
    is official ``lpips/weights/v0.1/alex.pth``. Both files are required;
    random features or uncalibrated distances are never reported as LPIPS.
    """

    shift: torch.Tensor
    scale: torch.Tensor

    def __init__(self, backbone: Path, calibration: Path) -> None:
        super().__init__()
        from torchvision.models import alexnet  # type: ignore[import-untyped]

        self.identity = {
            "backend": "lpips_alex_v0.1_local.v1",
            "reference": "richzhang/PerceptualSimilarity lpips==0.1.4",
            "backbone": str(backbone),
            "backbone_sha256": file_sha256(backbone),
            "calibration": str(calibration),
            "calibration_sha256": file_sha256(calibration),
            "input": "RGB [0,1] mapped to [-1,1], official v0.1 scaling",
        }
        # Pinned artifact identities, not machine paths. AlexNet is the
        # torchvision IMAGENET1K_V1 artifact alexnet-owt-7be5be79.pth;
        # calibration matches upstream lpips/weights/v0.1/alex.pth.
        if self.identity["backbone_sha256"] != (
            "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
        ) or self.identity["calibration_sha256"] != (
            "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
        ):
            raise MetricInputError(
                "LPIPS requires the pinned pretrained/calibrated artifact hashes"
            )
        self.features = alexnet(weights=None).features[:12]
        state = torch.load(backbone, map_location="cpu", weights_only=True)
        feature_state = {
            k.removeprefix("features."): v for k, v in state.items() if k.startswith("features.")
        }
        self.features.load_state_dict(feature_state, strict=True)
        linear = torch.load(calibration, map_location="cpu", weights_only=True)
        expected = {f"lin{i}.model.1.weight" for i in range(5)}
        if set(linear) != expected:
            raise MetricInputError("expected official five LPIPS-AlexNet v0.1 calibrated layers")
        for i, channels in enumerate((64, 192, 384, 256, 256)):
            weight = linear[f"lin{i}.model.1.weight"]
            if weight.shape != (1, channels, 1, 1) or not bool(torch.isfinite(weight).all()):
                raise MetricInputError("invalid LPIPS calibrated weight shape/value")
            self.register_buffer(f"linear_{i}", weight.float())
        if not all(bool(torch.isfinite(v).all()) for v in self.features.state_dict().values()):
            raise MetricInputError("nonfinite LPIPS backbone weights")
        self.register_buffer("shift", torch.tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([0.458, 0.448, 0.450])[None, :, None, None])
        self.requires_grad_(False)
        self.eval()

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return one perceptual distance per image, shape ``[B]``."""
        if (
            predicted.shape != target.shape
            or predicted.ndim != 4
            or predicted.shape[1] != 3
            or min(predicted.shape[-2:]) < 32
            or predicted.device != target.device
        ):
            raise MetricInputError("LPIPS needs matching RGB [B,3,H,W], H/W >=32, same device")
        if not bool((torch.isfinite(predicted) & torch.isfinite(target)).all()):
            raise MetricInputError("LPIPS requires finite full images")
        if not bool(((predicted >= 0) & (predicted <= 1) & (target >= 0) & (target <= 1)).all()):
            raise MetricInputError("LPIPS RGB must be in [0,1]")
        with torch.autocast(device_type=predicted.device.type, enabled=False):
            x = (2 * predicted.float() - 1 - self.shift) / self.scale
            y = (2 * target.float() - 1 - self.shift) / self.scale
            value = x.new_zeros(x.shape[0])
            output_index = 0
            for index, layer in enumerate(self.features):
                x, y = layer(x), layer(y)
                if index in (1, 4, 7, 9, 11):
                    nx = x / (x.square().sum(1, keepdim=True).sqrt() + 1e-10)
                    ny = y / (y.square().sum(1, keepdim=True).sqrt() + 1e-10)
                    distance = torch.nn.functional.conv2d(
                        (nx - ny).square(), getattr(self, f"linear_{output_index}")
                    )
                    value = value + distance.mean((1, 2, 3))
                    output_index += 1
        return value

    def measure(self, predicted: torch.Tensor, target: torch.Tensor) -> MetricResult:
        """Full-image mean over the batch; no masking or image resizing."""
        value = float(self(predicted, target).mean())
        return MetricResult(
            "lpips_alex",
            value,
            predicted.shape[0],
            predicted.shape[0],
            "LPIPS AlexNet v0.1 full_RGB_image per_frame_mean",
        )
