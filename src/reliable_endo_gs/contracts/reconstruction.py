"""Phase I reconstruction state contract."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

from reliable_endo_gs.contracts.common import freeze_mapping
from reliable_endo_gs.contracts.gaussians import GaussianField
from reliable_endo_gs.contracts.rendering import RenderOutput
from reliable_endo_gs.contracts.samples import StereoBatch
from reliable_endo_gs.contracts.stereo import StereoPrediction


@dataclass(frozen=True, slots=True)
class ReconstructionState:
    """Immutable-container bridge from Phase I to optional future Phase II.

    The state contains only input, prediction, Gaussian, render, diagnostic,
    and provenance data. It deliberately contains no router, action, region,
    oracle label, or budget object. Tensor objects are referenced rather than
    copied; callers remain responsible for not mutating tensor storage.
    """

    SCHEMA_NAME: ClassVar[str] = "reconstruction_state"
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    batch: StereoBatch
    stereo: StereoPrediction
    gaussians: GaussianField
    render_left: RenderOutput
    render_right: RenderOutput | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.batch, StereoBatch):
            raise TypeError("batch must be a StereoBatch")
        if not isinstance(self.stereo, StereoPrediction):
            raise TypeError("stereo must be a StereoPrediction")
        if not isinstance(self.gaussians, GaussianField):
            raise TypeError("gaussians must be a GaussianField")
        if not isinstance(self.render_left, RenderOutput):
            raise TypeError("render_left must be a RenderOutput")
        if self.render_right is not None and not isinstance(self.render_right, RenderOutput):
            raise TypeError("render_right must be a RenderOutput or None")

        batch_size = self.batch.batch_size
        spatial_shape = self.batch.spatial_shape
        if self.stereo.batch_size != batch_size:
            raise ValueError(
                "stereo batch dimension must match batch; "
                f"expected {batch_size}, got {self.stereo.batch_size}"
            )
        if self.stereo.spatial_shape != spatial_shape:
            raise ValueError(
                f"stereo spatial shape must match batch {spatial_shape}; "
                f"got {self.stereo.spatial_shape}"
            )
        if self.gaussians.batch_size != batch_size:
            raise ValueError(
                "gaussians batch dimension must match batch; "
                f"expected {batch_size}, got {self.gaussians.batch_size}"
            )
        self._validate_render("render_left", self.render_left, batch_size, spatial_shape)
        if self.render_right is not None:
            self._validate_render("render_right", self.render_right, batch_size, spatial_shape)

        expected_device = self.batch.device
        for name, device in (
            ("stereo", self.stereo.device),
            ("gaussians", self.gaussians.device),
            ("render_left", self.render_left.device),
        ):
            if device != expected_device:
                raise ValueError(
                    f"{name} must be on the same device as batch ({expected_device}); got {device}"
                )
        if self.render_right is not None and self.render_right.device != expected_device:
            raise ValueError(
                "render_right must be on the same device as batch "
                f"({expected_device}); got {self.render_right.device}"
            )

        object.__setattr__(
            self, "diagnostics", freeze_mapping(self.diagnostics, name="diagnostics")
        )
        object.__setattr__(self, "provenance", freeze_mapping(self.provenance, name="provenance"))

    @staticmethod
    def _validate_render(
        name: str,
        render: RenderOutput,
        batch_size: int,
        spatial_shape: tuple[int, int],
    ) -> None:
        if render.batch_size != batch_size:
            raise ValueError(
                f"{name} batch dimension must match batch; "
                f"expected {batch_size}, got {render.batch_size}"
            )
        if render.spatial_shape != spatial_shape:
            raise ValueError(
                f"{name} spatial shape must match batch {spatial_shape}; got {render.spatial_shape}"
            )
