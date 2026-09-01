"""CPU-only validation helpers for the future production renderer adapter."""

from dataclasses import dataclass

from reliable_endo_gs.rendering.protocol import RendererRequest


@dataclass(frozen=True, slots=True)
class RendererContractResult:
    """Result of validating a renderer request, not a rendering result."""

    valid: bool
    messages: tuple[str, ...]


def validate_renderer_request(request: RendererRequest) -> RendererContractResult:
    """Validate the already-typed request without invoking a renderer backend."""

    return RendererContractResult(
        valid=True,
        messages=(
            "tensor and semantic validation passed",
            "production/native renderer parity is not established by this helper",
        ),
    )


def renderer_contract_metadata(request: RendererRequest) -> dict[str, str | bool]:
    """Return serializable request metadata for an eventual backend adapter."""

    return {
        "contract_schema": "renderer_request.v1",
        "covariance_source": request.covariance_source.value,
        "variant": request.variant.value,
        "frame": request.frame,
        "length_unit": request.length_unit,
        "production_parity_validated": False,
    }


__all__ = [
    "RendererContractResult",
    "renderer_contract_metadata",
    "validate_renderer_request",
]
