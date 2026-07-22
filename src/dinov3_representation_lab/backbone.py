"""Frozen DINOv3 backbone loading and final-token feature extraction."""

from __future__ import annotations

from dataclasses import dataclass


LVD1689M_MEAN = (0.485, 0.456, 0.406)
LVD1689M_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BackboneSpec:
    model_id: str
    revision: str
    device: str


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only without dependencies
        raise RuntimeError("Install project dependencies with 'uv sync' before running Phase 1.") from error
    return torch


def resolve_device(requested_device: str) -> str:
    """Resolve ``auto`` without silently selecting a different explicit device."""
    torch = _torch()
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested_device


def make_lvd1689m_eval_transform(resolution: int):
    """Return Meta's LVD-1689M evaluation transform at a square resolution.

    The official DINOv3 repository specifies ToImage, square Resize with antialiasing,
    float conversion in [0, 1], then ImageNet normalization for LVD-1689M weights.
    """
    if resolution <= 0 or resolution % 16:
        raise ValueError("Resolution must be a positive multiple of the ViT patch size (16).")
    torch = _torch()
    try:
        from torchvision.transforms import v2
    except ImportError as error:  # pragma: no cover - exercised only without dependencies
        raise RuntimeError("Install torchvision with 'uv sync' before running Phase 1.") from error
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((resolution, resolution), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=LVD1689M_MEAN, std=LVD1689M_STD),
        ]
    )


def load_backbone(spec: BackboneSpec):
    """Load the pinned official Hugging Face checkpoint in frozen evaluation mode."""
    _torch()
    try:
        from transformers import AutoModel
    except ImportError as error:  # pragma: no cover - exercised only without dependencies
        raise RuntimeError("Install transformers with 'uv sync' before running Phase 1.") from error

    device = resolve_device(spec.device)
    try:
        model = AutoModel.from_pretrained(spec.model_id, revision=spec.revision)
    except OSError as error:
        raise RuntimeError(
            "Could not download the DINOv3 checkpoint. Accept its Hugging Face access "
            "conditions and authenticate locally before retrying."
        ) from error
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract_final_features(model, pixel_values):
    """Return class embeddings and final-layer patch tokens without gradients."""
    torch = _torch()
    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values)

    final_tokens = outputs.last_hidden_state
    if final_tokens is None:
        raise RuntimeError("The loaded model did not return last_hidden_state.")
    global_embeddings = outputs.pooler_output
    if global_embeddings is None:
        global_embeddings = final_tokens[:, 0]

    register_tokens = int(getattr(model.config, "num_register_tokens", 0))
    patch_tokens = final_tokens[:, 1 + register_tokens :]
    if not patch_tokens.shape[1]:
        raise RuntimeError("The model output does not contain patch tokens.")
    if global_embeddings.requires_grad or patch_tokens.requires_grad:
        raise RuntimeError("Frozen extraction unexpectedly produced tensors requiring gradients.")
    return global_embeddings, patch_tokens
