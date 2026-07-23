"""Phase 4: deterministic PCA and cosine-similarity views of DINOv3 patch tokens."""

from __future__ import annotations

import argparse
import json
import math
import time
import tomllib
from pathlib import Path
from typing import Any

from .backbone import BackboneSpec, extract_patch_tokens, load_backbone, make_lvd1689m_eval_transform
from .cli import run_smoke, validate_config
from .data import ImageNetSample, samples_for_split, select_deterministic_subset, validate_imagenet_layout
from .phase1 import _require_pinned_revision, _section
from .phase2 import _load_batch


def _torch():
    import torch

    return torch


def _square_grid_size(patch_tokens) -> int:
    patch_count = int(patch_tokens.shape[-2])
    grid_size = math.isqrt(patch_count)
    if grid_size * grid_size != patch_count:
        raise ValueError(f"Patch-token count {patch_count} does not form a square grid")
    return grid_size


def _fit_pca(patch_tokens, *, seed: int):
    """Fit a deterministic three-component PCA on CPU patch-token rows."""
    torch = _torch()
    if patch_tokens.ndim != 2 or patch_tokens.shape[0] < 3:
        raise ValueError("PCA needs a two-dimensional sample with at least three patch tokens")
    torch.manual_seed(seed)
    sample = patch_tokens.to(dtype=torch.float32, device="cpu")
    mean = sample.mean(dim=0)
    _, _, right_vectors = torch.linalg.svd(sample - mean, full_matrices=False)
    components = right_vectors[:3]
    projected = (sample - mean) @ components.T
    lower = torch.quantile(projected, 0.01, dim=0)
    upper = torch.quantile(projected, 0.99, dim=0)
    upper = torch.maximum(upper, lower + torch.finfo(projected.dtype).eps)
    return mean, components, lower, upper


def _pca_rgb(patch_tokens, mean, components, lower, upper):
    torch = _torch()
    grid_size = _square_grid_size(patch_tokens)
    projected = (patch_tokens.to(dtype=torch.float32, device="cpu") - mean) @ components.T
    rgb = ((projected - lower) / (upper - lower)).clamp(0, 1)
    return rgb.reshape(grid_size, grid_size, 3).numpy()


def _cosine_similarity_grid(patch_tokens, *, row: int, column: int):
    torch = _torch()
    grid_size = _square_grid_size(patch_tokens)
    if not 0 <= row < grid_size or not 0 <= column < grid_size:
        raise ValueError(f"Query patch ({row}, {column}) is outside the {grid_size}x{grid_size} grid")
    normalized = torch.nn.functional.normalize(patch_tokens.to(dtype=torch.float32, device="cpu"), dim=1)
    query = normalized[row * grid_size + column]
    return (normalized @ query).reshape(grid_size, grid_size).numpy()


def _render_figure(path: Path, original_image: Path, rgb_map, similarity_map, *, metadata: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency availability is checked by uv
        raise RuntimeError("Install plotting dependencies with 'uv sync' before running Phase 4.") from error

    with Image.open(original_image) as image:
        image_rgb = image.convert("RGB").resize((224, 224))
    figure, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input (224 px)")
    axes[1].imshow(rgb_map, interpolation="nearest")
    axes[1].set_title("Patch-token PCA (RGB)")
    similarity = axes[2].imshow(similarity_map, cmap="viridis", vmin=-1, vmax=1, interpolation="nearest")
    axes[2].set_title("Cosine similarity")
    figure.colorbar(similarity, ax=axes[2], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(metadata, fontsize=9)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_patch_visualization(
    config_path: Path,
    imagenet_root: Path | None = None,
    image_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Generate a PCA and query-patch cosine figure for a selected dataset image."""
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    validate_config(config)
    experiment = _section(config, "experiment")
    runtime = _section(config, "runtime")
    paths = _section(config, "paths")
    model = _section(config, "model")
    features = _section(config, "features")
    dataset = _section(config, "dataset")
    visualization = _section(config, "visualization")
    revision = _require_pinned_revision(model.get("revision"))
    destination = output_dir or Path(str(paths["output_dir"]))
    run_smoke(config_path, destination)
    root = imagenet_root or Path(str(paths["data_dir"])) / "imagenet"
    layout = validate_imagenet_layout(root)
    split = str(visualization.get("pca_split", dataset["split"]))
    selected = select_deterministic_subset(
        samples_for_split(layout, split),
        size=int(visualization["pca_sample_size"]),
        seed=int(experiment["seed"]),
    )
    image_index = int(visualization["image_index"])
    if not 0 <= image_index < len(selected):
        raise ValueError("visualization.image_index must reference the PCA sample")
    selected_image = image_path or selected[image_index].path
    if not selected_image.is_file():
        raise ValueError(f"Selected image does not exist: {selected_image}")

    started = time.perf_counter()
    transform = make_lvd1689m_eval_transform(int(features["resolution"]))
    backbone = load_backbone(
        BackboneSpec(str(model["id"]), revision, str(runtime["device"]))
    )
    device = next(backbone.parameters()).device
    sample_tokens = []
    batch_size = int(visualization["pca_batch_size"])
    if batch_size <= 0:
        raise ValueError("visualization.pca_batch_size must be positive")
    for start in range(0, len(selected), batch_size):
        pixels, _ = _load_batch(selected[start : start + batch_size], transform)
        sample_tokens.append(
            extract_patch_tokens(backbone, pixels.to(device), layer=str(features["layer"])).cpu()
        )
    torch = _torch()
    pca_sample = torch.cat(sample_tokens).flatten(0, 1)
    mean, components, lower, upper = _fit_pca(pca_sample, seed=int(experiment["seed"]))

    target_sample = ImageNetSample(path=selected_image, label=0)
    pixels, _ = _load_batch((target_sample,), transform)
    target_tokens = extract_patch_tokens(backbone, pixels.to(device), layer=str(features["layer"]))[0].cpu()
    grid_size = _square_grid_size(target_tokens)
    query_row = int(visualization.get("query_patch_row", grid_size // 2))
    query_column = int(visualization.get("query_patch_column", grid_size // 2))
    rgb_map = _pca_rgb(target_tokens, mean, components, lower, upper)
    similarity_map = _cosine_similarity_grid(target_tokens, row=query_row, column=query_column)
    figure_path = destination / "figures" / "patch-token-pca-and-similarity.png"
    metadata_text = (
        f"DINOv3 ViT-S/16 | layer={features['layer']} | grid={grid_size}x{grid_size} | "
        f"query=({query_row}, {query_column}) | PCA sample={len(selected)} images"
    )
    _render_figure(figure_path, selected_image, rgb_map, similarity_map, metadata=metadata_text)
    report: dict[str, Any] = {
        "model_id": model["id"],
        "model_revision": revision,
        "device": str(device),
        "layer": features["layer"],
        "resolution": features["resolution"],
        "image_path": str(selected_image),
        "pca": {
            "split": split,
            "sample_size": len(selected),
            "sample_paths": [str(sample.path.relative_to(layout.root)) for sample in selected],
            "fit_rows": int(pca_sample.shape[0]),
            "embedding_dimension": int(pca_sample.shape[1]),
            "components": components.tolist(),
            "mean": mean.tolist(),
            "lower_percentile": lower.tolist(),
            "upper_percentile": upper.tolist(),
        },
        "query_patch": {"row": query_row, "column": query_column, "grid_size": grid_size},
        "figure_path": str(figure_path),
        "runtime_seconds": time.perf_counter() - started,
    }
    report_path = destination / "metrics" / "patch-visualization.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render DINOv3 patch-token PCA and cosine-similarity maps.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase4-smoke.toml"))
    parser.add_argument("--imagenet-root", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report_path = run_patch_visualization(args.config, args.imagenet_root, args.image, args.output_dir)
    print(f"Wrote patch visualization report to {report_path}")
    return 0
