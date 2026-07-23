"""Phase 5: controlled depth, pooling, resolution, robustness and retrieval analyses."""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from .backbone import BackboneSpec, load_backbone, make_lvd1689m_eval_transform
from .cli import run_smoke, validate_config
from .data import ImageNetSample, samples_for_split, select_deterministic_subset, validate_imagenet_layout
from .phase1 import _require_pinned_revision, _section
from .phase2 import _load_batch
from .phase3 import _fit_linear_classifier, _linear_predictions, _metrics
from .phase4 import _fit_pca, _pca_rgb


def _torch():
    import torch

    return torch


def _layer_tokens(outputs, layer: str):
    requested = layer.strip().lower()
    if requested == "final":
        tokens = outputs.last_hidden_state
    else:
        try:
            index = int(requested)
        except ValueError as error:
            raise ValueError("analysis.depths must contain 'final' or hidden-state indices") from error
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("The backbone did not return requested hidden states")
        try:
            tokens = hidden_states[index]
        except IndexError as error:
            raise ValueError(f"Hidden-state index {index} is unavailable") from error
    if tokens is None:
        raise RuntimeError(f"Layer {layer} did not return features")
    return tokens


def _extract_features(backbone, samples, *, transform, layers: tuple[str, ...], batch_size: int, keep_patches: bool):
    """Extract normalized global features for several layers in one backbone pass."""
    torch = _torch()
    if batch_size < 1:
        raise ValueError("analysis.batch_size must be positive")
    device = next(backbone.parameters()).device
    registers = int(getattr(backbone.config, "num_register_tokens", 0))
    result: dict[str, dict[str, list[Any]]] = {
        layer: {"cls": [], "mean_patch": [], "patches": []} for layer in layers
    }
    labels = []
    for start in range(0, len(samples), batch_size):
        pixels, batch_labels = _load_batch(samples[start : start + batch_size], transform)
        with torch.inference_mode():
            outputs = backbone(pixel_values=pixels.to(device), output_hidden_states=True)
        for layer in layers:
            tokens = _layer_tokens(outputs, layer)
            patches = tokens[:, 1 + registers :]
            if patches.shape[1] == 0:
                raise RuntimeError("Selected layer has no patch tokens")
            result[layer]["cls"].append(torch.nn.functional.normalize(tokens[:, 0], dim=1).cpu())
            result[layer]["mean_patch"].append(
                torch.nn.functional.normalize(patches.mean(dim=1), dim=1).cpu()
            )
            if keep_patches:
                result[layer]["patches"].append(patches.cpu())
        labels.append(batch_labels)
    combined = {
        layer: {
            pooling: torch.cat(values) if values else None
            for pooling, values in feature_types.items()
        }
        for layer, feature_types in result.items()
    }
    return combined, torch.cat(labels)


def _peak_memory_bytes(device) -> int | None:
    torch = _torch()
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _render_depth_pca(path: Path, image_path: Path, depth_maps: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    with Image.open(image_path) as image:
        input_image = image.convert("RGB").resize((224, 224))
    figure, axes = plt.subplots(1, len(depth_maps) + 1, figsize=(4 * (len(depth_maps) + 1), 4))
    axes[0].imshow(input_image)
    axes[0].set_title("Input")
    axes[0].set_axis_off()
    for axis, (layer, rgb) in zip(axes[1:], depth_maps.items()):
        axis.imshow(rgb, interpolation="nearest")
        axis.set_title(f"PCA layer {layer}")
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _render_robustness(path: Path, records: dict[str, dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    names = list(records)
    cosine = [records[name]["mean_cosine_similarity"] for name in names]
    consistency = [records[name]["retrieval_consistency"] for name in names]
    positions = list(range(len(names)))
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(positions, cosine, marker="o", label="Mean cosine similarity")
    axis.plot(positions, consistency, marker="o", label="Top-1 retrieval consistency")
    axis.set_xticks(positions, names, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Final-layer CLS robustness on deterministic perturbations")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _render_retrieval(path: Path, queries, references, indices, classes: tuple[str, ...]) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = len(queries)
    columns = indices.shape[1] + 1
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    for row, (query, neighbours) in enumerate(zip(queries, indices.tolist())):
        items = [query] + [references[index] for index in neighbours]
        titles = [f"Query\n{classes[query.label]}"] + [
            f"#{rank + 1}: {classes[references[index].label]}" for rank, index in enumerate(neighbours)
        ]
        for axis, sample, title in zip(axes[row], items, titles):
            with Image.open(sample.path) as image:
                axis.imshow(image.convert("RGB").resize((160, 160)))
            axis.set_title(title, fontsize=8)
            axis.set_axis_off()
    figure.suptitle("Final-layer CLS nearest neighbours", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _perturb(image, name: str):
    from PIL import ImageDraw, ImageEnhance

    image = image.convert("RGB")
    width, height = image.size
    if name == "identity":
        return image
    if name == "crop":
        margin_x, margin_y = width // 10, height // 10
        return image.crop((margin_x, margin_y, width - margin_x, height - margin_y))
    if name == "rotation":
        return image.rotate(20, expand=False)
    if name == "colour":
        return ImageEnhance.Color(image).enhance(0.2)
    if name == "occlusion":
        result = image.copy()
        draw = ImageDraw.Draw(result)
        side = min(width, height) // 3
        left, top = (width - side) // 2, (height - side) // 2
        draw.rectangle((left, top, left + side, top + side), fill=(0, 0, 0))
        return result
    raise ValueError(f"Unknown perturbation: {name}")


def _perturbed_features(backbone, samples, *, transform, perturbation: str, layer: str):
    """Extract final pooled CLS features after one deterministic image perturbation."""
    from PIL import Image

    torch = _torch()
    device = next(backbone.parameters()).device
    registers = int(getattr(backbone.config, "num_register_tokens", 0))
    tensors = []
    for sample in samples:
        with Image.open(sample.path) as image:
            tensors.append(transform(_perturb(image, perturbation)))
    with torch.inference_mode():
        outputs = backbone(pixel_values=torch.stack(tensors).to(device), output_hidden_states=True)
    tokens = _layer_tokens(outputs, layer)
    if tokens.shape[1] <= registers:
        raise RuntimeError("Selected robustness layer has no CLS token")
    return torch.nn.functional.normalize(tokens[:, 0], dim=1).cpu()


def _nearest_indices(query_embeddings, reference_embeddings, *, k: int):
    torch = _torch()
    if not 1 <= k <= len(reference_embeddings):
        raise ValueError("analysis.retrieval_k must be within the reference set size")
    return torch.topk(query_embeddings @ reference_embeddings.T, k=k, dim=1).indices


def _diverse_sample_indices(samples, count: int) -> tuple[int, ...]:
    """Choose deterministic retrieval queries from as many classes as possible."""
    if count < 1:
        raise ValueError("analysis.retrieval_queries must be positive")
    chosen = []
    seen_labels = set()
    for index, sample in enumerate(samples):
        if sample.label not in seen_labels:
            chosen.append(index)
            seen_labels.add(sample.label)
            if len(chosen) == count:
                return tuple(chosen)
    for index in range(len(samples)):
        if index not in chosen:
            chosen.append(index)
            if len(chosen) == count:
                return tuple(chosen)
    return tuple(chosen)


def run_controlled_analysis(
    config_path: Path,
    imagenet_root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Run the Phase 5 controlled representation analyses without fine-tuning DINOv3."""
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    validate_config(config)
    experiment = _section(config, "experiment")
    runtime = _section(config, "runtime")
    paths = _section(config, "paths")
    model = _section(config, "model")
    features = _section(config, "features")
    dataset = _section(config, "dataset")
    analysis = _section(config, "analysis")
    revision = _require_pinned_revision(model.get("revision"))
    destination = output_dir or Path(str(paths["output_dir"]))
    run_smoke(config_path, destination)
    root = imagenet_root or Path(str(paths["data_dir"])) / "imagenet"
    layout = validate_imagenet_layout(root)
    seed = int(experiment["seed"])
    subset_size = int(dataset["subset_size"])
    train_samples = select_deterministic_subset(samples_for_split(layout, "train"), size=subset_size, seed=seed)
    validation_samples = select_deterministic_subset(samples_for_split(layout, "val"), size=subset_size, seed=seed)
    layers = tuple(str(layer) for layer in analysis["depths"])
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("analysis.depths must contain distinct layers")
    resolutions = tuple(int(resolution) for resolution in analysis["resolutions"])
    if int(features["resolution"]) not in resolutions:
        raise ValueError("analysis.resolutions must include features.resolution")

    started = time.perf_counter()
    torch = _torch()
    torch.manual_seed(seed)
    backbone = load_backbone(BackboneSpec(str(model["id"]), revision, str(runtime["device"])))
    device = next(backbone.parameters()).device
    batch_sizes = {str(key): int(value) for key, value in analysis["batch_sizes"].items()}
    base_resolution = int(features["resolution"])
    base_transform = make_lvd1689m_eval_transform(base_resolution)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    base_started = time.perf_counter()
    train_base, train_labels = _extract_features(
        backbone, train_samples, transform=base_transform, layers=layers,
        batch_size=batch_sizes[str(base_resolution)], keep_patches=False,
    )
    validation_base, validation_labels = _extract_features(
        backbone, validation_samples, transform=base_transform, layers=layers,
        batch_size=batch_sizes[str(base_resolution)], keep_patches=True,
    )
    base_runtime = time.perf_counter() - base_started
    base_peak_memory = _peak_memory_bytes(device)

    depth_pooling: dict[str, dict[str, Any]] = {}
    for layer in layers:
        depth_pooling[layer] = {}
        for pooling in ("cls", "mean_patch"):
            classifier = _fit_linear_classifier(
                train_base[layer][pooling], train_labels, num_classes=len(layout.classes), method="linear_probe",
                epochs=int(analysis["probe_epochs"]), batch_size=int(analysis["probe_batch_size"]),
                learning_rate=float(analysis["probe_learning_rate"]),
                weight_decay=float(analysis["probe_weight_decay"]), seed=seed, device=device,
            )
            rankings = _linear_predictions(
                classifier, validation_base[layer][pooling], chunk_size=int(analysis["probe_batch_size"]), device=device
            )
            metrics, _ = _metrics(rankings, validation_labels, num_classes=len(layout.classes))
            depth_pooling[layer][pooling] = metrics

    resolution_results: dict[str, dict[str, Any]] = {
        str(base_resolution): {
            "runtime_seconds": base_runtime,
            "peak_memory_bytes": base_peak_memory,
            "final_cls": depth_pooling["final"]["cls"],
        }
    }
    for resolution in resolutions:
        if resolution == base_resolution:
            continue
        transform = make_lvd1689m_eval_transform(resolution)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        extraction_started = time.perf_counter()
        train_features, _ = _extract_features(
            backbone, train_samples, transform=transform, layers=("final",),
            batch_size=batch_sizes[str(resolution)], keep_patches=False,
        )
        validation_features, _ = _extract_features(
            backbone, validation_samples, transform=transform, layers=("final",),
            batch_size=batch_sizes[str(resolution)], keep_patches=False,
        )
        classifier = _fit_linear_classifier(
            train_features["final"]["cls"], train_labels, num_classes=len(layout.classes), method="linear_probe",
            epochs=int(analysis["probe_epochs"]), batch_size=int(analysis["probe_batch_size"]),
            learning_rate=float(analysis["probe_learning_rate"]), weight_decay=float(analysis["probe_weight_decay"]),
            seed=seed, device=device,
        )
        rankings = _linear_predictions(classifier, validation_features["final"]["cls"], chunk_size=int(analysis["probe_batch_size"]), device=device)
        metrics, _ = _metrics(rankings, validation_labels, num_classes=len(layout.classes))
        resolution_results[str(resolution)] = {
            "runtime_seconds": time.perf_counter() - extraction_started,
            "peak_memory_bytes": _peak_memory_bytes(device),
            "final_cls": metrics,
        }

    reference_embeddings = train_base["final"]["cls"]
    original_embeddings = validation_base["final"]["cls"]
    original_neighbours = _nearest_indices(original_embeddings, reference_embeddings, k=1)[:, 0]
    robustness: dict[str, dict[str, float]] = {}
    for perturbation in ("identity", "crop", "rotation", "colour", "occlusion"):
        perturbed = _perturbed_features(
            backbone, validation_samples, transform=base_transform, perturbation=perturbation,
            layer=str(analysis["robustness_layer"]),
        )
        neighbours = _nearest_indices(perturbed, reference_embeddings, k=1)[:, 0]
        cosine = (perturbed * original_embeddings).sum(dim=1)
        robustness[perturbation] = {
            "mean_cosine_similarity": float(cosine.mean()),
            "retrieval_consistency": float(neighbours.eq(original_neighbours).float().mean()),
        }

    query_positions = _diverse_sample_indices(validation_samples, int(analysis["retrieval_queries"]))
    query_count = len(query_positions)
    query_samples = tuple(validation_samples[index] for index in query_positions)
    neighbour_indices = _nearest_indices(
        original_embeddings[list(query_positions)], reference_embeddings, k=int(analysis["retrieval_k"])
    )
    all_neighbour_indices = _nearest_indices(original_embeddings, reference_embeddings, k=1)[:, 0]
    failures = []
    for query, neighbour_index in zip(validation_samples, all_neighbour_indices.tolist()):
        neighbour = train_samples[neighbour_index]
        if query.label != neighbour.label:
            failures.append(
                {
                    "category": "nearest_neighbour_class_mismatch",
                    "query_path": str(query.path.relative_to(layout.root)),
                    "query_class": layout.classes[query.label],
                    "neighbour_path": str(neighbour.path.relative_to(layout.root)),
                    "neighbour_class": layout.classes[neighbour.label],
                }
            )
    failure_counts = Counter(item["category"] for item in failures)

    depth_maps = {}
    selected_image = validation_samples[0]
    for layer in layers:
        all_patches = validation_base[layer]["patches"].flatten(0, 1)
        mean, components, lower, upper = _fit_pca(all_patches, seed=seed)
        depth_maps[layer] = _pca_rgb(validation_base[layer]["patches"][0], mean, components, lower, upper)
    _render_depth_pca(destination / "figures" / "depth-pca.png", selected_image.path, depth_maps)
    _render_robustness(destination / "figures" / "robustness.png", robustness)
    _render_retrieval(destination / "figures" / "retrieval-grid.png", query_samples, train_samples, neighbour_indices, layout.classes)
    (destination / "metrics" / "retrieval-failures.json").write_text(
        json.dumps({"counts": failure_counts, "failures": failures}, indent=2) + "\n"
    )

    early_top1 = depth_pooling[layers[0]]["cls"]["top1"]
    final_top1 = depth_pooling["final"]["cls"]["top1"]
    margin = final_top1 - early_top1
    conclusion = {
        "depth_signal": "supports_final_layer_semantic_gain" if margin > 0.05 else "inconclusive_on_smoke_subset",
        "final_minus_early_cls_top1": margin,
        "interpretation": (
            "The final CLS probe exceeds the earliest selected-layer CLS probe by more than five points."
            if margin > 0.05
            else "The 20-image-per-split smoke subset is too small for a reliable depth conclusion."
        ),
        "limitation": "This is an Imagenette smoke analysis; confirm all conclusions on ImageNet-1k.",
    }
    report = {
        "model_id": model["id"], "model_revision": revision, "device": str(device), "seed": seed,
        "train_samples": len(train_samples), "validation_samples": len(validation_samples),
        "depth_pooling_probe": depth_pooling, "resolution": resolution_results, "robustness": robustness,
        "retrieval": {
            "query_count": query_count, "k": int(analysis["retrieval_k"]),
            "failure_counts": failure_counts, "failure_evaluation_samples": len(validation_samples),
        },
        "figures": {
            "depth_pca": str(destination / "figures" / "depth-pca.png"),
            "robustness": str(destination / "figures" / "robustness.png"),
            "retrieval": str(destination / "figures" / "retrieval-grid.png"),
        },
        "conclusion": conclusion, "runtime_seconds": time.perf_counter() - started,
    }
    report_path = destination / "metrics" / "controlled-representation-analysis.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled DINOv3 representation analyses.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase5-smoke.toml"))
    parser.add_argument("--imagenet-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report_path = run_controlled_analysis(args.config, args.imagenet_root, args.output_dir)
    print(f"Wrote controlled analysis report to {report_path}")
    return 0
