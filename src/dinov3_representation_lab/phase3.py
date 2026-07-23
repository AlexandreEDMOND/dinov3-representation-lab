"""Phase 3: evaluate frozen global-feature caches with three classifiers."""

from __future__ import annotations

import argparse
import csv
import json
import time
import tomllib
from pathlib import Path
from typing import Any

from .cache import FeatureCache
from .cli import run_smoke, validate_config
from .data import ImageNetLayout, samples_for_split, select_deterministic_subset, validate_imagenet_layout
from .phase1 import _require_pinned_revision, _section
from .phase2 import _batch_ranges, _metadata, _sample_fingerprint, _subset_size_for_split


def _torch():
    import torch

    return torch


def _device(requested: str):
    torch = _torch()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device requests CUDA, but CUDA is unavailable")
    return device


def _cache_for(
    cache_root: Path,
    *,
    model: dict[str, object],
    revision: str,
    resolution: int,
    layer: str,
    pooling: str,
    split: str,
    dtype: str,
    samples,
    layout: ImageNetLayout,
    batch_size: int,
) -> FeatureCache:
    import hashlib

    class_mapping_sha256 = hashlib.sha256(
        json.dumps(layout.class_to_idx, sort_keys=True).encode()
    ).hexdigest()
    metadata = _metadata(
        model=model,
        revision=revision,
        checkpoint_sha256=None,
        resolution=resolution,
        layer=layer,
        pooling=pooling,
        split=split,
        dtype=dtype,
        sample_count=len(samples),
        samples_sha256=_sample_fingerprint(samples, layout.root),
        class_mapping_sha256=class_mapping_sha256,
    )
    return FeatureCache(cache_root, metadata, _batch_ranges(len(samples), batch_size))


def _topk_from_scores(scores, maximum_k: int):
    torch = _torch()
    if scores.ndim != 2:
        raise ValueError("scores must be a two-dimensional tensor")
    k = min(maximum_k, scores.shape[1])
    if k < 1:
        raise ValueError("At least one class is required for evaluation")
    return torch.topk(scores, k=k, dim=1).indices


def _knn_predictions(
    train_embeddings,
    train_labels,
    validation_embeddings,
    *,
    num_classes: int,
    k: int,
    query_chunk_size: int,
    reference_chunk_size: int,
    device,
):
    """Return top-five class rankings using exact chunked cosine k-NN search."""
    torch = _torch()
    if k < 1 or k > len(train_labels):
        raise ValueError("benchmark.knn_k must be between 1 and the train-cache size")
    if query_chunk_size < 1 or reference_chunk_size < 1:
        raise ValueError("benchmark chunk sizes must be positive")

    rankings = []
    for query_start in range(0, len(validation_embeddings), query_chunk_size):
        queries = validation_embeddings[query_start : query_start + query_chunk_size].to(
            device, dtype=torch.float32
        )
        best_scores = None
        best_labels = None
        for reference_start in range(0, len(train_embeddings), reference_chunk_size):
            references = train_embeddings[
                reference_start : reference_start + reference_chunk_size
            ].to(device, dtype=torch.float32)
            similarities = queries @ references.T
            labels = train_labels[
                reference_start : reference_start + reference_chunk_size
            ].to(device)
            candidate_scores, candidate_indices = torch.topk(
                similarities, k=min(k, similarities.shape[1]), dim=1
            )
            candidate_labels = labels[candidate_indices]
            if best_scores is None:
                best_scores, best_labels = candidate_scores, candidate_labels
            else:
                merged_scores = torch.cat((best_scores, candidate_scores), dim=1)
                merged_labels = torch.cat((best_labels, candidate_labels), dim=1)
                best_scores, positions = torch.topk(merged_scores, k=k, dim=1)
                best_labels = merged_labels.gather(1, positions)

        votes = torch.zeros((queries.shape[0], num_classes), device=device)
        # Count votes (the conventional k-NN decision rule); add a tiny similarity
        # term only to resolve equal vote counts using the closer neighbourhood.
        votes.scatter_add_(1, best_labels, torch.ones_like(best_scores))
        votes.scatter_add_(1, best_labels, best_scores * 1e-6)
        rankings.append(_topk_from_scores(votes, 5).cpu())
    return torch.cat(rankings)


def _fit_linear_classifier(
    train_embeddings,
    train_labels,
    *,
    num_classes: int,
    method: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device,
):
    """Fit either SGD multinomial logistic regression or an AdamW linear probe."""
    torch = _torch()
    if method not in {"logistic_regression", "linear_probe"}:
        raise ValueError(f"Unsupported linear classifier method: {method}")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Linear-classifier hyperparameters must be positive (weight_decay non-negative)")

    torch.manual_seed(seed)
    model = torch.nn.Linear(train_embeddings.shape[1], num_classes).to(device)
    if method == "logistic_regression":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(epochs):
        indices = torch.randperm(len(train_labels), generator=generator)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            inputs = train_embeddings[batch_indices].to(device, dtype=torch.float32)
            labels = train_labels[batch_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(inputs), labels).backward()
            optimizer.step()
    return model


def _linear_predictions(model, validation_embeddings, *, chunk_size: int, device):
    torch = _torch()
    rankings = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(validation_embeddings), chunk_size):
            logits = model(validation_embeddings[start : start + chunk_size].to(device, dtype=torch.float32))
            rankings.append(_topk_from_scores(logits, 5).cpu())
    return torch.cat(rankings)


def _metrics(rankings, labels, *, num_classes: int) -> tuple[dict[str, Any], Any]:
    """Calculate Top-1/5, macro class accuracy, and the full confusion matrix."""
    torch = _torch()
    if rankings.shape[0] != labels.shape[0]:
        raise ValueError("Prediction and target counts differ")
    top1 = rankings[:, 0]
    correct = top1.eq(labels)
    top5 = rankings.eq(labels[:, None]).any(dim=1)
    confusion = torch.bincount(
        labels.to(torch.int64) * num_classes + top1.to(torch.int64),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    class_counts = confusion.sum(dim=1)
    per_class = torch.zeros(num_classes, dtype=torch.float64)
    present = class_counts > 0
    per_class[present] = (
        confusion.diag()[present].to(torch.float64)
        / class_counts[present].to(torch.float64)
    )
    per_class_accuracy = [
        float(per_class[index]) if bool(present[index]) else None
        for index in range(num_classes)
    ]
    return (
        {
            "sample_count": int(labels.shape[0]),
            "top1": float(correct.float().mean()),
            "top5": float(top5.float().mean()),
            "macro_per_class_accuracy": float(per_class[present].mean()) if present.any() else 0.0,
            "evaluated_class_count": int(present.sum()),
            "per_class_accuracy": per_class_accuracy,
        },
        confusion,
    )


def _write_predictions(path: Path, rankings, labels, samples, classes: tuple[str, ...]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("path", "true_label", "true_class", "top1_label", "top1_class", "top5_labels", "top5_classes"),
        )
        writer.writeheader()
        for sample, target, predicted in zip(samples, labels.tolist(), rankings.tolist()):
            writer.writerow(
                {
                    "path": str(sample.path),
                    "true_label": target,
                    "true_class": classes[target],
                    "top1_label": predicted[0],
                    "top1_class": classes[predicted[0]],
                    "top5_labels": " ".join(str(label) for label in predicted),
                    "top5_classes": " | ".join(classes[label] for label in predicted),
                }
            )


def _write_confusion(path: Path, confusion, classes: tuple[str, ...]) -> None:
    path.write_text(
        json.dumps({"class_names": classes, "counts": confusion.tolist()}, indent=2) + "\n"
    )


def run_frozen_feature_benchmark(
    config_path: Path,
    imagenet_root: Path | None = None,
    output_dir: Path | None = None,
    feature_cache_dir: Path | None = None,
) -> Path:
    """Benchmark CLS and mean-patch caches without loading the DINOv3 backbone."""
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    validate_config(config)
    experiment = _section(config, "experiment")
    runtime = _section(config, "runtime")
    paths = _section(config, "paths")
    model = _section(config, "model")
    features = _section(config, "features")
    dataset = _section(config, "dataset")
    cache_config = _section(config, "cache")
    benchmark = _section(config, "benchmark")
    revision = _require_pinned_revision(model.get("revision"))
    destination = output_dir or Path(str(paths["output_dir"]))
    run_smoke(config_path, destination)
    root = imagenet_root or Path(str(paths["data_dir"])) / "imagenet"
    layout = validate_imagenet_layout(root)
    seed = int(experiment["seed"])
    train_samples = select_deterministic_subset(
        samples_for_split(layout, "train"), size=_subset_size_for_split(dataset, "train"), seed=seed
    )
    validation_samples = select_deterministic_subset(
        samples_for_split(layout, "val"), size=_subset_size_for_split(dataset, "val"), seed=seed
    )
    cache_root = feature_cache_dir or Path(str(paths["feature_cache_dir"]))
    batch_size = int(dataset["batch_size"])
    device = _device(str(runtime["device"]))
    torch = _torch()

    report: dict[str, Any] = {
        "model_id": model["id"],
        "model_revision": revision,
        "backbone_loaded": False,
        "device": str(device),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "class_names": list(layout.classes),
        "benchmark": benchmark,
        "methods": {},
    }
    for pooling in ("cls", "mean_patch"):
        train_cache = _cache_for(
            cache_root, model=model, revision=revision, resolution=int(features["resolution"]),
            layer=str(features["layer"]), pooling=pooling, split="train", dtype=str(cache_config["dtype"]),
            samples=train_samples, layout=layout, batch_size=batch_size,
        )
        validation_cache = _cache_for(
            cache_root, model=model, revision=revision, resolution=int(features["resolution"]),
            layer=str(features["layer"]), pooling=pooling, split="val", dtype=str(cache_config["dtype"]),
            samples=validation_samples, layout=layout, batch_size=batch_size,
        )
        train_embeddings, train_labels = train_cache.load()
        validation_embeddings, validation_labels = validation_cache.load()
        if train_labels.device.type != "cpu" or validation_labels.device.type != "cpu":
            raise RuntimeError("Feature caches must load CPU labels")
        if train_embeddings.shape[1] != validation_embeddings.shape[1]:
            raise ValueError("Train and validation embedding dimensions differ")

        pooling_report: dict[str, Any] = {
            "cache": {"train": str(train_cache.path), "val": str(validation_cache.path)},
            "embedding_dimension": int(train_embeddings.shape[1]),
            "methods": {},
        }
        for method in ("knn", "logistic_regression", "linear_probe"):
            started = time.perf_counter()
            if method == "knn":
                rankings = _knn_predictions(
                    train_embeddings, train_labels, validation_embeddings,
                    num_classes=len(layout.classes), k=int(benchmark["knn_k"]),
                    query_chunk_size=int(benchmark["query_chunk_size"]),
                    reference_chunk_size=int(benchmark["reference_chunk_size"]), device=device,
                )
            else:
                classifier = _fit_linear_classifier(
                    train_embeddings, train_labels, num_classes=len(layout.classes), method=method,
                    epochs=int(benchmark[f"{method}_epochs"]), batch_size=int(benchmark["train_batch_size"]),
                    learning_rate=float(benchmark[f"{method}_learning_rate"]),
                    weight_decay=float(benchmark[f"{method}_weight_decay"]), seed=seed, device=device,
                )
                rankings = _linear_predictions(
                    classifier, validation_embeddings, chunk_size=int(benchmark["query_chunk_size"]), device=device
                )
            measurements, confusion = _metrics(rankings, validation_labels, num_classes=len(layout.classes))
            prediction_path = destination / "predictions" / f"{pooling}-{method}.csv"
            confusion_path = destination / "metrics" / f"{pooling}-{method}-confusion.json"
            _write_predictions(prediction_path, rankings, validation_labels, validation_samples, layout.classes)
            _write_confusion(confusion_path, confusion, layout.classes)
            pooling_report["methods"][method] = {
                **measurements,
                "runtime_seconds": time.perf_counter() - started,
                "predictions_path": str(prediction_path),
                "confusion_matrix_path": str(confusion_path),
            }
        report["methods"][pooling] = pooling_report
    report_path = destination / "metrics" / "frozen-feature-benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark frozen DINOv3 global feature caches.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase3-smoke.toml"))
    parser.add_argument("--imagenet-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--feature-cache-dir", type=Path)
    args = parser.parse_args()
    result_path = run_frozen_feature_benchmark(
        args.config, args.imagenet_root, args.output_dir, args.feature_cache_dir
    )
    print(f"Wrote frozen-feature benchmark to {result_path}")
    return 0
