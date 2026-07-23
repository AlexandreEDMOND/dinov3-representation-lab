"""Phase 6: frozen DINOv3, DINOv2 and ResNet-50 baseline comparison."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

from .backbone import BackboneSpec, extract_final_features, load_backbone, make_lvd1689m_eval_transform
from .cli import run_smoke, validate_config
from .data import samples_for_split, select_deterministic_subset, validate_imagenet_layout
from .phase1 import _require_pinned_revision, _section
from .phase2 import _load_batch
from .phase3 import _fit_linear_classifier, _knn_predictions, _linear_predictions, _metrics


def _torch():
    import torch

    return torch


def _load_model(spec: dict[str, object], device):
    """Load one frozen baseline and return its model plus an extraction family."""
    torch = _torch()
    kind = str(spec["kind"])
    if kind == "dinov3":
        model = load_backbone(
            BackboneSpec(str(spec["model_id"]), _require_pinned_revision(spec["revision"]), str(device))
        )
        return model, kind
    if kind == "dinov2":
        try:
            from transformers import AutoModel
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Install project dependencies with 'uv sync'.") from error
        model = AutoModel.from_pretrained(
            str(spec["model_id"]), revision=_require_pinned_revision(spec["revision"])
        ).to(device)
    elif kind == "resnet50":
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Install torchvision with 'uv sync'.") from error
        if spec.get("weights") != "IMAGENET1K_V2":
            raise ValueError("Only ResNet50 IMAGENET1K_V2 is supported by the baseline protocol")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
        model.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported baseline kind: {kind}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, kind


def _extract_embeddings(model, kind: str, samples, *, transform, batch_size: int):
    """Extract L2-normalized representations under the shared image transform."""
    torch = _torch()
    if batch_size < 1:
        raise ValueError("benchmark.batch_size must be positive")
    device = next(model.parameters()).device
    registers = int(getattr(getattr(model, "config", None), "num_register_tokens", 0))
    representations: dict[str, list[Any]] = {}
    labels = []
    for start in range(0, len(samples), batch_size):
        pixels, batch_labels = _load_batch(samples[start : start + batch_size], transform)
        pixels = pixels.to(device)
        with torch.inference_mode():
            if kind == "dinov3":
                cls, patches = extract_final_features(model, pixels)
                batch_representations = {"cls": cls, "mean_patch": patches.mean(dim=1)}
            elif kind == "dinov2":
                outputs = model(pixel_values=pixels)
                tokens = outputs.last_hidden_state
                patches = tokens[:, 1 + registers :]
                batch_representations = {"cls": tokens[:, 0], "mean_patch": patches.mean(dim=1)}
            else:
                batch_representations = {"global_average_pool": model(pixels)}
        for pooling, values in batch_representations.items():
            representations.setdefault(pooling, []).append(torch.nn.functional.normalize(values, dim=1).cpu())
        labels.append(batch_labels)
    return {pooling: torch.cat(values) for pooling, values in representations.items()}, torch.cat(labels)


def _environment(device) -> dict[str, Any]:
    import matplotlib
    import torch
    import torchvision
    import transformers

    return {
        "platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
        "torchvision": torchvision.__version__, "transformers": transformers.__version__,
        "matplotlib": matplotlib.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def _render_comparison(path: Path, records: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    labels = []
    values = []
    for name, model in records.items():
        for pooling, methods in model["results"].items():
            labels.append(f"{name}\n{pooling}")
            values.append(methods["linear_probe"]["top1"])
    figure, axis = plt.subplots(figsize=(max(7, 1.5 * len(labels)), 4))
    bars = axis.bar(range(len(labels)), values)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Top-1 accuracy")
    axis.set_title("Frozen-feature linear-probe comparison")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _final_markdown(report: dict[str, Any]) -> str:
    environment = report["environment"]
    lines = [
        "# Phase 6 baseline report", "",
        "## Protocol", "",
        f"- Dataset: deterministic Imagenette subset ({report['train_samples']} train / {report['validation_samples']} validation).",
        "- Shared transform: square 224 px resize, ImageNet normalization.",
        "- Backbones frozen; k-NN, multinomial logistic regression, and a PyTorch linear probe are evaluated.",
        "",
        "## Results", "",
        "| Model | Pooling | k-NN Top-1 | Logistic Top-1 | Linear probe Top-1 |", "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, model in report["models"].items():
        for pooling, methods in model["results"].items():
            lines.append(
                f"| {name} | {pooling} | {methods['knn']['top1']:.3f} | "
                f"{methods['logistic_regression']['top1']:.3f} | {methods['linear_probe']['top1']:.3f} |"
            )
    lines.extend([
        "", "## Reproducibility and compute", "",
        f"- Device: `{environment['device']}` ({environment['gpu'] or 'CPU'}).",
        f"- Software: Python {environment['python'].split()[0]}, PyTorch {environment['torch']}, "
        f"TorchVision {environment['torchvision']}, Transformers {environment['transformers']}.",
        f"- Total measured runtime: {report['runtime_seconds']:.2f} seconds.", "",
        "| Model | Checkpoint / weights | Parameters | Runtime |", "| --- | --- | ---: | ---: |",
    ])
    for name, model in report["models"].items():
        checkpoint = model["model_id"] or model["weights"]
        if model["revision"]:
            checkpoint = f"{checkpoint} @ {model['revision']}"
        lines.append(
            f"| {name} | `{checkpoint}` | {model['parameter_count']:,} | {model['runtime_seconds']:.2f} s |"
        )
    lines.extend([
        "", "## Limitations", "",
        "This is an Imagenette smoke comparison. It verifies the identical frozen-feature protocol, "
        "but it is not a substitute for the required ImageNet-1k full-validation baseline table.",
    ])
    return "\n".join(lines) + "\n"


def run_baseline_comparison(config_path: Path, imagenet_root: Path | None = None, output_dir: Path | None = None) -> Path:
    """Compare frozen DINOv3, DINOv2 and ResNet-50 using one explicit protocol."""
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    validate_config(config)
    experiment = _section(config, "experiment")
    runtime = _section(config, "runtime")
    paths = _section(config, "paths")
    features = _section(config, "features")
    dataset = _section(config, "dataset")
    benchmark = _section(config, "benchmark")
    model_specs = config.get("baseline")
    if not isinstance(model_specs, list) or len(model_specs) != 3:
        raise ValueError("Configuration must define three [[baseline]] entries")
    destination = output_dir or Path(str(paths["output_dir"]))
    run_smoke(config_path, destination)
    root = imagenet_root or Path(str(paths["data_dir"])) / "imagenet"
    layout = validate_imagenet_layout(root)
    subset_size = int(dataset["subset_size"])
    seed = int(experiment["seed"])
    train_samples = select_deterministic_subset(samples_for_split(layout, "train"), size=subset_size, seed=seed)
    validation_samples = select_deterministic_subset(samples_for_split(layout, "val"), size=subset_size, seed=seed)
    torch = _torch()
    device = torch.device("cuda" if str(runtime["device"]) == "auto" and torch.cuda.is_available() else str(runtime["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device requests CUDA, but CUDA is unavailable")
    transform = make_lvd1689m_eval_transform(int(features["resolution"]))
    results: dict[str, Any] = {}
    started = time.perf_counter()
    for raw_spec in model_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError("Each [[baseline]] entry must be a table")
        spec = {str(key): value for key, value in raw_spec.items()}
        name = str(spec["name"])
        model_started = time.perf_counter()
        model, kind = _load_model(spec, device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        train_embeddings, train_labels = _extract_embeddings(
            model, kind, train_samples, transform=transform, batch_size=int(benchmark["batch_size"])
        )
        validation_embeddings, validation_labels = _extract_embeddings(
            model, kind, validation_samples, transform=transform, batch_size=int(benchmark["batch_size"])
        )
        pooling_results = {}
        for pooling in train_embeddings:
            methods = {}
            rankings = _knn_predictions(
                train_embeddings[pooling], train_labels, validation_embeddings[pooling],
                num_classes=len(layout.classes), k=int(benchmark["knn_k"]),
                query_chunk_size=int(benchmark["query_chunk_size"]),
                reference_chunk_size=int(benchmark["reference_chunk_size"]), device=device,
            )
            methods["knn"], _ = _metrics(rankings, validation_labels, num_classes=len(layout.classes))
            for method in ("logistic_regression", "linear_probe"):
                classifier = _fit_linear_classifier(
                    train_embeddings[pooling], train_labels, num_classes=len(layout.classes), method=method,
                    epochs=int(benchmark[f"{method}_epochs"]), batch_size=int(benchmark["train_batch_size"]),
                    learning_rate=float(benchmark[f"{method}_learning_rate"]),
                    weight_decay=float(benchmark[f"{method}_weight_decay"]), seed=seed, device=device,
                )
                rankings = _linear_predictions(classifier, validation_embeddings[pooling], chunk_size=int(benchmark["query_chunk_size"]), device=device)
                methods[method], _ = _metrics(rankings, validation_labels, num_classes=len(layout.classes))
            pooling_results[pooling] = methods
        results[name] = {
            "kind": kind, "model_id": spec.get("model_id"), "revision": spec.get("revision"),
            "weights": spec.get("weights"), "parameter_count": parameter_count,
            "runtime_seconds": time.perf_counter() - model_started, "results": pooling_results,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "environment": _environment(device), "seed": seed, "transform": {
            "resolution": int(features["resolution"]), "name": "square_resize_imagenet_normalization",
        },
        "train_samples": len(train_samples), "validation_samples": len(validation_samples),
        "models": results, "runtime_seconds": time.perf_counter() - started,
    }
    figure_path = destination / "figures" / "baseline-linear-probe-comparison.png"
    _render_comparison(figure_path, results)
    report["figure_path"] = str(figure_path)
    json_path = destination / "metrics" / "baseline-comparison.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path = destination / "metrics" / "phase6-final-report.md"
    markdown_path.write_text(_final_markdown(report))
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare frozen DINOv3, DINOv2 and ResNet-50 baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase6-smoke.toml"))
    parser.add_argument("--imagenet-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report_path = run_baseline_comparison(args.config, args.imagenet_root, args.output_dir)
    print(f"Wrote baseline comparison to {report_path}")
    return 0
