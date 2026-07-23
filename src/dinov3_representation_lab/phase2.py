"""Phase 2 command: resumable extraction of normalized global feature caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from .backbone import BackboneSpec, extract_final_features, load_backbone, make_lvd1689m_eval_transform
from .cache import FeatureCache
from .cli import run_smoke, validate_config
from .data import ImageNetLayout, ImageNetSample, samples_for_split, select_deterministic_subset, validate_imagenet_layout
from .phase1 import _require_pinned_revision, _section


def _batch_ranges(total: int, batch_size: int) -> tuple[tuple[int, int], ...]:
    if batch_size <= 0:
        raise ValueError("dataset.batch_size must be positive")
    return tuple((start, min(start + batch_size, total)) for start in range(0, total, batch_size))


def _sample_fingerprint(samples: tuple[ImageNetSample, ...], root: Path) -> str:
    manifest = [f"{sample.path.relative_to(root)}\t{sample.label}" for sample in samples]
    return hashlib.sha256("\n".join(manifest).encode()).hexdigest()


def _metadata(
    *,
    model: dict[str, object],
    revision: str,
    checkpoint_sha256: str | None,
    resolution: int,
    layer: str,
    pooling: str,
    split: str,
    dtype: str,
    sample_count: int,
    samples_sha256: str,
    class_mapping_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_id": model["id"],
        "model_revision": revision,
        "checkpoint_sha256": checkpoint_sha256,
        "transform": {
            "name": "dinov3_lvd1689m_eval",
            "resolution": resolution,
            "resize": "square_antialias",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "layer": layer,
        "pooling": pooling,
        "split": split,
        "dtype": dtype,
        "sample_count": sample_count,
        "samples_sha256": samples_sha256,
        "class_mapping_sha256": class_mapping_sha256,
        "normalized": True,
    }


def _load_batch(samples: tuple[ImageNetSample, ...], transform):
    from PIL import Image
    from .backbone import _torch

    tensors = []
    labels = []
    for sample in samples:
        with Image.open(sample.path) as image:
            tensors.append(transform(image.convert("RGB")))
        labels.append(sample.label)
    return _torch().stack(tensors), _torch().tensor(labels, dtype=_torch().long)


def _normalized_embeddings(global_embeddings, patch_tokens):
    from .backbone import _torch

    torch = _torch()
    return {
        "cls": torch.nn.functional.normalize(global_embeddings, dim=1),
        "mean_patch": torch.nn.functional.normalize(patch_tokens.mean(dim=1), dim=1),
    }


def _dtype(name: str):
    from .backbone import _torch

    choices = {"float32": _torch().float32, "float16": _torch().float16}
    if name not in choices:
        raise ValueError("cache.dtype must be 'float32' or 'float16'")
    return choices[name]


def _subset_size_for_split(dataset: dict[str, object], split: str) -> int:
    """Return an optional split-specific cache size, preserving the Phase 2 default."""
    split_sizes = dataset.get("split_sizes")
    if split_sizes is None:
        return int(dataset["subset_size"])
    if not isinstance(split_sizes, dict) or split not in split_sizes:
        raise ValueError("dataset.split_sizes must specify every requested split")
    size = int(split_sizes[split])
    if size <= 0:
        raise ValueError("dataset.split_sizes values must be positive")
    return size


def run_feature_cache(
    config_path: Path,
    imagenet_root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Create or reuse `[CLS]` and mean-patch global embedding caches."""
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
    revision = _require_pinned_revision(model.get("revision"))

    destination = output_dir or Path(str(paths["output_dir"]))
    run_smoke(config_path, destination)
    root = imagenet_root or Path(str(paths["data_dir"])) / "imagenet"
    layout = validate_imagenet_layout(root)
    splits = tuple(str(split) for split in dataset.get("splits", [dataset["split"]]))
    if not splits or any(split not in {"train", "val"} for split in splits):
        raise ValueError("dataset.splits must contain 'train' and/or 'val'")
    dtype_name = str(cache_config["dtype"])
    output_dtype = _dtype(dtype_name)
    checkpoint_sha256 = None
    class_mapping_sha256 = hashlib.sha256(
        json.dumps(layout.class_to_idx, sort_keys=True).encode()
    ).hexdigest()

    caches: dict[str, dict[str, FeatureCache]] = {}
    selected_by_split: dict[str, tuple[ImageNetSample, ...]] = {}
    for split in splits:
        subset_size = _subset_size_for_split(dataset, split)
        ranges = _batch_ranges(subset_size, int(dataset["batch_size"]))
        selected_samples = select_deterministic_subset(
            samples_for_split(layout, split), size=subset_size, seed=int(experiment["seed"])
        )
        selected_by_split[split] = selected_samples
        samples_sha256 = _sample_fingerprint(selected_samples, layout.root)
        caches[split] = {}
        for pooling in ("cls", "mean_patch"):
            cache = FeatureCache(
                destination / "feature-cache",
                _metadata(
                    model=model,
                    revision=revision,
                    checkpoint_sha256=checkpoint_sha256,
                    resolution=int(features["resolution"]),
                    layer=str(features["layer"]),
                    pooling=pooling,
                    split=split,
                    dtype=dtype_name,
                    sample_count=subset_size,
                    samples_sha256=samples_sha256,
                    class_mapping_sha256=class_mapping_sha256,
                ),
                ranges,
            )
            cache.initialize()
            caches[split][pooling] = cache

    missing = {
        split: {pooling: cache.missing_ranges() for pooling, cache in split_caches.items()}
        for split, split_caches in caches.items()
    }
    backbone_loaded = False
    if any(ranges for split_ranges in missing.values() for ranges in split_ranges.values()):
        transform = make_lvd1689m_eval_transform(int(features["resolution"]))
        backbone = load_backbone(
            BackboneSpec(
                model_id=str(model["id"]),
                revision=revision,
                device=str(runtime["device"]),
            )
        )
        backbone_loaded = True
        device = next(backbone.parameters()).device
        for split, split_caches in caches.items():
            required_ranges = sorted(
                {item for pool_ranges in missing[split].values() for item in pool_ranges}
            )
            for start, stop in required_ranges:
                pixels, labels = _load_batch(selected_by_split[split][start:stop], transform)
                global_embeddings, patch_tokens = extract_final_features(backbone, pixels.to(device))
                for pooling, embeddings in _normalized_embeddings(global_embeddings, patch_tokens).items():
                    cache = split_caches[pooling]
                    if (start, stop) in missing[split][pooling]:
                        cache.write_chunk(
                            start,
                            stop,
                            embeddings.to(dtype=output_dtype),
                            labels,
                        )

    completion = {
        split: {pooling: cache.is_complete() for pooling, cache in split_caches.items()}
        for split, split_caches in caches.items()
    }
    if not all(done for split_done in completion.values() for done in split_done.values()):
        raise RuntimeError("Feature cache extraction did not complete.")
    record = {
        "checkpoint_source": model["id"],
        "backbone_loaded": backbone_loaded,
        "cache": {
            split: {
                pooling: {"key": cache.key, "path": str(cache.path), "metadata": cache.metadata}
                for pooling, cache in split_caches.items()
            }
            for split, split_caches in caches.items()
        },
    }
    result_path = destination / "metrics" / "feature-cache-run.json"
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or reuse resumable normalized DINOv3 global feature caches."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/phase2-smoke.toml"))
    parser.add_argument("--imagenet-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result_path = run_feature_cache(
        config_path=args.config,
        imagenet_root=args.imagenet_root,
        output_dir=args.output_dir,
    )
    print(f"Wrote feature cache report to {result_path}")
    return 0
