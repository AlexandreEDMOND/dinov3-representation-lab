"""Phase 1 command: validate data and extract frozen features for ten images."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

from .backbone import BackboneSpec, extract_final_features, load_backbone, make_lvd1689m_eval_transform
from .cli import run_smoke, validate_config
from .data import samples_for_split, select_deterministic_subset, validate_imagenet_layout


def _section(config: dict[str, object], name: str) -> dict[str, object]:
    values = config.get(name)
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section [{name}] must be a table.")
    return values


def _require_pinned_revision(revision: object) -> str:
    if not isinstance(revision, str) or len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision.lower()
    ):
        raise ValueError("model.revision must be a 40-character commit SHA.")
    return revision


def _file_sha256(path: Path) -> str | None:
    """Hash local weights when available to make an external checkpoint traceable."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_feature_smoke(
    config_path: Path,
    imagenet_root: Path | None = None,
    output_dir: Path | None = None,
    model_path: Path | None = None,
) -> Path:
    """Extract and record final DINOv3 features for a seeded ten-image subset."""
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    validate_config(config)

    experiment = _section(config, "experiment")
    runtime = _section(config, "runtime")
    paths = _section(config, "paths")
    model = _section(config, "model")
    features = _section(config, "features")
    dataset = _section(config, "dataset")
    revision = _require_pinned_revision(model.get("revision"))

    destination = output_dir or Path(str(paths["output_dir"]))
    run_smoke(config_path, destination)
    root = imagenet_root or Path(str(paths["data_dir"])) / "imagenet"
    layout = validate_imagenet_layout(root)
    split = str(dataset["split"])
    samples = samples_for_split(layout, split)
    subset_size = int(dataset.get("subset_size", 10))
    selected_samples = select_deterministic_subset(
        samples, size=subset_size, seed=int(experiment["seed"])
    )

    transform = make_lvd1689m_eval_transform(int(features["resolution"]))
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency installation is tested by uv
        raise RuntimeError("Install Pillow with 'uv sync' before running Phase 1.") from error
    from .backbone import _torch

    torch = _torch()
    image_tensors = []
    for sample in selected_samples:
        with Image.open(sample.path) as image:
            image_tensors.append(transform(image.convert("RGB")))
    pixel_values = torch.stack(image_tensors)

    backbone = load_backbone(
        BackboneSpec(
            model_id=str(model["id"]),
            revision=revision,
            device=str(runtime["device"]),
            local_path=model_path,
        )
    )
    device = next(backbone.parameters()).device
    global_embeddings, patch_tokens = extract_final_features(backbone, pixel_values.to(device))

    class_mapping = json.dumps(layout.class_to_idx, sort_keys=True).encode()
    record = {
        "model_id": model["id"],
        "model_revision": revision,
        "checkpoint_source": str(model_path.resolve()) if model_path is not None else model["id"],
        "checkpoint_sha256": _file_sha256(model_path / "model.safetensors")
        if model_path is not None
        else None,
        "device": str(device),
        "split": split,
        "seed": experiment["seed"],
        "resolution": features["resolution"],
        "subset_size": len(selected_samples),
        "class_mapping_sha256": hashlib.sha256(class_mapping).hexdigest(),
        "global_embedding_shape": list(global_embeddings.shape),
        "final_patch_token_shape": list(patch_tokens.shape),
        "requires_grad": {
            "global_embeddings": global_embeddings.requires_grad,
            "patch_tokens": patch_tokens.requires_grad,
        },
        "samples": [
            {
                "path": str(sample.path.relative_to(layout.root)),
                "label": sample.label,
                "class_name": layout.classes[sample.label],
            }
            for sample in selected_samples
        ],
    }
    result_path = destination / "metrics" / "feature-smoke.json"
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 global and final patch-token features for a ten-image subset."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/phase1-smoke.toml"))
    parser.add_argument("--imagenet-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Local Transformers checkpoint directory; its SHA-256 is recorded in metrics.",
    )
    args = parser.parse_args()
    result_path = run_feature_smoke(
        config_path=args.config,
        imagenet_root=args.imagenet_root,
        output_dir=args.output_dir,
        model_path=args.model_path,
    )
    print(f"Wrote feature smoke metrics to {result_path}")
    return 0
