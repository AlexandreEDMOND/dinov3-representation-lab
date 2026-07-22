"""ImageNet-compatible dataset discovery and deterministic subset selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random


IMAGE_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png", ".bmp", ".webp"})


@dataclass(frozen=True)
class ImageNetLayout:
    """A validated ImageFolder-style ImageNet directory layout."""

    root: Path
    classes: tuple[str, ...]
    class_to_idx: dict[str, int]


@dataclass(frozen=True)
class ImageNetSample:
    """A discovered image and its stable ImageNet-compatible class index."""

    path: Path
    label: int


def _class_directories(split_dir: Path) -> tuple[str, ...]:
    return tuple(sorted(path.name for path in split_dir.iterdir() if path.is_dir()))


def validate_imagenet_layout(root: Path) -> ImageNetLayout:
    """Validate ``root/train/<class>`` and ``root/val/<class>`` directories.

    Both splits must expose the same sorted class names. This creates the class-index
    mapping independently from torchvision, which keeps metadata validation light.
    """
    root = root.resolve()
    split_directories = {split: root / split for split in ("train", "val")}
    missing = [str(path) for path in split_directories.values() if not path.is_dir()]
    if missing:
        raise ValueError(
            "ImageNet root must contain train/ and val/ directories; missing: "
            + ", ".join(missing)
        )

    train_classes = _class_directories(split_directories["train"])
    val_classes = _class_directories(split_directories["val"])
    if not train_classes:
        raise ValueError(f"No class directories found in {split_directories['train']}")
    if train_classes != val_classes:
        raise ValueError(
            "ImageNet train and val class directories must match exactly; "
            f"train has {len(train_classes)}, val has {len(val_classes)}"
        )

    return ImageNetLayout(
        root=root,
        classes=train_classes,
        class_to_idx={class_name: index for index, class_name in enumerate(train_classes)},
    )


def samples_for_split(layout: ImageNetLayout, split: str) -> tuple[ImageNetSample, ...]:
    """Return image samples in a canonical, platform-independent order."""
    if split not in {"train", "val"}:
        raise ValueError("Dataset split must be 'train' or 'val'")

    samples = []
    for class_name in layout.classes:
        class_dir = layout.root / split / class_name
        image_paths = sorted(
            path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        samples.extend(
            ImageNetSample(path=image_path, label=layout.class_to_idx[class_name])
            for image_path in image_paths
        )
    if not samples:
        raise ValueError(f"No supported image files found in {layout.root / split}")
    return tuple(samples)


def select_deterministic_subset(
    samples: tuple[ImageNetSample, ...], *, size: int, seed: int
) -> tuple[ImageNetSample, ...]:
    """Select a stable random subset after canonical ordering of the full split."""
    if size <= 0:
        raise ValueError("Subset size must be positive")
    if size > len(samples):
        raise ValueError(f"Requested {size} samples, but split contains only {len(samples)}")
    selected_indices = sorted(Random(seed).sample(range(len(samples)), size))
    return tuple(samples[index] for index in selected_indices)
