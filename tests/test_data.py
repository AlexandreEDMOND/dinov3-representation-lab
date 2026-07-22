import tempfile
import unittest
from pathlib import Path

from dinov3_representation_lab.data import (
    samples_for_split,
    select_deterministic_subset,
    validate_imagenet_layout,
)


class ImageNetDataTests(unittest.TestCase):
    def _make_layout(self, root: Path) -> None:
        for split in ("train", "val"):
            for class_name in ("n00000002", "n00000001"):
                class_dir = root / split / class_name
                class_dir.mkdir(parents=True)
                for index in range(3):
                    (class_dir / f"image-{index}.jpg").write_bytes(b"placeholder")

    def test_validates_class_mapping_and_selects_seeded_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "imagenet"
            self._make_layout(root)

            layout = validate_imagenet_layout(root)
            samples = samples_for_split(layout, "val")
            first = select_deterministic_subset(samples, size=4, seed=7)
            second = select_deterministic_subset(samples, size=4, seed=7)

            self.assertEqual(layout.classes, ("n00000001", "n00000002"))
            self.assertEqual(layout.class_to_idx["n00000001"], 0)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 4)

    def test_rejects_mismatched_split_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "imagenet"
            (root / "train" / "n00000001").mkdir(parents=True)
            (root / "val" / "n00000002").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "must match"):
                validate_imagenet_layout(root)
