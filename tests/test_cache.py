import tempfile
import unittest
from pathlib import Path

import torch

from dinov3_representation_lab.cache import FeatureCache, cache_key


class FeatureCacheTests(unittest.TestCase):
    def test_cache_is_resumable_and_loads_in_sample_order(self) -> None:
        metadata = {
            "format_version": 1,
            "model_revision": "0" * 40,
            "pooling": "cls",
            "split": "val",
            "dtype": "float32",
        }
        ranges = ((0, 2), (2, 3))
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = FeatureCache(Path(temporary_directory), metadata, ranges)
            cache.initialize()
            self.assertFalse(cache.is_complete())
            cache.write_chunk(0, 2, torch.ones((2, 4)), torch.tensor([1, 2]))
            self.assertEqual(cache.missing_ranges(), ((2, 3),))
            cache.write_chunk(2, 3, torch.zeros((1, 4)), torch.tensor([3]))

            embeddings, labels = cache.load()

            self.assertTrue(cache.is_complete())
            self.assertEqual(tuple(embeddings.shape), (3, 4))
            self.assertEqual(labels.tolist(), [1, 2, 3])

    def test_cache_key_changes_when_pooling_changes(self) -> None:
        cls = {"model_revision": "0" * 40, "pooling": "cls"}
        mean_patch = {**cls, "pooling": "mean_patch"}

        self.assertNotEqual(cache_key(cls), cache_key(mean_patch))
