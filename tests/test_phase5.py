import unittest

import torch

from dinov3_representation_lab.data import ImageNetSample
from dinov3_representation_lab.phase5 import _diverse_sample_indices, _nearest_indices


class ControlledAnalysisTests(unittest.TestCase):
    def test_nearest_indices_returns_cosine_neighbours(self) -> None:
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        references = torch.tensor([[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0]])

        indices = _nearest_indices(queries, references, k=2)

        self.assertEqual(indices.tolist(), [[0, 1], [1, 0]])

    def test_diverse_query_indices_cover_labels_before_repeating(self) -> None:
        samples = tuple(ImageNetSample(path=None, label=label) for label in (0, 0, 1, 2, 1))

        indices = _diverse_sample_indices(samples, 3)

        self.assertEqual(indices, (0, 2, 3))
