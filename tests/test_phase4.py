import unittest

import torch

from dinov3_representation_lab.phase4 import _cosine_similarity_grid, _pca_rgb, _square_grid_size


class PatchVisualizationTests(unittest.TestCase):
    def test_patch_grid_and_cosine_query_map(self) -> None:
        patch_tokens = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

        similarity = _cosine_similarity_grid(patch_tokens, row=0, column=0)

        self.assertEqual(_square_grid_size(patch_tokens), 2)
        self.assertEqual(similarity.tolist(), [[1.0, 0.0], [1.0, 0.0]])

    def test_pca_rgb_returns_a_patch_grid(self) -> None:
        patch_tokens = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        mean = torch.zeros(6)
        components = torch.eye(6)[:3]
        lower = torch.zeros(3)
        upper = torch.full((3,), 10.0)

        rgb = _pca_rgb(patch_tokens, mean, components, lower, upper)

        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertEqual(float(rgb[0, 0, 0]), 0.0)
