from types import SimpleNamespace
import unittest

import torch

from dinov3_representation_lab.backbone import (
    extract_patch_tokens,
    extract_final_features,
    make_lvd1689m_eval_transform,
)


class _FakeDinov3Model:
    config = SimpleNamespace(num_register_tokens=4)

    def __call__(self, *, pixel_values: torch.Tensor, output_hidden_states: bool = False) -> SimpleNamespace:
        batch_size = pixel_values.shape[0]
        tokens = pixel_values.new_ones((batch_size, 201, 3))
        hidden_states = (tokens * 2, tokens * 3) if output_hidden_states else None
        return SimpleNamespace(
            last_hidden_state=tokens, pooler_output=tokens[:, 0], hidden_states=hidden_states
        )


class BackboneTests(unittest.TestCase):
    def test_official_transform_produces_normalized_square_tensor(self) -> None:
        transform = make_lvd1689m_eval_transform(224)
        image = torch.zeros((3, 40, 20), dtype=torch.uint8)

        transformed = transform(image)

        self.assertEqual(transformed.shape, (3, 224, 224))
        self.assertEqual(transformed.dtype, torch.float32)

    def test_extracts_class_and_final_patch_tokens_without_gradients(self) -> None:
        global_embeddings, patch_tokens = extract_final_features(
            _FakeDinov3Model(), torch.zeros((2, 3, 224, 224))
        )

        self.assertEqual(global_embeddings.shape, (2, 3))
        self.assertEqual(patch_tokens.shape, (2, 196, 3))
        self.assertFalse(global_embeddings.requires_grad)
        self.assertFalse(patch_tokens.requires_grad)

    def test_extracts_patch_tokens_from_a_selected_hidden_state(self) -> None:
        patch_tokens = extract_patch_tokens(
            _FakeDinov3Model(), torch.zeros((2, 3, 224, 224)), layer="1"
        )

        self.assertEqual(patch_tokens.shape, (2, 196, 3))
        self.assertEqual(float(patch_tokens[0, 0, 0]), 3.0)
