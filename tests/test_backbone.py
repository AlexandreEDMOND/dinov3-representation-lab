from types import SimpleNamespace
import unittest
from pathlib import Path

import torch

from dinov3_representation_lab.backbone import (
    BackboneSpec,
    extract_final_features,
    load_backbone,
    make_lvd1689m_eval_transform,
)


class _FakeDinov3Model:
    config = SimpleNamespace(num_register_tokens=4)

    def __call__(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        batch_size = pixel_values.shape[0]
        tokens = pixel_values.new_ones((batch_size, 201, 3))
        return SimpleNamespace(last_hidden_state=tokens, pooler_output=tokens[:, 0])


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

    def test_rejects_missing_local_checkpoint_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a directory"):
            load_backbone(
                BackboneSpec(
                    model_id="facebook/example",
                    revision="0" * 40,
                    device="cpu",
                    local_path=Path("missing-checkpoint"),
                )
            )
