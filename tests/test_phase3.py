import unittest

import torch

from dinov3_representation_lab.phase3 import _knn_predictions, _metrics


class FrozenFeatureBenchmarkTests(unittest.TestCase):
    def test_knn_search_is_chunked_and_returns_class_rankings(self) -> None:
        train_embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        train_embeddings = torch.nn.functional.normalize(train_embeddings, dim=1)
        train_labels = torch.tensor([0, 0, 1, 1])
        validation_embeddings = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1
        )

        rankings = _knn_predictions(
            train_embeddings,
            train_labels,
            validation_embeddings,
            num_classes=2,
            k=2,
            query_chunk_size=1,
            reference_chunk_size=1,
            device=torch.device("cpu"),
        )

        self.assertEqual(rankings.tolist(), [[0, 1], [1, 0]])

    def test_metrics_include_topk_macro_accuracy_and_confusion(self) -> None:
        rankings = torch.tensor([[0, 1, 2], [1, 2, 0], [1, 0, 2]])
        labels = torch.tensor([0, 2, 1])

        metrics, confusion = _metrics(rankings, labels, num_classes=3)

        self.assertAlmostEqual(metrics["top1"], 2 / 3)
        self.assertEqual(metrics["top5"], 1.0)
        self.assertAlmostEqual(metrics["macro_per_class_accuracy"], 2 / 3)
        self.assertEqual(metrics["per_class_accuracy"], [1.0, 1.0, 0.0])
        self.assertEqual(confusion.tolist(), [[1, 0, 0], [0, 1, 0], [0, 1, 0]])
