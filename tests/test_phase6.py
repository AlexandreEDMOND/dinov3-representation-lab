import unittest

from dinov3_representation_lab.phase6 import _final_markdown


class BaselineComparisonTests(unittest.TestCase):
    def test_final_report_contains_baseline_rows(self) -> None:
        methods = {name: {"top1": 0.5} for name in ("knn", "logistic_regression", "linear_probe")}
        report = {
            "train_samples": 20, "validation_samples": 20, "runtime_seconds": 1.0,
            "environment": {
                "device": "cpu", "gpu": None, "python": "3.11.0", "torch": "2.7",
                "torchvision": "0.22", "transformers": "4.57",
            },
            "models": {"example": {
                "model_id": "example/model", "revision": None, "weights": None,
                "parameter_count": 10, "runtime_seconds": 0.5, "results": {"cls": methods},
            }},
        }

        markdown = _final_markdown(report)

        self.assertIn("| example | cls | 0.500 | 0.500 | 0.500 |", markdown)
        self.assertIn("Limitations", markdown)
        self.assertIn("Reproducibility and compute", markdown)
