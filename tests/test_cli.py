import json
import tempfile
import unittest
from pathlib import Path

from dinov3_representation_lab.cli import run_smoke


class SmokeCommandTests(unittest.TestCase):
    def test_writes_resolved_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "smoke.toml"
            output_dir = root / "output"
            config_path.write_text(
                "[experiment]\nname = 'test'\nseed = 7\n"
                "[runtime]\ndevice = 'cpu'\n"
                "[paths]\ndata_dir = 'data'\noutput_dir = 'unused'\n"
            )

            result_path = run_smoke(config_path, output_dir)

            self.assertEqual(result_path, output_dir / "resolved-config.json")
            for artifact_directory in ("figures", "logs", "metrics", "predictions"):
                self.assertTrue((output_dir / artifact_directory).is_dir())
            result = json.loads(result_path.read_text())
            self.assertEqual(result["config"]["experiment"]["seed"], 7)
            self.assertEqual(result["config"]["runtime"]["device"], "cpu")
