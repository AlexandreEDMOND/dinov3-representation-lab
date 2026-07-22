"""Small command-line utilities shared by future experiments."""

import argparse
import json
import platform
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path


def run_smoke(config_path: Path, output_dir: Path | None = None) -> Path:
    """Write a resolved experiment record without requiring data or a GPU."""
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    destination = output_dir or Path(config["paths"]["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    for artifact_directory in ("figures", "logs", "metrics", "predictions"):
        (destination / artifact_directory).mkdir(exist_ok=True)
    resolved_config = {
        "config": config,
        "config_path": str(config_path),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
    }
    result_path = destination / "resolved-config.json"
    result_path.write_text(json.dumps(resolved_config, indent=2) + "\n")
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the experiment configuration without data or a model."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.toml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    result_path = run_smoke(args.config, args.output_dir)
    print(f"Wrote resolved configuration to {result_path}")
    return 0
