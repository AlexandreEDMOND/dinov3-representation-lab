"""Resumable, metadata-keyed storage for frozen global image embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _torch():
    import torch

    return torch


def canonical_json(value: dict[str, Any]) -> str:
    """Serialize cache identity deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def cache_key(metadata: dict[str, Any]) -> str:
    """Create a content key from the complete cache metadata contract."""
    return hashlib.sha256(canonical_json(metadata).encode()).hexdigest()[:20]


class FeatureCache:
    """A cache whose independently written batches can be resumed safely."""

    def __init__(
        self,
        root: Path,
        metadata: dict[str, Any],
        batch_ranges: tuple[tuple[int, int], ...],
    ) -> None:
        self.metadata = metadata
        self.batch_ranges = batch_ranges
        self.key = cache_key(metadata)
        self.path = root / metadata["split"] / metadata["pooling"] / self.key
        self.metadata_path = self.path / "metadata.json"
        self.chunks_path = self.path / "chunks"

    def initialize(self) -> None:
        """Create metadata or reject a collision with a different experiment."""
        self.chunks_path.mkdir(parents=True, exist_ok=True)
        if self.metadata_path.exists():
            existing = json.loads(self.metadata_path.read_text())
            if existing != self.metadata:
                raise ValueError(f"Cache metadata collision at {self.path}")
            return
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2) + "\n")

    def chunk_path(self, start: int, stop: int) -> Path:
        return self.chunks_path / f"{start:08d}-{stop:08d}.pt"

    def _valid_chunk(self, start: int, stop: int) -> bool:
        path = self.chunk_path(start, stop)
        if not path.is_file():
            return False
        try:
            payload = _torch().load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError):
            return False
        embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        labels = payload.get("labels") if isinstance(payload, dict) else None
        return bool(
            getattr(embeddings, "ndim", None) == 2
            and embeddings.shape[0] == stop - start
            and getattr(labels, "ndim", None) == 1
            and labels.shape[0] == stop - start
        )

    def missing_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (start, stop)
            for start, stop in self.batch_ranges
            if not self._valid_chunk(start, stop)
        )

    def is_complete(self) -> bool:
        return not self.missing_ranges()

    def write_chunk(self, start: int, stop: int, embeddings, labels) -> None:
        """Atomically write a CPU embedding batch and its labels."""
        torch = _torch()
        if embeddings.ndim != 2 or embeddings.shape[0] != stop - start:
            raise ValueError("Embedding chunk shape does not match its sample range.")
        if labels.ndim != 1 or labels.shape[0] != stop - start:
            raise ValueError("Label chunk shape does not match its sample range.")
        path = self.chunk_path(start, stop)
        temporary_path = path.with_suffix(".tmp")
        torch.save(
            {"embeddings": embeddings.cpu(), "labels": labels.cpu()}, temporary_path
        )
        temporary_path.replace(path)

    def load(self):
        """Load a validated complete cache as embeddings and labels in sample order."""
        if not self.is_complete():
            raise ValueError(f"Cache is incomplete: {self.path}")
        torch = _torch()
        chunks = [
            torch.load(self.chunk_path(start, stop), map_location="cpu", weights_only=True)
            for start, stop in self.batch_ranges
        ]
        return (
            torch.cat([chunk["embeddings"] for chunk in chunks]),
            torch.cat([chunk["labels"] for chunk in chunks]),
        )
