from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .chunker import Chunk


class FaissVectorStore:
    def __init__(self, dimension: int) -> None:
        """Create an in-memory FAISS index for vectors of a fixed size."""
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[Chunk] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors so inner product behaves like cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def add(self, vectors: list[list[float]], chunks: list[Chunk]) -> None:
        """Add chunk vectors and their metadata to the search index."""
        if not vectors:
            return

        matrix = np.array(vectors, dtype="float32")
        matrix = self._normalize(matrix)
        self.index.add(matrix)
        self.metadata.extend(chunks)

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[float, Chunk]]:
        """Return the top matching chunks for one embedded query."""
        if self.index.ntotal == 0:
            return []

        query = np.array([query_vector], dtype="float32")
        query = self._normalize(query)

        scores, indices = self.index.search(query, top_k)
        matches: list[tuple[float, Chunk]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            matches.append((float(score), self.metadata[idx]))

        return matches

    def save(self, index_path: Path, metadata_path: Path) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        payload = [asdict(item) for item in self.metadata]
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "FaissVectorStore":
        """Reload a previously saved FAISS index and its chunk metadata."""
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Vector index files are missing. Run ingest first to build the index."
            )

        index = faiss.read_index(str(index_path))
        store = cls(index.d)
        store.index = index

        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        store.metadata = [Chunk(**item) for item in raw]
        return store
