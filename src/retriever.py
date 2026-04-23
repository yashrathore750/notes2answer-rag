from __future__ import annotations

from dataclasses import dataclass

from .chunker import Chunk
from .embeddings import OpenAIEmbedder
from .vector_store import FaissVectorStore


@dataclass(frozen=True)
class RetrievalResult:
    score: float
    chunk: Chunk


class Retriever:
    def __init__(self, embedder: OpenAIEmbedder, store: FaissVectorStore) -> None:
        """Bind the embedding model and vector store into one retrieval service."""
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Embed the question and fetch the most similar stored chunks."""
        query_vector = self.embedder.embed_query(query)
        raw_results = self.store.search(query_vector, top_k=top_k)
        return [RetrievalResult(score=s, chunk=c) for s, c in raw_results]
