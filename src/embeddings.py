from __future__ import annotations

from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        """Create an embedding client configured with one OpenAI model."""
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks for indexing."""
        if not texts:
            return []

        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question so it can be compared with stored chunk vectors."""
        response = self._client.embeddings.create(model=self._model, input=[text])
        return response.data[0].embedding
