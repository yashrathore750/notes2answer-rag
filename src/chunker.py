from __future__ import annotations

from dataclasses import dataclass

from .document_loader import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str



def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split one document string into overlapping character-based chunks."""
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(clean)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = clean[start:end].strip()
        if piece:
            chunks.append(piece)

        if end == text_len:
            break

        start = max(0, end - chunk_overlap)

    return chunks



def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Convert loaded documents into chunk records ready for embedding and retrieval."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        pieces = _split_text(doc.text, chunk_size, chunk_overlap)
        for idx, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{doc.source}::chunk_{idx}",
                    source=doc.source,
                    text=piece,
                )
            )

    return all_chunks
