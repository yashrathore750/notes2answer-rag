from __future__ import annotations

from .retriever import RetrievalResult


# SYSTEM_PROMPT = (
#     "You are a helpful assistant that answers using the provided context. "
#     "If the answer is not in the context, say you do not know and ask the user for more data. "
#     "Cite the source identifiers you used in a short Sources section."
# )

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer ONLY using the provided context. "
    "If the context does not contain enough information to answer, "
    "respond with exactly: 'I don't have enough information in my notes to answer this.' "
    "Do not use any outside knowledge. "
    "Cite the source identifiers you used in a short Sources section."
)


def build_context_block(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks into a labeled context block for the LLM prompt."""
    lines: list[str] = []
    for idx, item in enumerate(results, start=1):
        lines.append(f"[Source {idx}] {item.chunk.source}")
        lines.append(item.chunk.text)
        lines.append("")
    return "\n".join(lines).strip()



def build_user_prompt(question: str, context_block: str) -> str:
    """Build the final user prompt that combines retrieved context and the question."""
    return (
        "Answer the question using only this context.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Provide a concise answer then a Sources section."
    )
