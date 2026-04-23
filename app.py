from __future__ import annotations

import argparse
from pathlib import Path

from src.chat_memory import ChatMemory
from src.chunker import chunk_documents
from src.config import load_settings
from src.document_loader import load_documents
from src.embeddings import OpenAIEmbedder
from src.rag_engine import RAGEngine
from src.retriever import Retriever
from src.vector_store import FaissVectorStore


def _index_paths(storage_dir: Path) -> tuple[Path, Path]:
    """Return the standard file paths used to store the FAISS index and metadata."""
    return storage_dir / "faiss.index", storage_dir / "metadata.json"

# Ingest workflow: load documents, chunk, embed, and save the index
def run_ingest() -> None:
    """Load notes, chunk them, embed them, and save the searchable index."""
    settings = load_settings()

    docs = load_documents(settings.data_dir)
    if not docs:
        print("No documents found in data directory.")
        return

    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        print("Documents were loaded, but chunking produced no chunks.")
        return

    # Embed the chunk texts and build a FAISS index for similarity search at runtime.
    embedder = OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])

    dimension = len(vectors[0])
    store = FaissVectorStore(dimension=dimension)
    store.add(vectors, chunks)

    index_path, metadata_path = _index_paths(settings.storage_dir)
    store.save(index_path=index_path, metadata_path=metadata_path)

    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")
    print(f"Saved index to: {index_path}")


def _load_runtime() -> tuple[RAGEngine, int]:
    """Load the saved vector index and build the runtime RAG objects."""
    settings = load_settings()

    index_path, metadata_path = _index_paths(settings.storage_dir)
    store = FaissVectorStore.load(index_path=index_path, metadata_path=metadata_path)
    embedder = OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)

    retriever = Retriever(embedder=embedder, store=store)
    engine = RAGEngine(
        api_key=settings.openai_api_key,
        chat_model=settings.chat_model,
        retriever=retriever,
    )
    return engine, settings.top_k


def run_ask(question: str) -> None:
    """Answer a single question from the command line using the saved index."""
    engine, top_k = _load_runtime()
    result = engine.ask(question=question, top_k=top_k)

    print("\nAnswer:\n")
    print(result.answer)

    if result.sources:
        print("\nRetrieved sources:")
        for source in result.sources:
            print(f"- {source}")


def run_chat() -> None:
    """Start an interactive terminal chat that keeps short conversation memory."""
    engine, top_k = _load_runtime()
    memory = ChatMemory(max_turns=6)

    print("Chat started. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue

        result = engine.ask(question=question, top_k=top_k, memory=memory)
        print("\nAssistant:\n")
        print(result.answer)
        print()


def parse_args() -> argparse.Namespace:
    """Define the CLI commands and parse the user's terminal input."""
    parser = argparse.ArgumentParser(description="Beginner-friendly RAG app")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Load documents, chunk, embed, and index")

    ask_parser = sub.add_parser("ask", help="Ask one question")
    ask_parser.add_argument("question", type=str)

    sub.add_parser("chat", help="Run interactive chat with memory")

    return parser.parse_args()


def main() -> None:
    """Dispatch the selected CLI command to the correct app workflow."""
    args = parse_args()

    if args.command == "ingest":
        run_ingest()
    elif args.command == "ask":
        run_ask(args.question)
    elif args.command == "chat":
        run_chat()
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
