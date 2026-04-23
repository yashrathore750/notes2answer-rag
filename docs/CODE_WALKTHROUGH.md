# Code Walkthrough

This guide maps each file to its role in the RAG pipeline.

## app.py

Orchestrates commands:

- ingest: build vector index from documents
- ask: ask one question
- chat: interactive mode with memory

## src/config.py

Loads runtime settings from environment.

Why it matters:

- Separates code from configuration
- Makes experiments easy without code edits

## src/document_loader.py

Reads files from data directory.

Supported:

- .pdf (via pypdf)
- .txt
- .md

Output:

- List of Document objects containing source path and text

## src/chunker.py

Splits documents into overlapping character windows.

Why overlap:

- Prevents context loss at chunk boundaries

Output:

- List of Chunk objects with unique ids and source mapping

## src/embeddings.py

OpenAIEmbedder wraps embedding API calls:

- embed_texts for chunk indexing
- embed_query for question retrieval

## src/vector_store.py

FaissVectorStore handles:

- Index creation using cosine-like similarity (IndexFlatIP + normalization)
- Add chunk vectors
- Search top-k nearest chunks
- Save and load index + metadata

Key detail:

- Metadata stores chunk text and source so retrieval can be traced to files.

## src/retriever.py

Retriever combines embedder + vector store:

- Embed query
- Search nearest chunks
- Return typed RetrievalResult objects

## src/prompting.py

Defines:

- System instruction for grounded behavior
- Context block formatting for retrieved chunks
- User prompt template

Good prompts reduce hallucination and improve citation behavior.

## src/chat_memory.py

Simple bounded memory:

- Stores recent user and assistant messages
- Exposes chat-format message list

## src/rag_engine.py

The main pipeline per question:

1. Retrieve top-k chunks
2. Build prompt with context
3. Call LLM
4. Return answer + deduplicated sources
5. Update memory (chat mode)

## How to Extend Safely

1. Keep each module single-purpose
2. Add tests before major changes
3. Log retrieval scores for debugging
4. Measure effect of each tweak using fixed evaluation questions
