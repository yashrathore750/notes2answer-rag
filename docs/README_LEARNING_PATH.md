# Learning Path for This Project

Use this order to learn RAG deeply while staying practical.

## Stage 1: Understand the Full Pipeline

Read first:

- README.md
- docs/RAG_THEORY.md (Sections 1 to 4)

Goal:

- Understand why RAG exists
- Understand how retrieval changes LLM behavior

## Stage 2: Run the App End-to-End

1. Add 2-3 small documents in data/
2. Run: python app.py ingest
3. Run: python app.py ask "Summarize my notes"
4. Run: python app.py chat

Goal:

- Experience full data-to-answer flow

## Stage 3: Read the Code in This Sequence

1. src/document_loader.py
2. src/chunker.py
3. src/embeddings.py
4. src/vector_store.py
5. src/retriever.py
6. src/prompting.py
7. src/rag_engine.py
8. app.py

Goal:

- Build an internal map of responsibilities

## Stage 4: Try Controlled Experiments

Change one variable at a time:

1. CHUNK_SIZE: 300, 700, 1200
2. CHUNK_OVERLAP: 40, 120, 250
3. TOP_K: 2, 4, 8
4. Ask broad vs specific questions

Observe:

- Answer quality
- Source relevance
- Hallucination behavior

## Stage 5: Build Intuition

For each bad answer, ask:

1. Retrieval issue: Did we fetch wrong chunks?
2. Chunking issue: Was context split badly?
3. Prompt issue: Did instructions allow guessing?
4. Data issue: Is answer actually in notes?

That diagnosis loop is core RAG skill.

## Stage 6: Next Upgrades

1. Add reranking after retrieval
2. Add hybrid search (keyword + vector)
3. Add document metadata filters
4. Add evaluation set and scoring
5. Add streaming responses
