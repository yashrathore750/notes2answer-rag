# Chat with Your Notes (Beginner RAG Project)

This project helps you build your first Retrieval-Augmented Generation (RAG) app while learning how each part works.

## What You Will Build

- Upload your own notes (PDF, TXT, MD)
- Build a vector index from those notes
- Ask questions and get answers grounded in your data
- Chat with short conversation memory
- See source citations

## RAG Mental Model

User Question -> Embed -> Retrieve Similar Chunks -> Add Context to Prompt -> LLM Answer

In simple terms:

1. Convert both documents and questions into vectors (embeddings)
2. Find document chunks that are semantically closest to the question
3. Pass those chunks into the LLM prompt
4. Generate an answer based on retrieved evidence

## Project Structure

- app.py: CLI entry point (ingest, ask, chat)
- src/document_loader.py: Loads PDF/TXT/MD files
- src/chunker.py: Splits long docs into overlapping chunks
- src/embeddings.py: Calls OpenAI embedding API
- src/vector_store.py: Stores and searches vectors with FAISS
- src/retriever.py: Retrieval orchestration
- src/prompting.py: Prompt templates and context formatting
- src/rag_engine.py: End-to-end answer generation
- src/chat_memory.py: Chat history memory
- docs/: Deep explanations and learning guides

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Copy environment template:

   copy .env.example .env

4. Add your OpenAI API key in .env.

## Put Your Notes in Data Folder

- Add files into data/
- Supported extensions: .pdf, .txt, .md

## Run the App

### 1) Build the index

python app.py ingest

### 2) Ask one question

python app.py ask "What are the key ideas in my notes?"

### 3) Start chat mode

python app.py chat

Type exit to stop.

## Stretch Goals Included

- Chat history memory (short rolling context)
- Source citations (retrieved file paths)

## Important Learning Note

This app is intentionally minimal and educational. It is designed so you can understand each moving part before using larger frameworks.

Continue with the deep guides:

- docs/README_LEARNING_PATH.md
- docs/RAG_THEORY.md
- docs/CODE_WALKTHROUGH.md
