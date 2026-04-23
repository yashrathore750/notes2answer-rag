# RAG Theory (Beginner to Practical)

## 1) Why RAG Exists

A base LLM does not know your private notes unless you include them in the prompt. RAG solves this by retrieving relevant pieces of your data at question time.

Benefits:

- Grounded answers from your data
- Better freshness than static fine-tuning
- Easier updates: re-index docs instead of retraining model

## 2) Core Components

1. Embeddings
   Convert text into numeric vectors that capture semantic meaning.
2. Vector database
   Stores vectors and supports nearest-neighbor search.
3. Retriever
   Finds top-k chunks similar to the question.
4. Prompt augmentation
   Injects retrieved chunks into the model input.
5. Generator (LLM)
   Produces final answer constrained by context.

## 3) What Makes RAG Good or Bad

Main quality levers:

- Chunking strategy
- Embedding model quality
- Retrieval strategy (k value, filtering, reranking)
- Prompt constraints
- Data quality and coverage

## 4) Chunking Intuition

If chunks are too small:

- You lose context
- Retrieval may miss important details

If chunks are too large:

- Retrieval gets noisy
- Prompt cost increases

Overlap helps preserve continuity between adjacent chunks.

## 5) Retrieval Intuition

Similarity search returns chunks close in vector space. Good retrieval means the evidence used for answering is relevant.

Common failures:

- Relevant chunk not retrieved (recall failure)
- Retrieved chunk is related but not enough (precision failure)
- Lexical mismatch between query and docs

## 6) Prompt Augmentation

You should explicitly tell the model:

- Use only provided context
- Say when context is insufficient
- Cite sources

Without strict prompt instructions, the model may guess.

## 7) Why Memory Helps in Chat

In chat mode, users ask follow-up questions. Memory keeps recent turns so references like "that section" remain understandable.

But too much memory can inject noise. Keep a bounded history window.

## 8) Evaluation Mindset

Do not judge by one demo question.

Create a mini benchmark:

- 20 real questions
- Expected answers and sources
- Compare settings using the same questions

Track:

- Accuracy
- Citation quality
- Refusal behavior when info is missing

## 9) Next Concepts to Learn

- Hybrid retrieval (BM25 + embeddings)
- Reranking
- Query rewriting
- Multi-hop retrieval
- Agentic RAG
