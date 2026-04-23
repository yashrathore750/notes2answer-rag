from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from .chat_memory import ChatMemory
from .prompting import SYSTEM_PROMPT, build_context_block, build_user_prompt
from .retriever import Retriever


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    sources: list[str]


class RAGEngine:
    def __init__(
        self,
        api_key: str,
        chat_model: str,
        retriever: Retriever,
    ) -> None:
        """Create the main RAG pipeline with one chat model and one retriever."""
        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model
        self.retriever = retriever

    def ask(self, question: str, top_k: int, memory: ChatMemory | None = None) -> RAGAnswer:
        """Retrieve relevant chunks, build the prompt, call the LLM, and return sources."""
        results = self.retriever.retrieve(question, top_k=top_k)
        context_block = build_context_block(results)
        user_prompt = build_user_prompt(question, context_block)

        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if memory is not None:
            messages.extend(memory.as_chat_messages())
        messages.append({"role": "user", "content": user_prompt})

        completion = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=0.2,
        )
        answer_text = completion.choices[0].message.content or "No answer returned."

        unique_sources = sorted({item.chunk.source for item in results})

        if memory is not None:
            memory.add_user(question)
            memory.add_assistant(answer_text)

        return RAGAnswer(answer=answer_text, sources=unique_sources)
