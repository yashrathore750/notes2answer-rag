from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str
    content: str


class ChatMemory:
    def __init__(self, max_turns: int = 6) -> None:
        """Keep a short rolling window of recent chat turns."""
        self._messages: deque[Message] = deque(maxlen=max_turns * 2)

    def add_user(self, text: str) -> None:
        """Store the latest user message in memory."""
        self._messages.append(Message(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        """Store the latest assistant message in memory."""
        self._messages.append(Message(role="assistant", content=text))

    def as_chat_messages(self) -> list[dict[str, str]]:
        """Return memory in the chat API message format expected by OpenAI."""
        return [{"role": msg.role, "content": msg.content} for msg in self._messages]
