from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    embedding_model: str
    chat_model: str
    data_dir: Path
    storage_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int



def load_settings() -> Settings:
    """Load environment variables and convert them into typed runtime settings."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your environment or .env file.")

    return Settings(
        openai_api_key=api_key,
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        data_dir=Path(os.getenv("DATA_DIR", "./data")),
        storage_dir=Path(os.getenv("STORAGE_DIR", "./storage")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "700")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        top_k=int(os.getenv("TOP_K", "4")),
    )
