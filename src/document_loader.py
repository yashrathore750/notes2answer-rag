from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class Document:
    source: str
    text: str



def _load_pdf(path: Path) -> str:
    """Extract and join text from every page in a PDF file."""
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()



def _load_text(path: Path) -> str:
    """Read a plain text or markdown file as UTF-8 text."""
    return path.read_text(encoding="utf-8", errors="ignore").strip()



def load_documents(data_dir: Path) -> list[Document]:
    """Load all supported files from the data directory into Document objects."""
    if not data_dir.exists():
        return []

    documents: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".pdf":
            content = _load_pdf(path)
        else:
            content = _load_text(path)

        if content:
            documents.append(Document(source=str(path), text=content))

    return documents
