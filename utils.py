"""
Utility functions for StudyMate AI.

Handles:
- Document loading (PDF, TXT, MD)
- Text cleaning
- File validation

Credits:
- PDF text extraction uses pypdf (Apache 2.0)
- Markdown handling is native Python
"""

from __future__ import annotations

import io
import logging
import re
from typing import Union

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 25
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


def validate_file(uploaded_file) -> bool:
    """Check that uploaded file is an allowed type and size."""
    if uploaded_file is None:
        return False
    name = getattr(uploaded_file, "name", "")
    if not name:
        return False
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False
    size = getattr(uploaded_file, "size", 0)
    if size and size > MAX_FILE_SIZE_MB * 1024 * 1024:
        logger.warning(f"File {name} exceeds {MAX_FILE_SIZE_MB}MB limit")
        return False
    return True


def load_document(uploaded_file) -> str:
    """Load a document into raw text."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return load_pdf(uploaded_file)
    if name.endswith(".txt") or name.endswith(".md"):
        return load_text(uploaded_file)
    raise ValueError(f"Unsupported file type: {name}")


def load_pdf(uploaded_file) -> str:
    """Extract text from PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf not installed. Run: pip install pypdf")

    data = uploaded_file.read()
    reader = PdfReader(io.BytesIO(data))
    text_parts = []
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        except Exception as e:
            logger.warning(f"Failed to extract page {i}: {e}")
    return clean_text("\n".join(text_parts))


def load_text(uploaded_file) -> str:
    """Load a plain-text or markdown file."""
    data = uploaded_file.read()
    if isinstance(data, bytes):
        # Try multiple encodings
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return clean_text(data.decode(enc))
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode file — unknown encoding")
    return clean_text(str(data))


def clean_text(text: str) -> str:
    """Basic text cleaning: collapse whitespace, strip control chars."""
    if not text:
        return ""
    # Remove non-printable characters except newlines and tabs
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
    # Collapse runs of >2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list:
    """Simple wrapper so the chunker is importable from a single namespace."""
    from .rag_engine import TextSplitter
    return TextSplitter(chunk_size, overlap).split(text)
