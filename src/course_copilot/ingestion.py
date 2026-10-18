from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

from pypdf import PdfReader

from course_copilot.config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_UPLOAD_BYTES
from course_copilot.database import add_document
from course_copilot.embeddings import embed


def split_text(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + CHUNK_SIZE)
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + CHUNK_SIZE // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def parse_file(name: str, content: bytes) -> list[dict[str, Any]]:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("File exceeds the 8 MB upload limit.")
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if suffix == "pdf":
        pages = [(number, page.extract_text() or "") for number, page in enumerate(PdfReader(BytesIO(content)).pages, start=1)]
        if not any(text.strip() for _, text in pages):
            raise ValueError("PDF contains no extractable text. OCR scanned pages before upload.")
    elif suffix in {"txt", "md"}:
        pages = [(None, content.decode("utf-8", errors="replace"))]
    else:
        raise ValueError("Supported formats are PDF, TXT, and Markdown.")
    chunks = []
    for page, text in pages:
        for piece in split_text(text):
            chunks.append({"page": page, "ordinal": len(chunks), "text": piece})
    vectors = embed([chunk["text"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk["embedding"] = vector
    return chunks


def ingest(name: str, content: bytes, source_url: str = "") -> dict[str, Any]:
    digest = hashlib.sha256(content).hexdigest()
    return add_document(name, source_url, digest, parse_file(name, content))
