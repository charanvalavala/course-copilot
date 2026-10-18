from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from typing import Any

import numpy as np

from course_copilot.config import DATA_DIR


@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATA_DIR / "course-copilot.sqlite3")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize():
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                url TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page INTEGER,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                text, content='chunks', content_rowid='id', tokenize='porter unicode61'
            );
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', old.id, old.text);
            END;
            """
        )


def add_document(name: str, url: str, content_hash: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    with connect() as db:
        existing = db.execute("SELECT id FROM documents WHERE content_hash = ?", (content_hash,)).fetchone()
        if existing:
            return {"id": int(existing["id"]), "chunks": 0, "duplicate": True}
        cursor = db.execute(
            "INSERT INTO documents(name, url, content_hash) VALUES (?, ?, ?)",
            (name, url, content_hash),
        )
        document_id = int(cursor.lastrowid)
        for chunk in chunks:
            embedding = chunk.get("embedding")
            db.execute(
                "INSERT INTO chunks(document_id, page, ordinal, text, embedding) VALUES (?, ?, ?, ?, ?)",
                (
                    document_id,
                    chunk.get("page"),
                    int(chunk["ordinal"]),
                    str(chunk["text"]),
                    json.dumps(np.asarray(embedding, dtype=np.float32).tolist()) if embedding is not None else None,
                ),
            )
        return {"id": document_id, "chunks": len(chunks), "duplicate": False}


def has_document_hash(content_hash: str) -> bool:
    with connect() as db:
        return db.execute("SELECT 1 FROM documents WHERE content_hash = ?", (content_hash,)).fetchone() is not None


def list_documents() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT d.id, d.name, d.url, d.created_at, COUNT(c.id) AS chunks "
            "FROM documents d LEFT JOIN chunks c ON c.document_id=d.id "
            "GROUP BY d.id ORDER BY d.created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def remove_document(document_id: int) -> bool:
    with connect() as db:
        db.execute("PRAGMA foreign_keys = ON")
        cursor = db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0


def chunks() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT c.id, c.document_id, c.page, c.ordinal, c.text, c.embedding, d.name, d.url "
            "FROM chunks c JOIN documents d ON d.id=c.document_id ORDER BY c.id"
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["embedding"] = np.asarray(json.loads(item["embedding"]), dtype=np.float32) if item["embedding"] else None
            item["source"] = item.pop("name")
            result.append(item)
        return result


def lexical_search(query: str, limit: int) -> list[int]:
    stop_words = {"the", "and", "for", "from", "with", "that", "this", "what", "when", "where", "which", "who", "does", "did", "how", "can", "are", "was", "were", "is", "in", "on", "of", "to", "a", "an", "it", "my", "me", "i"}
    words = list(dict.fromkeys(token for token in re.findall(r"[a-z0-9]+", query.casefold()) if token not in stop_words and len(token) > 2))
    if not words:
        return []
    expression = " OR ".join('"' + word.replace('"', '') + '"' for word in words[:32])
    with connect() as db:
        rows = db.execute(
            "SELECT rowid, text FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts) LIMIT ?",
            (expression, max(limit * 3, 30)),
        ).fetchall()
        normalized = {word.rstrip("s") for word in words}
        minimum_overlap = min(2, len(normalized))
        matches = []
        for row in rows:
            document_terms = {token.rstrip("s") for token in re.findall(r"[a-z0-9]+", row["text"].casefold())}
            if len(normalized & document_terms) >= minimum_overlap:
                matches.append(int(row["rowid"]))
            if len(matches) == limit:
                break
        return matches
