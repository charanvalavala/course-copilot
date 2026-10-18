from __future__ import annotations

from typing import Any

import numpy as np

from course_copilot.database import chunks, lexical_search
from course_copilot.config import MIN_DENSE_SIMILARITY
from course_copilot.embeddings import embed


def retrieve(question: str, top_k: int = 5) -> list[dict[str, Any]]:
    items = chunks()
    if not items:
        return []
    query_vector = embed([question])[0]
    dense = sorted(
        ((item, float(np.dot(query_vector, item["embedding"]))) for item in items if item["embedding"] is not None and item["embedding"].shape == query_vector.shape),
        key=lambda pair: pair[1],
        reverse=True,
    )[: max(20, top_k * 4)]
    dense = [(item, score) for item, score in dense if score >= MIN_DENSE_SIMILARITY]
    lexical_ids = lexical_search(question, max(20, top_k * 4))
    ranks: dict[int, float] = {}
    dense_scores: dict[int, float] = {}
    for rank, (item, score) in enumerate(dense, start=1):
        chunk_id = int(item["id"])
        ranks[chunk_id] = ranks.get(chunk_id, 0.0) + 1 / (60 + rank)
        dense_scores[chunk_id] = score
    for rank, chunk_id in enumerate(lexical_ids, start=1):
        ranks[chunk_id] = ranks.get(chunk_id, 0.0) + 1 / (60 + rank)
    by_id = {int(item["id"]): item for item in items}
    ordered_ids = sorted(ranks, key=lambda chunk_id: ranks[chunk_id], reverse=True)
    return [
        {
            "chunk_id": chunk_id,
            "source": by_id[chunk_id]["source"],
            "url": by_id[chunk_id]["url"],
            "page": by_id[chunk_id]["page"],
            "text": by_id[chunk_id]["text"],
            "dense_score": dense_scores.get(chunk_id),
            "rrf_score": ranks[chunk_id],
        }
        for chunk_id in ordered_ids[:top_k]
    ]
