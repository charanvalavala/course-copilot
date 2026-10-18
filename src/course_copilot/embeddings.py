from __future__ import annotations

from functools import lru_cache

import numpy as np

from course_copilot.config import MODEL_NAME, MODEL_REVISION


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)


def embed(texts: list[str]) -> list[np.ndarray]:
    if not texts:
        return []
    vectors = model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [np.asarray(vector, dtype=np.float32) for vector in vectors]
