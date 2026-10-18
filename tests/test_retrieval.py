import numpy as np

from course_copilot import retrieval


def test_weighted_rrf_gives_lexical_rank_a_two_to_one_vote(monkeypatch):
    items = [
        {"id": 1, "source": "dense-first.md", "url": "", "page": 1, "text": "first", "embedding": np.array([1.0, 0.0])},
        {"id": 2, "source": "bm25-first.md", "url": "", "page": 1, "text": "second", "embedding": np.array([0.9, 0.1])},
    ]
    monkeypatch.setattr(retrieval, "chunks", lambda: items)
    monkeypatch.setattr(retrieval, "embed", lambda values: [np.array([1.0, 0.0])])
    monkeypatch.setattr(retrieval, "lexical_search", lambda question, limit: [2, 1])

    result = retrieval.retrieve("example query", 1)

    assert result[0]["source"] == "bm25-first.md"
