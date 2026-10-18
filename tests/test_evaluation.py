from course_copilot.evaluation import evaluate


def test_evaluate_tracks_source_ranks_and_abstention():
    cases = [
        {"question": "first", "expected_sources": ["a.md"]},
        {"question": "second", "expected_sources": ["b.md"]},
        {"question": "outside", "expected_sources": [], "should_abstain": True},
    ]

    def search(question, top_k):
        results = {
            "first": [{"source": "A.md"}, {"source": "b.md"}],
            "second": [{"source": "c.md"}, {"source": "B.md"}],
            "outside": [],
        }
        return results[question][:top_k]

    report = evaluate(cases, search, 3)

    assert report["queries"] == 3
    assert report["answerable_queries"] == 2
    assert report["unanswerable_queries"] == 1
    assert report["hit_rate_at_1"] == 0.5
    assert report["hit_rate_at_k"] == 1.0
    assert report["mean_reciprocal_rank"] == 0.75
    assert report["unanswerable_abstention_accuracy"] == 1.0
    assert "does not score answer correctness" in report["scope"]


def test_evaluate_rejects_empty_case_sets():
    try:
        evaluate([], lambda question, top_k: [], 5)
    except ValueError as error:
        assert "At least one" in str(error)
    else:
        raise AssertionError("An empty evaluation set should be rejected.")
