from __future__ import annotations

from typing import Any, Callable


def evaluate(cases: list[dict[str, Any]], search: Callable[[str, int], list[dict[str, Any]]], top_k: int) -> dict[str, Any]:
    if not cases:
        raise ValueError("At least one evaluation case is required.")
    hits = []
    hits_at_1 = []
    reciprocal_ranks = []
    details = []
    abstention_results = []
    for case in cases:
        relevant = {str(source).casefold() for source in case["expected_sources"]}
        results = search(str(case["question"]), top_k)
        ranks = [rank for rank, result in enumerate(results, start=1) if str(result["source"]).casefold() in relevant]
        hit = bool(ranks)
        reciprocal = 1 / min(ranks) if ranks else 0.0
        should_abstain = bool(case.get("should_abstain", False))
        if relevant:
            hits.append(hit)
            hits_at_1.append(bool(ranks and min(ranks) == 1))
            reciprocal_ranks.append(reciprocal)
        if should_abstain:
            abstention_results.append(not results)
        details.append({"question": case["question"], "hit": hit, "reciprocal_rank": reciprocal, "should_abstain": should_abstain, "abstained": not results, "retrieved": [item["source"] for item in results]})
    answerable = sum(bool(case["expected_sources"]) for case in cases)
    return {
        "queries": len(cases),
        "answerable_queries": answerable,
        "unanswerable_queries": len(cases) - answerable,
        "hit_rate_at_k": sum(hits) / len(hits) if hits else 0.0,
        "hit_rate_at_1": sum(hits_at_1) / len(hits_at_1) if hits_at_1 else 0.0,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "unanswerable_abstention_accuracy": sum(abstention_results) / len(abstention_results) if abstention_results else None,
        "top_k": top_k,
        "details": details,
        "scope": "File-level retrieval relevance only; does not score answer correctness or citation faithfulness.",
    }
