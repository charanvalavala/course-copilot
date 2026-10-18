from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))


def run(output: Path, top_k: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="course-copilot-ablation-") as data_dir:
        os.environ["COURSE_DATA_DIR"] = data_dir
        os.environ["COURSE_READ_ONLY"] = "false"

        from course_copilot import database
        from course_copilot.api import seed_demo_documents
        from course_copilot.config import BM25_RRF_WEIGHT, DENSE_RRF_WEIGHT, MIN_DENSE_SIMILARITY, MODEL_NAME, MODEL_REVISION
        from course_copilot.embeddings import embed
        from course_copilot.evaluation import evaluate
        from course_copilot.retrieval import retrieve

        database.initialize()
        seed_demo_documents()
        items = database.chunks()
        by_id = {int(item["id"]): item for item in items}
        cases = [
            json.loads(line)
            for line in (ROOT / "data" / "eval" / "manual-67.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        def bm25_only(question: str, count: int) -> list[dict[str, Any]]:
            ids = database.lexical_search(question, max(20, count * 4))
            return [by_id[chunk_id] for chunk_id in ids[:count] if chunk_id in by_id]

        def dense_only(question: str, count: int) -> list[dict[str, Any]]:
            query_vector = embed([question])[0]
            ranked = sorted(
                ((item, float(np.dot(query_vector, item["embedding"]))) for item in items if item["embedding"] is not None),
                key=lambda pair: pair[1],
                reverse=True,
            )
            return [item for item, score in ranked if score >= MIN_DENSE_SIMILARITY][:count]

        def weighted_rrf(bm25_weight: float, dense_weight: float, require_dense: bool = False) -> Callable[[str, int], list[dict[str, Any]]]:
            def search(question: str, count: int) -> list[dict[str, Any]]:
                limit = max(20, count * 4)
                query_vector = embed([question])[0]
                dense = sorted(
                    ((item, float(np.dot(query_vector, item["embedding"]))) for item in items if item["embedding"] is not None),
                    key=lambda pair: pair[1],
                    reverse=True,
                )[:limit]
                dense = [(item, score) for item, score in dense if score >= MIN_DENSE_SIMILARITY]
                if require_dense and not dense:
                    return []
                lexical = database.lexical_search(question, limit)
                scores: dict[int, float] = {}
                for rank, (item, _) in enumerate(dense, start=1):
                    chunk_id = int(item["id"])
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + dense_weight / (60 + rank)
                for rank, chunk_id in enumerate(lexical, start=1):
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + bm25_weight / (60 + rank)
                return [by_id[chunk_id] for chunk_id in sorted(scores, key=scores.get, reverse=True)[:count]]

            return search

        selected_name = f"Selected weighted RRF {BM25_RRF_WEIGHT:g}:{DENSE_RRF_WEIGHT:g} BM25:dense"
        retrievers: dict[str, Callable[[str, int], list[dict[str, Any]]]] = {
            "BM25 only": bm25_only,
            "Dense only": dense_only,
            "Equal-weight RRF": weighted_rrf(1.0, 1.0),
            selected_name: retrieve,
            "Weighted RRF 2:1 with dense gate": weighted_rrf(2.0, 1.0, require_dense=True),
            "Weighted RRF 3:1 BM25:dense": weighted_rrf(3.0, 1.0),
        }
        metrics: dict[str, Any] = {}
        for name, search in retrievers.items():
            metrics[name] = evaluate(cases, search, top_k)
        report = {
            "corpus_documents": len(database.list_documents()),
            "corpus_chunks": len(items),
            "queries": len(cases),
            "answerable_queries": sum(bool(case["expected_sources"]) for case in cases),
            "unanswerable_queries": sum(bool(case.get("should_abstain")) for case in cases),
            "top_k": top_k,
            "candidate_depth": max(20, top_k * 4),
            "rrf_rank_constant": 60,
            "bm25_rrf_weight": BM25_RRF_WEIGHT,
            "dense_rrf_weight": DENSE_RRF_WEIGHT,
            "minimum_dense_similarity": MIN_DENSE_SIMILARITY,
            "embedding_model": MODEL_NAME,
            "embedding_revision": MODEL_REVISION,
            "metrics": metrics,
            "scope": "Source-file retrieval relevance and empty-result abstention on the bundled manually authored query set. Not answer correctness or citation faithfulness.",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "retrieval-ablation.json")
    parser.add_argument("--top-k", type=int, default=5)
    arguments = parser.parse_args()
    report = run(arguments.output, arguments.top_k)
    for name, scores in report["metrics"].items():
        print(
            f"{name}: Hit@1={scores['hit_rate_at_1']:.3f} "
            f"Hit@{report['top_k']}={scores['hit_rate_at_k']:.3f} "
            f"MRR={scores['mean_reciprocal_rank']:.3f} "
            f"Abstention={scores['unanswerable_abstention_accuracy']:.3f}"
        )
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
