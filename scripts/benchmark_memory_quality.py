#!/usr/bin/env python3
"""Measure conversation-memory retrieval quality against a curated corpus.

The benchmark uses Kestrel's real embedding query contract and weighted
``MemoryRetriever`` scorer.  It reports retrieval quality and abstention
separately: a system that returns ten plausible memories for an unknown fact
must not receive the same score as one that correctly returns nothing.
It intentionally exercises the component scorer on a fixed corpus rather than
the full storage-backed ``retrieve()`` path; echo suppression and candidate
generation have separate integration/unit coverage.

Usage:
    uv run python scripts/benchmark_memory_quality.py \
        --model qwen3-embedding:8b
    uv run python scripts/benchmark_memory_quality.py \
        --model qwen3-embedding:0.6b --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from kestrel_sovereign.llm.embedding_service import EmbeddingService
from kestrel_sovereign.storage.memory_models import MemoryMetadata
from kestrel_sovereign.storage.memory_retriever import MemoryRetriever


DEFAULT_SUITE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "memory_quality_eval.json"
)


@dataclass(frozen=True)
class QualityMetrics:
    answerable_queries: int
    abstention_queries: int
    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float
    top1_accuracy: float
    abstention_accuracy: float
    forbidden_hit_rate: float
    mean_returned: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b))
    denominator = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return numerator / denominator if denominator else 0.0


def _metadata(memory: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "importance",
        "emotional_valence",
        "emotional_confidence",
        "access_count",
        "claim_certainty",
        "operator_signal",
        "superseded_by",
    )
    return {field: memory[field] for field in fields if field in memory}


def rank_query(
    *,
    retriever: MemoryRetriever,
    memories: Sequence[Mapping[str, Any]],
    query: Mapping[str, Any],
    similarities: Mapping[str, float],
    cosine_floor: float,
    min_score: float,
    min_relevance: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Run the production component scorer over one labeled query."""
    emotional_context = None
    if "mood" in query:
        emotional_context = MemoryMetadata(emotional_valence=float(query["mood"]))
    now = datetime.now(timezone.utc)
    ranked: list[dict[str, Any]] = []
    for memory in memories:
        metadata = _metadata(memory)
        if metadata.get("operator_signal") or metadata.get("superseded_by"):
            continue
        components = retriever._calculate_score_components(
            content=str(memory["text"]),
            query=str(query["query"]),
            metadata=metadata,
            emotional_context=emotional_context,
            created_at=(
                now - timedelta(days=float(memory.get("days_old", 0)))
            ).isoformat(),
            expanded_concepts=[],
            semantic_similarity=similarities[str(memory["id"])],
            vector_cosine_floor=cosine_floor,
        )
        if (
            components["semantic"] >= min_relevance
            and components["total"] >= min_score
        ):
            ranked.append(
                {
                    "id": str(memory["id"]),
                    "total": components["total"],
                    "semantic": components["semantic"],
                    "raw_cosine": similarities[str(memory["id"])],
                }
            )
    ranked.sort(key=lambda row: (row["total"], row["semantic"]), reverse=True)
    if ranked:
        strongest = max(row["semantic"] for row in ranked)
        ranked = [
            row for row in ranked
            if row["semantic"]
            >= strongest * MemoryRetriever.RELATIVE_RELEVANCE_RATIO
        ]
    return ranked[:limit]


def summarize(
    query_results: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    *,
    limit: int,
) -> QualityMetrics:
    answerable = [(q, rows) for q, rows in query_results if q.get("relevant")]
    abstentions = [(q, rows) for q, rows in query_results if q.get("expect_empty")]
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    top1: list[float] = []
    forbidden_hits = 0
    forbidden_total = 0
    for query, rows in answerable:
        returned = [str(row["id"]) for row in rows]
        relevant = {str(value) for value in query["relevant"]}
        hits = relevant.intersection(returned)
        recalls.append(len(hits) / len(relevant))
        precisions.append(len(hits) / max(1, len(returned)))
        rank = next((i + 1 for i, value in enumerate(returned) if value in relevant), 0)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        top1.append(float(bool(returned and returned[0] in relevant)))
        forbidden = {str(value) for value in query.get("forbidden", [])}
        forbidden_hits += len(forbidden.intersection(returned))
        forbidden_total += len(forbidden)
    return QualityMetrics(
        answerable_queries=len(answerable),
        abstention_queries=len(abstentions),
        recall_at_k=mean(recalls) if recalls else 1.0,
        precision_at_k=mean(precisions) if precisions else 1.0,
        mean_reciprocal_rank=mean(reciprocal_ranks) if reciprocal_ranks else 1.0,
        top1_accuracy=mean(top1) if top1 else 1.0,
        abstention_accuracy=(
            mean(float(not rows) for _query, rows in abstentions)
            if abstentions else 1.0
        ),
        forbidden_hit_rate=(
            forbidden_hits / forbidden_total if forbidden_total else 0.0
        ),
        mean_returned=mean(len(rows) for _query, rows in query_results),
    )


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    memories = suite["memories"]
    queries = suite["queries"]
    service = EmbeddingService(model=args.model, base_url=args.ollama_url)
    document_vectors = await service.aembed_batch(
        [str(memory["text"]) for memory in memories]
    )
    if len(document_vectors) != len(memories) or any(
        vector is None for vector in document_vectors
    ):
        raise RuntimeError("embedding model did not return every document vector")
    retriever = MemoryRetriever(None)
    floor = (
        service.retrieval_similarity_floor()
        if args.cosine_floor is None else float(args.cosine_floor)
    )
    results = []
    details = []
    for query in queries:
        query_vector = await service.aembed_query(str(query["query"]))
        if query_vector is None:
            raise RuntimeError(f"query embedding failed: {query['id']}")
        similarities = {
            str(memory["id"]): cosine(query_vector, vector)
            for memory, vector in zip(memories, document_vectors)
        }
        ranked = rank_query(
            retriever=retriever,
            memories=memories,
            query=query,
            similarities=similarities,
            cosine_floor=floor,
            min_score=args.min_score,
            min_relevance=args.min_relevance,
            limit=args.limit,
        )
        results.append((query, ranked))
        details.append(
            {
                "id": query["id"],
                "query": query["query"],
                "relevant": query.get("relevant", []),
                "forbidden": query.get("forbidden", []),
                "returned": ranked,
            }
        )
    metrics = summarize(results, limit=args.limit)
    return {
        "suite_version": suite["version"],
        "model": args.model,
        "cosine_floor": floor,
        "min_score": args.min_score,
        "min_relevance": args.min_relevance,
        "limit": args.limit,
        "metrics": metrics.as_dict(),
        "queries": details,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model", default="qwen3-embedding:8b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--cosine-floor", type=float)
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--min-relevance", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = asyncio.run(evaluate(args))
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"model={report['model']} floor={report['cosine_floor']:.3f} "
            f"min_relevance={report['min_relevance']:.3f}"
        )
        for name, value in report["metrics"].items():
            print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
        print("\nPer-query top results:")
        for query in report["queries"]:
            top = ", ".join(
                f"{row['id']}({row['raw_cosine']:.3f}/{row['total']:.3f})"
                for row in query["returned"]
            ) or "<empty>"
            print(f"- {query['id']}: {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
