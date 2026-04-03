"""Retrieval evaluation metrics."""
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class HitRateResult:
    """Hit rate at fixed cutoffs."""

    method_name: str
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    total_queries: int
    questions_with_hits: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "hit_at_10": self.hit_at_10,
            "total_queries": self.total_queries,
            "questions_with_hits": self.questions_with_hits,
        }


@dataclass
class MRRResult:
    """Mean Reciprocal Rank result."""

    method_name: str
    mrr: float
    total_queries: int
    reciprocal_ranks: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "mrr": self.mrr,
            "total_queries": self.total_queries,
            "reciprocal_ranks": self.reciprocal_ranks,
        }


def _method_name(results: List[Dict[str, Any]]) -> str:
    for result in results:
        if result.get("method"):
            return result["method"]
    return "unknown"


def _first_hit_rank(retrieved_ids: List[str], expected_ids: List[str]) -> int | None:
    expected = set(expected_ids)
    for index, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected:
            return index
    return None


def calculate_hit_rate(
    results: List[Dict[str, Any]],
    k_values: List[int] | None = None,
) -> HitRateResult:
    """Calculate Hit@K for retrieval results."""
    if k_values is None:
        k_values = [3, 5, 10]

    total_queries = len(results)
    hits_at_k = {k: 0 for k in k_values}
    questions_with_hits = 0

    for result in results:
        first_hit_rank = result.get("rank")
        if first_hit_rank is None:
            retrieved_ids = result.get("retrieved_chunk_ids", [])
            expected_ids = result.get("expected_chunk_ids", [])
            first_hit_rank = _first_hit_rank(retrieved_ids, expected_ids)

        if first_hit_rank is None:
            continue

        questions_with_hits += 1
        for k in k_values:
            if first_hit_rank <= k:
                hits_at_k[k] += 1

    def rate(k: int) -> float:
        if total_queries == 0:
            return 0.0
        return hits_at_k.get(k, 0) / total_queries

    return HitRateResult(
        method_name=_method_name(results),
        hit_at_3=rate(3),
        hit_at_5=rate(5),
        hit_at_10=rate(10),
        total_queries=total_queries,
        questions_with_hits=questions_with_hits,
    )


def calculate_mrr(results: List[Dict[str, Any]]) -> MRRResult:
    """Calculate Mean Reciprocal Rank for retrieval results."""
    reciprocal_ranks: List[float] = []

    for result in results:
        first_hit_rank = result.get("rank")
        if first_hit_rank is None:
            retrieved_ids = result.get("retrieved_chunk_ids", [])
            expected_ids = result.get("expected_chunk_ids", [])
            first_hit_rank = _first_hit_rank(retrieved_ids, expected_ids)
        reciprocal_ranks.append(0.0 if first_hit_rank is None else 1 / first_hit_rank)

    mrr = sum(reciprocal_ranks) / len(results) if results else 0.0

    return MRRResult(
        method_name=_method_name(results),
        mrr=mrr,
        total_queries=len(results),
        reciprocal_ranks=reciprocal_ranks,
    )
