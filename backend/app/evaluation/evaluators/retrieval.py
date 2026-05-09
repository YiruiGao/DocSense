"""Retrieval comparison evaluator."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Protocol

from app.evaluation.metrics.retrieval import calculate_hit_rate, calculate_mrr
from app.evaluation.models import MethodResult, RetrievalCaseResult, TestCase, TestCaseSet
from app.evaluation.targets.retrieval_methods import RetrievalMethodTarget


class RetrievalTarget(Protocol):
    """Protocol for objects that can retrieve chunks for a query."""

    method: str

    def retrieve(
        self,
        query: str,
        top_k: int,
        document_id: Optional[str],
        namespace: Optional[str] = None,
        corpus_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return retrieved chunks for the query."""


class RetrievalComparisonEvaluator:
    """Evaluate and compare retrieval methods over one test set."""

    async def evaluate(
        self,
        methods: List[str],
        test_set: Optional[TestCaseSet] = None,
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        corpus_id: Optional[str] = None,
        top_k: int = 10,
        targets: Optional[Dict[str, RetrievalTarget]] = None,
    ) -> Dict[str, MethodResult]:
        selected_test_set = test_set or TestCaseSet(
            id="empty",
            name="Empty evaluation set",
            test_cases=[],
        )
        targets = targets or {
            method: RetrievalMethodTarget(method)
            for method in methods
        }
        results: Dict[str, MethodResult] = {}
        for method in methods:
            results[method] = await self._evaluate_target(
                target=targets[method],
                test_cases=selected_test_set.test_cases,
                document_id=document_id,
                namespace=namespace,
                corpus_id=corpus_id,
                top_k=top_k,
            )
        return results

    async def _evaluate_target(
        self,
        target: RetrievalTarget,
        test_cases: List[TestCase],
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        corpus_id: Optional[str] = None,
        top_k: int = 10,
    ) -> MethodResult:
        results: List[Dict[str, Any]] = []
        errors: List[str] = []

        for test_case in test_cases:
            start_time = time.time()
            try:
                retrieved = target.retrieve(
                    query=test_case.question,
                    top_k=top_k,
                    document_id=document_id or test_case.document_id,
                    namespace=namespace,
                    corpus_id=corpus_id,
                )
                response_time = time.time() - start_time
                hit_rank = _first_hit_rank(retrieved, test_case)
                case_result = RetrievalCaseResult(
                    test_case_id=test_case.id,
                    question=test_case.question,
                    expected_chunk_ids=test_case.expected_chunks,
                    retrieved_chunk_ids=[result.get("chunk_id", "") for result in retrieved],
                    retrieved_contents=[result.get("content", "") for result in retrieved],
                    method=target.method,
                    response_time=response_time,
                    hit=hit_rank is not None,
                    rank=hit_rank,
                )
                results.append(case_result.to_dict())
            except Exception as exc:
                response_time = time.time() - start_time
                errors.append(f"Test case {test_case.id} error: {exc}")
                case_result = RetrievalCaseResult(
                    test_case_id=test_case.id,
                    question=test_case.question,
                    expected_chunk_ids=test_case.expected_chunks,
                    retrieved_chunk_ids=[],
                    retrieved_contents=[],
                    method=target.method,
                    response_time=response_time,
                    hit=False,
                    error=str(exc),
                )
                results.append(case_result.to_dict())

        avg_time = (
            sum(result["response_time"] for result in results) / len(results)
            if results
            else 0.0
        )
        return MethodResult(
            method_name=target.method,
            results=results,
            hit_rate=calculate_hit_rate(results, k_values=[3, 5, 10]),
            mrr=calculate_mrr(results),
            avg_response_time=avg_time,
            errors=errors,
        )


def _matches_expected(retrieved: Dict[str, Any], test_case: TestCase) -> bool:
    chunk_id = retrieved.get("chunk_id", "")
    if chunk_id and chunk_id in set(test_case.expected_chunks):
        return True

    content = retrieved.get("content", "").lower()
    keyword_hits = [
        keyword
        for keyword in test_case.expected_chunks
        if keyword and keyword.lower() in content
    ]
    if keyword_hits:
        return True

    expected_pages = set(test_case.expected_page_numbers or [])
    page_number = retrieved.get("page_number")
    metadata = retrieved.get("metadata") or {}
    if page_number is None:
        page_number = metadata.get("page_number")
    return bool(expected_pages and page_number in expected_pages)


def _first_hit_rank(retrieved: List[Dict[str, Any]], test_case: TestCase) -> Optional[int]:
    for index, result in enumerate(retrieved, start=1):
        if _matches_expected(result, test_case):
            return index
    return None
