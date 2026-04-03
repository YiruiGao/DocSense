"""Generation comparison evaluator for fixed retrieval contexts."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Protocol

from app.evaluation.metrics.matching import keyword_match, source_match
from app.evaluation.models import GenerationInput, GenerationMethodResult
from app.evaluation.targets.generation_methods import GenerationMethodTarget


class GenerationTarget(Protocol):
    """Protocol for generation targets evaluated with fixed context."""

    method: str

    async def generate(self, question: str, context_chunks: List[dict]) -> object:
        """Generate an answer and citations from fixed context."""


class GenerationComparisonEvaluator:
    """Evaluate and compare generation methods on fixed retrieval contexts."""

    async def evaluate(
        self,
        methods: List[str],
        cases: List[GenerationInput],
        targets: Optional[Dict[str, GenerationTarget]] = None,
    ) -> Dict[str, GenerationMethodResult]:
        targets = targets or {
            method: GenerationMethodTarget(method)
            for method in methods
        }
        return {
            method: await self._evaluate_method(method, targets[method], cases)
            for method in methods
        }

    async def _evaluate_method(
        self,
        method: str,
        target: GenerationTarget,
        cases: List[GenerationInput],
    ) -> GenerationMethodResult:
        results: List[dict] = []
        errors: List[str] = []

        for case in cases:
            try:
                output = await target.generate(case.question, case.context_chunks)
                answer_evaluation = _evaluate_answer(
                    answer=output.answer,
                    expected_keywords=case.expected_keywords,
                )
                citation_evaluation = _evaluate_citations(
                    citations=output.citations,
                    expected_source=case.expected_source,
                )
                failure_reasons = [
                    *answer_evaluation["failure_reasons"],
                    *citation_evaluation["failure_reasons"],
                ]
                results.append({
                    "case_id": case.case_id,
                    "question": case.question,
                    "method": method,
                    "answer": output.answer,
                    "citations": output.citations,
                    "latency_ms": output.latency_ms,
                    "answer_evaluation": answer_evaluation,
                    "citation_evaluation": citation_evaluation,
                    "passed": not failure_reasons,
                    "failure_reasons": failure_reasons,
                })
            except Exception as exc:
                errors.append(f"Case {case.case_id} error: {exc}")
                results.append({
                    "case_id": case.case_id,
                    "question": case.question,
                    "method": method,
                    "answer": "",
                    "citations": [],
                    "latency_ms": 0.0,
                    "answer_evaluation": {},
                    "citation_evaluation": {},
                    "passed": False,
                    "failure_reasons": [str(exc)],
                })

        passed_count = sum(1 for result in results if result["passed"])
        return GenerationMethodResult(
            method_name=method,
            results=results,
            pass_rate=round(passed_count / len(results), 4) if results else 0.0,
            avg_latency_ms=_avg_latency(result["latency_ms"] for result in results),
            errors=errors,
        )


def _evaluate_answer(answer: str, expected_keywords: Iterable[str]) -> dict:
    keyword_result = keyword_match(answer, expected_keywords)
    failure_reasons: list[str] = []
    answer_present = bool(answer.strip())
    if not answer_present:
        failure_reasons.append("answer is empty")
    if keyword_result.missing_items:
        failure_reasons.append(f"missing keywords: {', '.join(keyword_result.missing_items)}")
    return {
        "passed": not failure_reasons,
        "answer_present": answer_present,
        "matched_keywords": keyword_result.matched_items,
        "missing_keywords": keyword_result.missing_items,
        "failure_reasons": failure_reasons,
    }


def _evaluate_citations(citations: Iterable[dict], expected_source: Optional[str]) -> dict:
    sources = _citation_sources(citations)
    match_result = source_match(sources, expected_source)
    failure_reasons: list[str] = []
    if not match_result.matched:
        failure_reasons.append(
            f"expected source {expected_source}, got {', '.join(sources) or 'none'}"
        )
    return {
        "passed": not failure_reasons,
        "expected_source": expected_source,
        "matched_sources": match_result.matched_items,
        "missing_sources": match_result.missing_items,
        "failure_reasons": failure_reasons,
    }


def _citation_sources(citations: Iterable[dict]) -> List[str]:
    return [
        citation.get("document_name") or citation.get("source") or ""
        for citation in citations
    ]


def _avg_latency(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 2) if items else 0.0
