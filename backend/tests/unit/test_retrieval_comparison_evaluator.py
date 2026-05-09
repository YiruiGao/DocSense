import asyncio

import pytest

import app.evaluation.evaluators.retrieval as retrieval
from app.evaluation.evaluators.retrieval import RetrievalComparisonEvaluator
from app.evaluation.models import TestCase as EvalTestCase
from app.evaluation.models import TestCaseSet as EvalTestCaseSet


class StubRetrievalTarget:
    def __init__(self, method):
        self.method = method

    def retrieve(self, query, top_k, document_id, namespace=None, corpus_id=None):
        return [
            {
                "chunk_id": f"{self.method}-chunk",
                "content": f"{self.method} answer content",
                "metadata": {"page_number": 1},
            }
        ]


@pytest.mark.unit
def test_retrieval_comparison_evaluator_scores_one_target():
    test_set = EvalTestCaseSet(
        id="sample",
        name="Sample",
        test_cases=[
            EvalTestCase(
                id="case-1",
                question="Question?",
                expected_chunks=["baseline-chunk"],
                expected_page_numbers=[],
            )
        ],
    )

    results = asyncio.run(
        RetrievalComparisonEvaluator().evaluate(
            methods=["baseline"],
            test_set=test_set,
            targets={"baseline": StubRetrievalTarget("baseline")},
        )
    )

    result = results["baseline"]
    assert result.method_name == "baseline"
    assert result.hit_rate.hit_at_3 == 1
    assert result.mrr.mrr == 1
    assert result.results[0]["hit"] is True
    assert result.results[0]["rank"] == 1


@pytest.mark.unit
def test_retrieval_comparison_evaluator_orchestrates_multiple_methods(monkeypatch):
    monkeypatch.setattr(retrieval, "RetrievalMethodTarget", StubRetrievalTarget)
    test_set = EvalTestCaseSet(
        id="sample",
        name="Sample",
        test_cases=[
            EvalTestCase(
                id="case-1",
                question="Question?",
                expected_chunks=["baseline-chunk"],
                expected_page_numbers=[],
            )
        ],
    )

    results = asyncio.run(
        RetrievalComparisonEvaluator().evaluate(
            methods=["baseline", "hybrid"],
            test_set=test_set,
        )
    )

    assert list(results) == ["baseline", "hybrid"]
    assert results["baseline"].results[0]["hit"] is True
    assert results["hybrid"].results[0]["hit"] is False
