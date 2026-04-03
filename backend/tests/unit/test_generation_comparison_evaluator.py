import asyncio

import pytest

from app.evaluation.models import GenerationInput, GenerationOutput
from app.evaluation.evaluators.generation import GenerationComparisonEvaluator


class StubGenerationTarget:
    def __init__(self, method, answer, citations):
        self.method = method
        self._answer = answer
        self._citations = citations

    async def generate(self, question, context_chunks):
        return GenerationOutput(
            answer=self._answer,
            citations=self._citations,
            latency_ms=12.5,
        )


@pytest.mark.unit
def test_generation_comparison_evaluator_evaluates_answer_and_citation():
    case = GenerationInput(
        case_id="refund_policy",
        question="What is ACME's refund policy?",
        context_chunks=[
            {
                "chunk_id": "acme_chunk_1",
                "content": "Refunds are available within 30 days with an order number.",
                "metadata": {"source": "acme_support_guide.md", "page_number": 1},
            }
        ],
        expected_keywords=["refund", "30 days", "order number"],
        expected_source="acme_support_guide.md",
    )
    targets = {
        "baseline_prompt": StubGenerationTarget(
            method="baseline_prompt",
            answer="Customers can request a refund within 30 days with an order number. [1]",
            citations=[
                {
                    "chunk_id": "acme_chunk_1",
                    "document_name": "acme_support_guide.md",
                    "page_number": 1,
                }
            ],
        )
    }

    results = asyncio.run(
        GenerationComparisonEvaluator().evaluate(
            methods=["baseline_prompt"],
            cases=[case],
            targets=targets,
        )
    )

    method_result = results["baseline_prompt"]
    assert method_result.pass_rate == 1
    assert method_result.avg_latency_ms == 12.5
    assert method_result.results[0]["passed"] is True
    assert method_result.results[0]["answer_evaluation"]["matched_keywords"] == [
        "refund",
        "30 days",
        "order number",
    ]
    assert method_result.results[0]["citation_evaluation"]["matched_sources"] == [
        "acme_support_guide.md"
    ]


@pytest.mark.unit
def test_generation_comparison_evaluator_reports_stage_failures():
    case = GenerationInput(
        case_id="refund_policy",
        question="What is ACME's refund policy?",
        context_chunks=[
            {
                "chunk_id": "acme_chunk_1",
                "content": "Refunds are available within 30 days.",
                "metadata": {"source": "acme_support_guide.md"},
            }
        ],
        expected_keywords=["refund", "30 days"],
        expected_source="acme_support_guide.md",
    )
    targets = {
        "bad_prompt": StubGenerationTarget(
            method="bad_prompt",
            answer="Customers can request support through the portal.",
            citations=[
                {
                    "chunk_id": "wrong_chunk",
                    "document_name": "pricing_guide.md",
                }
            ],
        )
    }

    results = asyncio.run(
        GenerationComparisonEvaluator().evaluate(
            methods=["bad_prompt"],
            cases=[case],
            targets=targets,
        )
    )

    case_result = results["bad_prompt"].results[0]
    assert results["bad_prompt"].pass_rate == 0
    assert case_result["passed"] is False
    assert "missing keywords: refund, 30 days" in case_result["failure_reasons"]
    assert (
        "expected source acme_support_guide.md, got pricing_guide.md"
        in case_result["failure_reasons"]
    )
