"""Shared evaluation result models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.evaluation.metrics.retrieval import HitRateResult, MRRResult


@dataclass
class RetrievalCaseResult:
    """Case-level result from a retrieval-stage evaluation."""

    test_case_id: str
    question: str
    expected_chunk_ids: List[str]
    retrieved_chunk_ids: List[str]
    retrieved_contents: List[str]
    method: str
    response_time: float
    hit: bool
    rank: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "question": self.question,
            "expected_chunk_ids": self.expected_chunk_ids,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrieved_contents": self.retrieved_contents,
            "method": self.method,
            "response_time": self.response_time,
            "hit": self.hit,
            "rank": self.rank,
            "error": self.error,
        }


@dataclass
class MethodResult:
    """Result bundle for one retrieval method."""

    method_name: str
    results: List[Dict[str, Any]]
    hit_rate: HitRateResult
    mrr: MRRResult
    avg_response_time: float
    errors: List[str] = field(default_factory=list)


@dataclass
class GenerationInput:
    """Fixed-context input for generation-stage evaluation."""

    case_id: str
    question: str
    context_chunks: List[Dict[str, Any]]
    expected_keywords: List[str] = field(default_factory=list)
    expected_source: Optional[str] = None


@dataclass
class GenerationOutput:
    """Output produced by one generation target."""

    answer: str
    citations: List[Dict[str, Any]]
    latency_ms: float


@dataclass
class GenerationMethodResult:
    """Result bundle for one generation method."""

    method_name: str
    results: List[Dict[str, Any]]
    pass_rate: float
    avg_latency_ms: float
    errors: List[str] = field(default_factory=list)
