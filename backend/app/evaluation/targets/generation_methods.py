"""Adapters for generation methods used by generation-stage evaluation."""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from app.generation.llm import zai_llm
from app.evaluation.models import GenerationOutput


class GenerationMethodTarget:
    """Call one named generation method through a stable evaluation interface."""

    def __init__(self, method: str = "default"):
        self.method = method

    async def generate(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
    ) -> GenerationOutput:
        if self.method != "default":
            raise ValueError(f"Unsupported generation method: {self.method}")

        started = time.time()
        answer = await zai_llm.generate_with_sources(
            question=question,
            chunks=context_chunks,
        )
        latency_ms = round((time.time() - started) * 1000, 2)
        return GenerationOutput(
            answer=answer.strip(),
            citations=_citations_from_answer(answer, context_chunks),
            latency_ms=latency_ms,
        )


def _citations_from_answer(answer: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map bracket citations like [1] back to context chunks."""
    citations: List[Dict[str, Any]] = []
    seen_indexes: set[int] = set()
    for match in re.finditer(r"\[(\d+)\]", answer):
        index = int(match.group(1))
        if index in seen_indexes or index < 1 or index > len(context_chunks):
            continue
        seen_indexes.add(index)
        chunk = context_chunks[index - 1]
        metadata = chunk.get("metadata") or {}
        citations.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "document_id": metadata.get("document_id") or chunk.get("document_id"),
            "document_name": metadata.get("source") or chunk.get("document_name") or chunk.get("source"),
            "page_number": metadata.get("page_number", chunk.get("page_number")),
            "chunk_index": metadata.get("chunk_index", chunk.get("chunk_index")),
            "content": chunk.get("content", ""),
        })
    return citations
