"""Adapters for DocSense retrieval methods used by evaluation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import reranker
from app.retrieval.vector_store import vector_store


class RetrievalMethodTarget:
    """Call one named retrieval method through a stable evaluation interface."""

    def __init__(self, method: str):
        self.method = method

    def retrieve(
        self,
        query: str,
        top_k: int,
        document_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        if self.method == "baseline":
            return vector_store.search(query=query, top_k=top_k, document_id=document_id)

        if self.method == "hybrid":
            return hybrid_search.search(query=query, top_k=top_k, document_id=document_id)

        if self.method == "hybrid_rerank":
            candidates = hybrid_search.search(
                query=query,
                top_k=top_k * 2,
                document_id=document_id,
            )
            return reranker.rerank(query=query, chunks=candidates, top_k=top_k)

        raise ValueError(f"Unsupported retrieval method: {self.method}")
