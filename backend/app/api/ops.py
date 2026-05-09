"""RAGOps diagnostic APIs."""
from statistics import mean
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from app.api.documents import _diagnose_chunks_auto, _documents_store
from app.evaluation import run_store
from app.ops import badcase as badcase_store
from app.ops import traces as trace_store
from app.retrieval.vector_store import vector_store

router = APIRouter(tags=["ops"])


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return round(ordered[index], 2)


def _summarize_traces() -> Dict[str, Any]:
    trace_items = trace_store.load_all_traces()
    latencies = [
        float(trace.get("total_latency_ms") or 0)
        for trace in trace_items
        if trace.get("total_latency_ms") is not None
    ]
    success_count = sum(1 for trace in trace_items if trace.get("status") == "success")
    error_count = sum(1 for trace in trace_items if trace.get("status") not in {"success", "running"})
    final_chunks = [
        int(trace.get("metadata", {}).get("final_chunks") or 0)
        for trace in trace_items
        if trace.get("metadata", {}).get("final_chunks") is not None
    ]
    total_candidates = [
        int(trace.get("metadata", {}).get("total_candidates") or 0)
        for trace in trace_items
        if trace.get("metadata", {}).get("total_candidates") is not None
    ]
    retrieval_seconds = []
    llm_seconds = []
    total_tokens = 0
    estimated_cost = 0.0
    priced_trace_count = 0

    for trace in trace_items:
        metadata = trace.get("metadata", {}) or {}
        timings = metadata.get("timings", {}) or {}
        if timings.get("retrieval_seconds") is not None:
            retrieval_seconds.append(float(timings.get("retrieval_seconds") or 0))
        if timings.get("llm_seconds") is not None:
            llm_seconds.append(float(timings.get("llm_seconds") or 0))
        if metadata.get("total_tokens") is not None:
            total_tokens += int(metadata.get("total_tokens") or 0)
        if metadata.get("estimated_cost") is not None:
            priced_trace_count += 1
            estimated_cost += float(metadata.get("estimated_cost") or 0)

    trace_count = len(trace_items)
    return {
        "trace_count": trace_count,
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": round(success_count / trace_count, 4) if trace_count else 0,
        "avg_latency_ms": round(mean(latencies), 2) if latencies else 0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "avg_final_chunks": round(mean(final_chunks), 2) if final_chunks else 0,
        "avg_total_candidates": round(mean(total_candidates), 2) if total_candidates else 0,
        "avg_retrieval_seconds": round(mean(retrieval_seconds), 3) if retrieval_seconds else 0,
        "avg_llm_seconds": round(mean(llm_seconds), 3) if llm_seconds else 0,
        "total_tokens": total_tokens,
        "estimated_cost": round(estimated_cost, 6),
        "priced_trace_count": priced_trace_count,
    }


def _summarize_knowledge_base() -> Dict[str, Any]:
    user_documents = {
        doc_id: document
        for doc_id, document in _documents_store.items()
        if document.namespace == "user"
    }
    totals = {
        "document_count": len(user_documents),
        "chunk_count": sum(document.chunk_count for document in user_documents.values()),
        "too_short_chunk_count": 0,
        "too_long_chunk_count": 0,
        "duplicate_pair_count": 0,
        "code_block_cut_count": 0,
        "empty_chunk_count": 0,
    }
    for doc_id in user_documents:
        chunks = vector_store.get_document_chunks(doc_id)
        diagnostics = _diagnose_chunks_auto(chunks)
        for key in [
            "too_short_chunk_count",
            "too_long_chunk_count",
            "duplicate_pair_count",
            "code_block_cut_count",
            "empty_chunk_count",
        ]:
            totals[key] += int(diagnostics.get(key) or 0)

    totals["avg_chunks_per_doc"] = (
        round(totals["chunk_count"] / totals["document_count"], 2)
        if totals["document_count"]
        else 0
    )
    return totals


def _summarize_evaluation() -> Dict[str, Any]:
    runs = run_store.list_runs(limit=100)
    latest_run = runs[0] if runs else None
    best_method = None
    best_hit_at_5 = 0.0
    best_mrr = 0.0

    if latest_run:
        for method, metric in (latest_run.get("metrics") or {}).items():
            hit_at_5 = float(metric.get("hit_at_5") or 0)
            if hit_at_5 >= best_hit_at_5:
                best_method = method
                best_hit_at_5 = hit_at_5
                best_mrr = float(metric.get("mrr") or 0)

    return {
        "run_count": len(runs),
        "latest_run_id": latest_run.get("run_id") if latest_run else None,
        "latest_dataset_name": latest_run.get("dataset_name") if latest_run else None,
        "latest_case_count": latest_run.get("case_count") if latest_run else 0,
        "best_method": best_method,
        "best_hit_at_5": round(best_hit_at_5, 4),
        "best_mrr": round(best_mrr, 4),
    }


def _summarize_badcases() -> Dict[str, Any]:
    badcases = badcase_store.list_badcases(limit=500)
    by_failure_type: Dict[str, int] = {}
    for badcase in badcases:
        failure_type = badcase.get("failure_type") or "other"
        by_failure_type[failure_type] = by_failure_type.get(failure_type, 0) + 1

    return {
        "total_count": len(badcases),
        "open_count": sum(1 for item in badcases if item.get("status") == "open"),
        "resolved_count": sum(1 for item in badcases if item.get("status") == "resolved"),
        "by_failure_type": by_failure_type,
    }


@router.get("/dashboard")
async def get_dashboard():
    """Return a compact RAGOps dashboard summary."""
    return {
        "success": True,
        "data": {
            "online_quality": _summarize_traces(),
            "knowledge_base": _summarize_knowledge_base(),
            "offline_evaluation": _summarize_evaluation(),
            "badcases": _summarize_badcases(),
        },
    }


@router.get("/traces")
async def list_traces(limit: int = 50):
    """List recent RAGOps traces."""
    return {
        "success": True,
        "data": trace_store.list_traces(limit=limit),
    }


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    """Get one RAGOps trace."""
    trace = trace_store.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace 不存在")
    return {
        "success": True,
        "data": trace,
    }


@router.delete("/traces")
async def clear_traces():
    """Clear local RAGOps traces."""
    trace_store.clear_traces()
    return {
        "success": True,
        "message": "RAGOps traces 已清空",
    }


@router.get("/badcases")
async def list_badcases(
    status: Optional[str] = None,
    failure_type: Optional[str] = None,
    limit: int = 50,
):
    """List RAG badcases."""
    return {
        "success": True,
        "data": badcase_store.list_badcases(
            status=status,
            failure_type=failure_type,
            limit=limit,
        ),
    }


@router.post("/badcases")
async def create_badcase(payload: Dict[str, Any]):
    """Create a badcase, optionally from a trace."""
    trace_id = payload.get("trace_id")
    if trace_id:
        trace = trace_store.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace 不存在")
        payload = {
            "trace_id": trace_id,
            "question": trace.get("question"),
            "answer": trace.get("answer"),
            "document_id": trace.get("document_id"),
            "trace_snapshot": {
                "metadata": trace.get("metadata", {}),
                "final_context": trace.get("final_context", []),
                "candidates": trace.get("candidates", [])[:30],
                "sources": trace.get("sources", []),
            },
            **payload,
        }

    badcase = badcase_store.create_badcase(payload)
    return {
        "success": True,
        "data": badcase,
    }


@router.get("/badcases/{badcase_id}")
async def get_badcase(badcase_id: str):
    """Get one badcase."""
    badcase = badcase_store.get_badcase(badcase_id)
    if not badcase:
        raise HTTPException(status_code=404, detail="Badcase 不存在")
    return {
        "success": True,
        "data": badcase,
    }


@router.patch("/badcases/{badcase_id}")
async def update_badcase(badcase_id: str, payload: Dict[str, Any]):
    """Update one badcase."""
    badcase = badcase_store.update_badcase(badcase_id, payload)
    if not badcase:
        raise HTTPException(status_code=404, detail="Badcase 不存在")
    return {
        "success": True,
        "data": badcase,
    }


@router.delete("/badcases/{badcase_id}")
async def delete_badcase(badcase_id: str):
    """Delete one badcase."""
    if not badcase_store.delete_badcase(badcase_id):
        raise HTTPException(status_code=404, detail="Badcase 不存在")
    return {
        "success": True,
        "message": "Badcase 已删除",
    }
