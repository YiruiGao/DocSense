"""Lightweight RAGOps trace storage for local diagnostics."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.common.config import settings
from app.common.logging import get_logger

logger = get_logger(__name__)

_TRACE_FILE = settings.cache_dir / "ragops_traces.json"
_MAX_TRACES = 300


def _now_ms() -> int:
    return int(time.time() * 1000)


def _load_traces() -> List[Dict[str, Any]]:
    if not _TRACE_FILE.exists():
        return []
    try:
        with open(_TRACE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning(f"加载 RAGOps traces 失败: {exc}")
        return []


def _save_traces(traces: List[Dict[str, Any]]) -> None:
    try:
        _TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_TRACE_FILE, "w", encoding="utf-8") as f:
            json.dump(traces[-_MAX_TRACES:], f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error(f"保存 RAGOps traces 失败: {exc}")


def start_trace(
    question: str,
    document_id: Optional[str],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "trace_id": f"trace_{uuid.uuid4().hex[:12]}",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_at_ms": _now_ms(),
        "status": "running",
        "question": question,
        "document_id": document_id,
        "options": options,
        "answer": "",
        "metadata": {},
        "spans": [],
        "candidates": [],
        "final_context": [],
        "sources": [],
        "error": None,
    }


def add_span(
    trace: Dict[str, Any],
    name: str,
    span_type: str,
    started_at: float,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    ended_at = time.time()
    trace["spans"].append({
        "span_id": f"span_{uuid.uuid4().hex[:10]}",
        "name": name,
        "type": span_type,
        "started_at_ms": int(started_at * 1000),
        "ended_at_ms": int(ended_at * 1000),
        "latency_ms": round((ended_at - started_at) * 1000, 2),
        "input": inputs or {},
        "output": outputs or {},
        "metadata": metadata or {},
    })


def finish_trace(
    trace: Dict[str, Any],
    status: str,
    answer: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    final_context: Optional[List[Dict[str, Any]]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    ended_at_ms = _now_ms()
    trace["ended_at_ms"] = ended_at_ms
    trace["total_latency_ms"] = ended_at_ms - int(trace.get("started_at_ms", ended_at_ms))
    trace["status"] = status
    trace["answer"] = answer
    trace["metadata"] = metadata or {}
    trace["candidates"] = candidates or []
    trace["final_context"] = final_context or []
    trace["sources"] = sources or []
    trace["error"] = error

    traces = _load_traces()
    traces.append(trace)
    _save_traces(traces)
    return trace


def list_traces(limit: int = 50) -> List[Dict[str, Any]]:
    traces = list(reversed(_load_traces()))
    items = []
    for trace in traces[:limit]:
        items.append({
            "trace_id": trace.get("trace_id"),
            "created_at": trace.get("created_at"),
            "status": trace.get("status"),
            "question": trace.get("question"),
            "document_id": trace.get("document_id"),
            "retrieval_method": trace.get("metadata", {}).get("retrieval_method"),
            "final_chunks": trace.get("metadata", {}).get("final_chunks"),
            "total_candidates": trace.get("metadata", {}).get("total_candidates"),
            "total_latency_ms": trace.get("total_latency_ms"),
            "estimated_cost": trace.get("metadata", {}).get("estimated_cost"),
        })
    return items


def load_all_traces() -> List[Dict[str, Any]]:
    return _load_traces()


def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    for trace in _load_traces():
        if trace.get("trace_id") == trace_id:
            return trace
    return None


def clear_traces() -> None:
    _save_traces([])


def trace_file_path() -> Path:
    return _TRACE_FILE
