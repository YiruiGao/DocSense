"""Evaluation API for RAGOps."""
import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.api.chat import _query_cache
from app.api.documents import _documents_store
from app.evaluation import dataset_store, run_store
from app.evaluation.evaluators.retrieval import RetrievalComparisonEvaluator
from app.evaluation.models import MethodResult
from app.evaluation.test_cases import TestCaseSet
from app.models.schemas import EvaluationRequest, EvaluationRunResponse

router = APIRouter(tags=["evaluation"])


def _method_metrics(method_result: MethodResult) -> Dict[str, Any]:
    return {
        "hit_at_3": method_result.hit_rate.hit_at_3,
        "hit_at_5": method_result.hit_rate.hit_at_5,
        "hit_at_10": method_result.hit_rate.hit_at_10,
        "mrr": method_result.mrr.mrr,
        "avg_response_time": method_result.avg_response_time,
        "errors": len(method_result.errors),
    }


def _method_results(method_result: MethodResult) -> list[Dict[str, Any]]:
    items = []
    for result in method_result.results:
        items.append({
            "test_case_id": result.get("test_case_id"),
            "question": result.get("question"),
            "expected_chunks": result.get("expected_chunk_ids", []),
            "retrieved_chunks": result.get("retrieved_chunk_ids", []),
            "retrieved_previews": [
                content[:180] + "..." if len(content) > 180 else content
                for content in result.get("retrieved_contents", [])[:5]
            ],
            "hit": result.get("hit", False),
            "rank": result.get("rank"),
            "response_time": result.get("response_time", 0.0),
            "error": result.get("error"),
        })
    return items


@router.post("/run", response_model=EvaluationRunResponse)
async def run_evaluation(request: EvaluationRequest):
    """Run retrieval evaluation across selected methods."""
    if not _documents_store:
        raise HTTPException(status_code=400, detail="请先上传文档")

    doc_id = request.document_id
    if doc_id and doc_id not in _documents_store:
        raise HTTPException(status_code=404, detail="文档不存在")

    methods = request.methods or ["baseline", "hybrid", "hybrid_rerank"]
    dataset_id = request.test_set_id or "default"
    dataset = dataset_store.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="评测集不存在")

    test_cases = dataset_store.cases_as_test_cases(dataset, document_id=doc_id)
    if not test_cases:
        raise HTTPException(status_code=400, detail="评测集没有可用 case")

    test_set = TestCaseSet(
        id=dataset.get("dataset_id"),
        name=dataset.get("name"),
        description=dataset.get("description"),
        test_cases=test_cases,
        document_id=dataset.get("document_id"),
    )
    run_id = f"eval_{uuid.uuid4().hex[:12]}"
    started_at = time.time()

    evaluator = RetrievalComparisonEvaluator()
    try:
        method_results = await evaluator.evaluate(
            methods=methods,
            test_set=test_set,
            document_id=doc_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results_by_method = {
        method: _method_results(result)
        for method, result in method_results.items()
    }
    metrics_by_method = {
        method: _method_metrics(result)
        for method, result in method_results.items()
    }

    run = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_id": dataset.get("dataset_id"),
        "dataset_name": dataset.get("name"),
        "document_id": doc_id,
        "document_name": _documents_store[doc_id].name if doc_id else "全部文档",
        "methods": methods,
        "case_count": len(test_cases),
        "strategy_config": {
            "methods": methods,
        },
        "status": "success",
        "duration_seconds": round(time.time() - started_at, 3),
        "results": results_by_method,
        "metrics": metrics_by_method,
    }
    run_store.add_run(run)

    return EvaluationRunResponse(
        success=True,
        data={
            "run_id": run_id,
            "dataset_id": dataset.get("dataset_id"),
            "dataset_name": dataset.get("name"),
            "document_id": doc_id,
            "document_name": run["document_name"],
            "methods": methods,
            "case_count": run["case_count"],
            "duration_seconds": run["duration_seconds"],
            "results": results_by_method,
            "comparison": metrics_by_method,
        },
    )


@router.get("/datasets")
async def list_datasets(include_cases: bool = False):
    """List evaluation datasets."""
    return {
        "success": True,
        "data": dataset_store.list_datasets(include_cases=include_cases),
    }


@router.post("/datasets")
async def create_dataset(payload: Dict[str, Any]):
    """Create an evaluation dataset."""
    dataset = dataset_store.create_dataset(payload)
    return {
        "success": True,
        "data": dataset,
    }


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get one evaluation dataset."""
    dataset = dataset_store.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="评测集不存在")
    return {
        "success": True,
        "data": dataset,
    }


@router.patch("/datasets/{dataset_id}")
async def update_dataset(dataset_id: str, payload: Dict[str, Any]):
    """Update one evaluation dataset."""
    dataset = dataset_store.update_dataset(dataset_id, payload)
    if not dataset:
        raise HTTPException(status_code=404, detail="评测集不存在")
    return {
        "success": True,
        "data": dataset,
    }


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete one evaluation dataset."""
    if dataset_id == "default":
        raise HTTPException(status_code=400, detail="默认评测集不能删除")
    if not dataset_store.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="评测集不存在")
    return {"success": True, "message": "评测集已删除"}


@router.post("/datasets/{dataset_id}/cases")
async def add_case(dataset_id: str, payload: Dict[str, Any]):
    """Add one test case to a dataset."""
    case = dataset_store.add_case(dataset_id, payload)
    if not case:
        raise HTTPException(status_code=404, detail="评测集不存在")
    return {
        "success": True,
        "data": case,
    }


@router.patch("/datasets/{dataset_id}/cases/{case_id}")
async def update_case(dataset_id: str, case_id: str, payload: Dict[str, Any]):
    """Update one test case."""
    case = dataset_store.update_case(dataset_id, case_id, payload)
    if not case:
        raise HTTPException(status_code=404, detail="Case 不存在")
    return {
        "success": True,
        "data": case,
    }


@router.delete("/datasets/{dataset_id}/cases/{case_id}")
async def delete_case(dataset_id: str, case_id: str):
    """Delete one test case."""
    if not dataset_store.delete_case(dataset_id, case_id):
        raise HTTPException(status_code=404, detail="Case 不存在")
    return {"success": True, "message": "Case 已删除"}


@router.get("/runs")
async def list_evaluation_runs(limit: int = 20):
    """List recent evaluation runs."""
    return {
        "success": True,
        "data": run_store.list_runs(limit=limit),
    }


@router.get("/runs/{run_id}")
async def get_evaluation_run(run_id: str):
    """Get evaluation run detail."""
    run = run_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="评估运行不存在")

    return {
        "success": True,
        "data": run,
    }


@router.delete("/runs")
async def clear_evaluation_runs():
    """Clear local evaluation runs."""
    run_store.clear_runs()
    return {"success": True, "message": "评估运行已清空"}


@router.get("/test-cases")
async def get_test_cases():
    """Get default test cases."""
    dataset = dataset_store.get_dataset("default")
    return {
        "success": True,
        "data": dataset.get("cases", []) if dataset else [],
    }


@router.post("/clear-cache")
async def clear_cache():
    """Clear query cache."""
    _query_cache.clear()
    return {"success": True, "message": "缓存已清除"}
