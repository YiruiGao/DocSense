"""聊天问答 API"""
from fastapi import APIRouter, HTTPException
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
import re
import time

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    QueryMetadata,
    QueryOptions,
    Source,
    ErrorResponse
)
from app.retrieval.bm25_search import bm25_search
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import reranker
from app.retrieval.vector_store import vector_store
from app.generation.llm import zai_llm
from app.common.config import settings
from app.ops import traces as trace_store
from app.utils.cache import _query_cache
from app.common.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["chat"])


def _bounded_score(value: float) -> float:
    """Keep source scores compatible with the public response schema."""
    return max(0.0, min(float(value), 1.0))


def _normalize_content(content: str) -> str:
    return re.sub(r"\s+", "", content.lower())


def _is_similar_content(a: str, b: str) -> bool:
    """Detect near-duplicate chunks before sending context to the LLM."""
    norm_a = _normalize_content(a)
    norm_b = _normalize_content(b)
    if not norm_a or not norm_b:
        return False

    shorter, longer = sorted((norm_a, norm_b), key=len)
    if len(shorter) >= 80 and shorter in longer:
        return True

    ratio = SequenceMatcher(None, norm_a[:1200], norm_b[:1200]).ratio()
    return ratio >= settings.duplicate_similarity_threshold


def _dedupe_similar_chunks(
    chunks: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    for chunk in chunks:
        content = chunk.get("content", "")
        if any(_is_similar_content(content, kept.get("content", "")) for kept in deduped):
            continue
        deduped.append(chunk)
        if len(deduped) >= top_k:
            break
    return deduped


def _preview_content(content: str, limit: int = 240) -> str:
    if len(content) <= limit:
        return content
    return content[:limit] + "..."


def _candidate_record(
    result: Dict[str, Any],
    stage: str,
    rank: int,
    final_chunk_ids: Optional[set[str]] = None,
) -> Dict[str, Any]:
    metadata = result.get("metadata", {})
    return {
        "chunk_id": result.get("chunk_id", ""),
        "document_id": metadata.get("document_id"),
        "document_name": metadata.get("source"),
        "page_number": metadata.get("page_number", result.get("page_number")),
        "chunk_index": metadata.get("chunk_index"),
        "content_preview": _preview_content(result.get("content", "")),
        "stage": stage,
        "rank": rank,
        "score": result.get("score"),
        "vector_score": result.get("vector_score"),
        "bm25_score": result.get("bm25_score"),
        "hybrid_score": result.get("hybrid_score", result.get("score") if result.get("source") == "hybrid" else None),
        "rerank_score": result.get("rerank_score"),
        "selected_for_context": result.get("chunk_id") in final_chunk_ids if final_chunk_ids is not None else False,
        "selection_reason": "selected_for_llm" if final_chunk_ids and result.get("chunk_id") in final_chunk_ids else "",
    }


def _context_record(result: Dict[str, Any], rank: int) -> Dict[str, Any]:
    metadata = result.get("metadata", {})
    return {
        "rank": rank,
        "chunk_id": result.get("chunk_id", ""),
        "document_id": metadata.get("document_id"),
        "document_name": metadata.get("source"),
        "page_number": metadata.get("page_number", result.get("page_number")),
        "chunk_index": metadata.get("chunk_index"),
        "content_preview": _preview_content(result.get("content", ""), limit=500),
        "score": result.get("rerank_score", result.get("score")),
    }


def _mark_selected(candidates: List[Dict[str, Any]], final_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    final_ids = {result.get("chunk_id") for result in final_results}
    updated = []
    for candidate in candidates:
        item = candidate.copy()
        if item.get("chunk_id") in final_ids:
            item["selected_for_context"] = True
            item["selection_reason"] = "selected_for_llm"
        updated.append(item)
    return updated


def _source_from_result(result: Dict[str, Any], preview: bool = False) -> Source:
    content = result.get("content", "")
    if preview and len(content) > 200:
        content = content[:200] + "..."

    return Source(
        chunk_id=result.get("chunk_id", ""),
        document_id=result.get("metadata", {}).get("document_id"),
        document_name=result.get("metadata", {}).get("source"),
        page_number=result.get("metadata", {}).get("page_number", result.get("page_number", 0)),
        chunk_index=result.get("metadata", {}).get("chunk_index"),
        content=content,
        score=_bounded_score(result.get("rerank_score", result.get("score", 0.0))),
    )


def _is_overview_question(question: str) -> bool:
    """Questions asking for a document-level summary need broad context."""
    patterns = [
        "主要内容",
        "核心观点",
        "关键结论",
        "主要观点",
        "总结",
        "概括",
        "讲了什么",
        "是什么文档",
        "有哪些关键",
    ]
    return any(pattern in question for pattern in patterns)


def _document_overview_chunks(document_id: Optional[str], top_k: int) -> List[Dict[str, Any]]:
    if not document_id:
        return []

    chunks = vector_store.get_document_chunks(document_id)
    chunks.sort(key=lambda item: item.get("metadata", {}).get("chunk_index", 0))
    selected = chunks[: max(top_k, settings.overview_context_chunks)]

    for chunk in selected:
        chunk["score"] = 1.0
        chunk["source"] = "document_overview"

    return selected


def _has_enough_relevance(results: List[Dict[str, Any]], used_rerank: bool) -> bool:
    if not results:
        return False

    if used_rerank:
        best_score = max(float(result.get("rerank_score", 0.0)) for result in results)
        return best_score >= settings.rerank_min_relevance_score

    return True


def _empty_answer_response(
    method_name: str,
    candidate_count: int,
    retrieval_time: float,
    rerank_time: float,
    start_time: float,
    trace_id: Optional[str] = None,
) -> QueryResponse:
    response_time = time.time() - start_time
    metadata = QueryMetadata(
        retrieval_method=method_name,
        total_candidates=candidate_count,
        final_chunks=0,
        response_time_seconds=round(response_time, 3),
        trace_id=trace_id,
        timings={
            "retrieval_seconds": round(retrieval_time, 3),
            "rerank_seconds": round(rerank_time, 3),
            "llm_seconds": 0.0,
        }
    )
    response_data = {
        "answer": "文档中没有足够相关的信息来回答这个问题。",
        "sources": [],
        "metadata": metadata.model_dump()
    }
    return QueryResponse(success=True, data=response_data)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    RAG问答接口

    流程:
    1. 检查缓存
    2. 选择检索策略
    3. 检索相关文档块
    4. 可选：重排序
    5. 生成回答
    6. 返回结果
    """
    start_time = time.time()
    logger.info(f"收到查询请求: question={request.question[:50]}...")
    options = request.options or QueryOptions()
    trace = trace_store.start_trace(
        question=request.question,
        document_id=request.document_id,
        options=options.model_dump(),
    )

    # 检查缓存
    cached = _query_cache.get(request.question, request.document_id)
    if cached:
        logger.info("命中缓存")
        cached = {
            **cached,
            "metadata": dict(cached.get("metadata") or {}),
            "sources": list(cached.get("sources") or []),
        }
        cached["metadata"]["from_cache"] = True
        cached["metadata"]["trace_id"] = trace["trace_id"]
        trace_store.finish_trace(
            trace=trace,
            status="cache_hit",
            answer=cached.get("answer", ""),
            metadata=cached.get("metadata", {}),
            candidates=[],
            final_context=[],
            sources=cached.get("sources", []),
        )
        return QueryResponse(success=True, data=cached)

    # 选择检索方法
    use_hybrid = options.use_hybrid_search
    use_rerank = options.use_rerank
    top_k = options.top_k

    logger.debug(f"检索配置: hybrid={use_hybrid}, rerank={use_rerank}, top_k={top_k}")

    try:
        retrieval_started = time.time()
        candidate_top_k = top_k * settings.retrieval_candidate_multiplier
        is_overview_question = _is_overview_question(request.question)
        trace_candidates: List[Dict[str, Any]] = []

        # 检索
        if is_overview_question and request.document_id:
            logger.debug("使用文档总览上下文")
            results = _document_overview_chunks(request.document_id, top_k)
            method_name = "document_overview"
            trace_candidates.extend([
                _candidate_record(result, "document_overview", rank)
                for rank, result in enumerate(results, 1)
            ])
        elif use_hybrid:
            logger.debug("使用混合检索")
            vector_started = time.time()
            vector_results = vector_store.search(
                query=request.question,
                top_k=candidate_top_k * 2,
                document_id=request.document_id,
            )
            trace_store.add_span(
                trace,
                name="vector_search",
                span_type="retrieval",
                started_at=vector_started,
                inputs={"top_k": candidate_top_k * 2, "document_id": request.document_id},
                outputs={"count": len(vector_results)},
            )
            bm25_started = time.time()
            bm25_results = bm25_search.search(
                query=request.question,
                top_k=candidate_top_k * 2,
                document_id=request.document_id,
            )
            trace_store.add_span(
                trace,
                name="bm25_search",
                span_type="retrieval",
                started_at=bm25_started,
                inputs={"top_k": candidate_top_k * 2, "document_id": request.document_id},
                outputs={"count": len(bm25_results)},
            )
            merge_started = time.time()
            results = hybrid_search._rrf_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results,
                top_k=candidate_top_k,
            )
            trace_store.add_span(
                trace,
                name="hybrid_merge",
                span_type="retrieval",
                started_at=merge_started,
                inputs={"vector_count": len(vector_results), "bm25_count": len(bm25_results)},
                outputs={"count": len(results)},
                metadata={"vector_weight": settings.vector_weight},
            )
            trace_candidates.extend([
                _candidate_record(result, "vector_search", rank)
                for rank, result in enumerate(vector_results, 1)
            ])
            trace_candidates.extend([
                _candidate_record(result, "bm25_search", rank)
                for rank, result in enumerate(bm25_results, 1)
            ])
            trace_candidates.extend([
                _candidate_record(result, "hybrid_merge", rank)
                for rank, result in enumerate(results, 1)
            ])
            method_name = "hybrid_search"
        else:
            logger.debug("使用纯向量检索")
            results = vector_store.search(
                query=request.question,
                top_k=candidate_top_k,
                document_id=request.document_id
            )
            method_name = "vector_search"
            trace_candidates.extend([
                _candidate_record(result, "vector_search", rank)
                for rank, result in enumerate(results, 1)
            ])

        trace_store.add_span(
            trace,
            name=method_name,
            span_type="retrieval",
            started_at=retrieval_started,
            inputs={"candidate_top_k": candidate_top_k, "document_id": request.document_id},
            outputs={"count": len(results)},
        )

        retrieval_time = time.time() - retrieval_started
        candidate_count = len(results)
        rerank_time = 0.0
        used_rerank = False

        if not results:
            response = _empty_answer_response(
                method_name=method_name,
                candidate_count=0,
                retrieval_time=retrieval_time,
                rerank_time=0.0,
                start_time=start_time,
                trace_id=trace["trace_id"],
            )
            trace_store.finish_trace(
                trace=trace,
                status="empty_retrieval",
                answer=response.data["answer"],
                metadata=response.data["metadata"],
                candidates=trace_candidates,
                final_context=[],
                sources=[],
            )
            _query_cache.set(request.question, response.data, request.document_id)
            return response

        # Rerank
        if use_rerank and not is_overview_question and len(results) > top_k:
            logger.debug(f"执行 Rerank, 候选数={len(results)}")
            rerank_started = time.time()
            results = reranker.rerank(
                query=request.question,
                chunks=results,
                top_k=len(results)
            )
            rerank_time = time.time() - rerank_started
            used_rerank = True
            trace_store.add_span(
                trace,
                name="rerank",
                span_type="rerank",
                started_at=rerank_started,
                inputs={"count": candidate_count},
                outputs={"count": len(results)},
            )
            trace_candidates.extend([
                _candidate_record(result, "rerank", rank)
                for rank, result in enumerate(results, 1)
            ])
        else:
            logger.debug("跳过 Rerank (结果数不足)")

        results = _dedupe_similar_chunks(results, top_k)
        trace_candidates = _mark_selected(trace_candidates, results)
        if not _has_enough_relevance(results, used_rerank):
            response = _empty_answer_response(
                method_name=method_name,
                candidate_count=candidate_count,
                retrieval_time=retrieval_time,
                rerank_time=rerank_time,
                start_time=start_time,
                trace_id=trace["trace_id"],
            )
            trace_store.finish_trace(
                trace=trace,
                status="low_relevance",
                answer=response.data["answer"],
                metadata=response.data["metadata"],
                candidates=trace_candidates,
                final_context=[],
                sources=[],
            )
            _query_cache.set(request.question, response.data, request.document_id)
            return response

        logger.info(f"检索完成: {len(results)} 个结果")

        # 生成回答
        try:
            logger.debug("调用 LLM 生成回答")
            llm_started = time.time()
            answer = await zai_llm.generate_with_sources(
                question=request.question,
                chunks=results
            )
            answer = answer.strip()
            llm_time = time.time() - llm_started
            trace_store.add_span(
                trace,
                name="llm_generate",
                span_type="llm",
                started_at=llm_started,
                inputs={
                    "provider": getattr(zai_llm, "provider", "unknown"),
                    "model": getattr(zai_llm, "model", "unknown"),
                    "context_chunks": len(results),
                },
                outputs={"answer_chars": len(answer)},
            )
            if not answer:
                logger.warning("LLM 返回空内容")
                answer = "文档中没有足够相关的信息来回答这个问题。"
            logger.info(f"LLM 回答生成成功: {len(answer)} 字符")
        except Exception as e:
            logger.error(f"LLM 调用失败: {str(e)}")
            trace_store.finish_trace(
                trace=trace,
                status="llm_error",
                metadata={
                    "retrieval_method": method_name,
                    "total_candidates": candidate_count,
                    "final_chunks": len(results),
                    "trace_id": trace["trace_id"],
                },
                candidates=trace_candidates,
                final_context=[_context_record(result, rank) for rank, result in enumerate(results, 1)],
                error=str(e),
            )
            raise HTTPException(status_code=500, detail=f"生成回答失败: {str(e)}")

        # 构建响应
        sources = [_source_from_result(r, preview=True) for r in results]

        response_time = time.time() - start_time

        metadata = QueryMetadata(
            retrieval_method=method_name,
            total_candidates=candidate_count,
            final_chunks=len(results),
            response_time_seconds=round(response_time, 3),
            trace_id=trace["trace_id"],
            timings={
                "retrieval_seconds": round(retrieval_time, 3),
                "rerank_seconds": round(rerank_time, 3),
                "llm_seconds": round(llm_time, 3),
            }
        )

        response_data = {
            "answer": answer,
            "sources": [s.model_dump() for s in sources],
            "metadata": metadata.model_dump()
        }

        trace_store.finish_trace(
            trace=trace,
            status="success",
            answer=answer,
            metadata=metadata.model_dump() | {
                "llm_provider": getattr(zai_llm, "provider", "unknown"),
                "llm_model": getattr(zai_llm, "model", "unknown"),
            },
            candidates=trace_candidates,
            final_context=[_context_record(result, rank) for rank, result in enumerate(results, 1)],
            sources=[s.model_dump() for s in sources],
        )

        # 缓存结果
        _query_cache.set(request.question, response_data, request.document_id)
        logger.info(f"查询完成: 总耗时 {response_time:.2f}s")

        return QueryResponse(success=True, data=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询处理失败: {str(e)}")
        trace_store.finish_trace(
            trace=trace,
            status="error",
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_only(request: QueryRequest):
    """
    仅检索，不生成回答
    """
    start_time = time.time()
    logger.info(f"收到检索请求: question={request.question[:50]}...")

    options = request.options or QueryOptions()
    use_hybrid = options.use_hybrid_search
    use_rerank = options.use_rerank
    top_k = options.top_k

    logger.debug(f"检索配置: hybrid={use_hybrid}, rerank={use_rerank}, top_k={top_k}")

    try:
        candidate_top_k = top_k * settings.retrieval_candidate_multiplier
        if use_hybrid:
            logger.debug("使用混合检索")
            results = hybrid_search.search(
                query=request.question,
                top_k=candidate_top_k,
                document_id=request.document_id
            )
            method_name = "hybrid_search"
        else:
            logger.debug("使用纯向量检索")
            results = vector_store.search(
                query=request.question,
                top_k=candidate_top_k,
                document_id=request.document_id
            )
            method_name = "vector_search"

        if use_rerank and len(results) > top_k:
            logger.debug(f"执行 Rerank, 候选数={len(results)}")
            results = reranker.rerank(
                query=request.question,
                chunks=results,
                top_k=len(results)
            )

        results = _dedupe_similar_chunks(results, top_k)
        logger.info(f"检索完成: {len(results)} 个结果")

        sources = [_source_from_result(r) for r in results]

        response_time = time.time() - start_time

        logger.info(f"检索完成: 总耗时 {response_time:.2f}s")

        return {
            "success": True,
            "data": {
                "results": [s.model_dump() for s in sources],
                "total": len(sources)
            }
        }
    except Exception as e:
        logger.error(f"检索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
