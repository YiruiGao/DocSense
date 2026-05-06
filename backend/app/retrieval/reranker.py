"""Reranker 模块 - 使用 FlagEmbedding 进行重排序"""
from typing import List, Dict, Any, Optional
from FlagEmbedding import FlagReranker
from app.common.config import settings
import threading


class Reranker:
    """重排序器"""

    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        pass

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = FlagReranker(
                        settings.rerank_model,
                        use_fp16=True
                    )
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        对检索结果进行重排序

        Args:
            query: 查询文本
            chunks: 检索结果列表
            top_k: 返回数量

        Returns:
            List[Dict]: 重排序后的结果
        """
        if not chunks:
            return []

        # 构建查询-文档对
        pairs = [[query, chunk["content"]] for chunk in chunks]

        # 计算重排序分数
        scores = self._ensure_model().compute_score(pairs, normalize=True)

        # 确保scores是列表
        if not isinstance(scores, list):
            scores = [scores]

        # 按分数排序
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # 返回top_k结果
        results = []
        for chunk, score in scored_chunks[:top_k]:
            result = chunk.copy()
            result["rerank_score"] = float(score)
            result["source"] = "hybrid_rerank"
            results.append(result)

        return results

    def rerank_with_threshold(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 3,
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        带阈值过滤的重排序

        Args:
            query: 查询文本
            chunks: 检索结果列表
            top_k: 返回数量
            min_score: 最小分数阈值

        Returns:
            List[Dict]: 过滤后的重排序结果
        """
        reranked = self.rerank(query, chunks, top_k=len(chunks))

        # 过滤低分结果
        filtered = [
            chunk for chunk in reranked
            if chunk.get("rerank_score", 0) >= min_score
        ]

        return filtered[:top_k]


class SimpleReranker:
    """简单的重排序器（不使用模型，基于关键词匹配）"""

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        基于关键词匹配的简单重排序

        Args:
            query: 查询文本
            chunks: 检索结果列表
            top_k: 返回数量

        Returns:
            List[Dict]: 重排序后的结果
        """
        if not chunks:
            return []

        # 提取查询关键词
        query_keywords = set(query.lower().split())

        # 计算每个chunk的关键词匹配分数
        scored_chunks = []
        for chunk in chunks:
            content = chunk["content"].lower()
            content_words = set(content.split())

            # 计算关键词重叠率
            overlap = len(query_keywords & content_words)
            overlap_score = overlap / max(len(query_keywords), 1)

            # 结合原始分数
            original_score = chunk.get("score", 0.5)
            combined_score = 0.3 * overlap_score + 0.7 * original_score

            result = chunk.copy()
            result["rerank_score"] = combined_score
            result["source"] = "hybrid_simple_rerank"
            scored_chunks.append((result, combined_score))

        # 排序
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in scored_chunks[:top_k]]


def get_reranker(use_simple: bool = False) -> Reranker | SimpleReranker:
    """获取重排序器"""
    if use_simple:
        return SimpleReranker()
    return Reranker()


# 全局单例
reranker = Reranker()
