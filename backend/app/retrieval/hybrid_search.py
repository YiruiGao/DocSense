"""混合检索模块 - 向量检索 + BM25 融合"""
from typing import List, Dict, Any, Optional
from app.retrieval.vector_store import vector_store
from app.retrieval.bm25_search import bm25_search
from app.common.config import settings
from app.retrieval.embeddings import embedding_model
import numpy as np


class HybridSearch:
    """混合检索器"""

    def __init__(
        self,
        vector_weight: float = None,
        bm25_weight: float = None
    ):
        self.vector_weight = vector_weight or settings.vector_weight
        self.bm25_weight = bm25_weight or (1 - self.vector_weight)

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_id: Optional[str] = None,
        use_rerank: bool = False
    ) -> List[Dict[str, Any]]:
        """
        混合检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            document_id: 限定文档ID
            use_rerank: 是否使用重排序

        Returns:
            List[Dict]: 融合后的搜索结果
        """
        # 1. 向量检索
        vector_results = vector_store.search(
            query=query,
            top_k=top_k * 2,  # 获取更多候选
            document_id=document_id
        )

        # 2. BM25 检索
        bm25_results = bm25_search.search(
            query=query,
            top_k=top_k * 2,
            document_id=document_id
        )

        # 3. RRF 融合
        merged_results = self._rrf_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=top_k
        )

        return merged_results

    def _rrf_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 10,
        k: int = 60  # RRF 常数
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion 融合算法

        RRF(d) = sum(1 / (k + rank(d))) for each ranking

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果
            top_k: 返回数量
            k: RRF常数

        Returns:
            List[Dict]: 融合后的结果
        """
        # 收集所有chunk_id
        all_chunk_ids = set()
        for r in vector_results:
            all_chunk_ids.add(r["chunk_id"])
        for r in bm25_results:
            all_chunk_ids.add(r["chunk_id"])

        # 计算RRF分数
        scores = {}
        chunk_data = {}  # 存储chunk的详细数据

        # 向量检索排名
        for rank, result in enumerate(vector_results, 1):
            chunk_id = result["chunk_id"]
            rrf_score = self.vector_weight / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            chunk_data[chunk_id] = result

        # BM25检索排名
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result["chunk_id"]
            rrf_score = self.bm25_weight / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

        # 按分数排序
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 构建最终结果
        merged = []
        for chunk_id, score in sorted_results[:top_k]:
            data = chunk_data[chunk_id]
            merged.append({
                "chunk_id": chunk_id,
                "content": data["content"],
                "metadata": data["metadata"],
                "score": score,
                "source": "hybrid"
            })

        return merged

    def vector_only(
        self,
        query: str,
        top_k: int = 10,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """仅使用向量检索（baseline）"""
        return vector_store.search(
            query=query,
            top_k=top_k,
            document_id=document_id
        )

    def bm25_only(
        self,
        query: str,
        top_k: int = 10,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """仅使用BM25检索"""
        return bm25_search.search(
            query=query,
            top_k=top_k,
            document_id=document_id
        )


# 全局单例
hybrid_search = HybridSearch()
