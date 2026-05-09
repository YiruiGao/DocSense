"""BM25 检索模块"""
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional, Tuple
import jieba
import json
from pathlib import Path
from app.common.config import settings
import pickle


class BM25Search:
    """BM25 检索器"""

    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._documents: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._chunk_ids: List[str] = []
        self._tokenized_docs: List[List[str]] = []

        # 加载词典（可选）
        self._load_custom_dict()

        # 尝试加载已有索引
        self._load_index()

    def _load_custom_dict(self):
        """加载自定义词典"""
        dict_path = settings.data_dir / "custom_dict.txt"
        if dict_path.exists():
            jieba.load_userdict(str(dict_path))

    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        # 使用 jieba 分词
        tokens = jieba.lcut(text.lower())
        # 过滤空白和标点
        tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]
        return tokens

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        chunk_ids: List[str]
    ) -> None:
        """
        添加文档到索引

        Args:
            documents: 文档内容列表
            metadatas: 元数据列表
            chunk_ids: 块ID列表
        """
        if not documents:
            return

        # 分词
        new_tokenized = [self._tokenize(doc) for doc in documents]

        # 添加到存储
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)
        self._chunk_ids.extend(chunk_ids)
        self._tokenized_docs.extend(new_tokenized)

        # 重建索引
        self._index = BM25Okapi(self._tokenized_docs)

        # 保存索引
        self._save_index()

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        corpus_id: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        BM25 搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            document_id: 限定文档ID
            min_score: 最小分数

        Returns:
            List[Dict]: 搜索结果
        """
        if self._index is None:
            return []

        # 分词查询
        query_tokens = self._tokenize(query)

        # 获取分数
        scores = self._index.get_scores(query_tokens)

        # 排序并获取top_k
        results = []
        scored_docs = list(enumerate(scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        for idx, score in scored_docs:
            if score < min_score:
                continue

            # 过滤文档ID
            if document_id:
                if self._metadatas[idx].get("document_id") != document_id:
                    continue
            if namespace and self._metadatas[idx].get("namespace", "user") != namespace:
                continue
            if corpus_id and self._metadatas[idx].get("corpus_id") != corpus_id:
                continue

            results.append({
                "chunk_id": self._chunk_ids[idx],
                "content": self._documents[idx],
                "metadata": self._metadatas[idx],
                "score": float(score)
            })

            if len(results) >= top_k:
                break

        return results

    def remove_document(self, document_id: str) -> int:
        """删除文档的所有块"""
        # 找到要删除的索引
        indices_to_remove = [
            i for i, meta in enumerate(self._metadatas)
            if meta.get("document_id") == document_id
        ]

        if not indices_to_remove:
            return 0

        # 从后向前删除以保持索引正确
        for idx in sorted(indices_to_remove, reverse=True):
            del self._documents[idx]
            del self._metadatas[idx]
            del self._chunk_ids[idx]
            del self._tokenized_docs[idx]

        # 重建索引
        if self._tokenized_docs:
            self._index = BM25Okapi(self._tokenized_docs)
        else:
            self._index = None

        # 保存
        self._save_index()

        return len(indices_to_remove)

    def remove_documents_not_in(self, valid_document_ids: set[str]) -> int:
        """删除元信息存储中不存在的文档块"""
        stale_document_ids = {
            meta.get("document_id")
            for meta in self._metadatas
            if meta.get("document_id") and meta.get("document_id") not in valid_document_ids
        }

        removed = 0
        for document_id in stale_document_ids:
            removed += self.remove_document(document_id)
        return removed

    def _save_index(self):
        """保存索引到磁盘"""
        index_path = settings.cache_dir / "bm25_index.pkl"
        data = {
            "documents": self._documents,
            "metadatas": self._metadatas,
            "chunk_ids": self._chunk_ids,
            "tokenized_docs": self._tokenized_docs
        }
        with open(index_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_index(self):
        """从磁盘加载索引"""
        index_path = settings.cache_dir / "bm25_index.pkl"
        if not index_path.exists():
            return

        try:
            with open(index_path, 'rb') as f:
                data = pickle.load(f)

            self._documents = data.get("documents", [])
            self._metadatas = data.get("metadatas", [])
            self._chunk_ids = data.get("chunk_ids", [])
            self._tokenized_docs = data.get("tokenized_docs", [])

            if self._tokenized_docs:
                self._index = BM25Okapi(self._tokenized_docs)

        except Exception as e:
            print(f"加载BM25索引失败: {e}")

    def count(self, document_id: Optional[str] = None) -> int:
        """统计文档数量"""
        if document_id:
            return sum(
                1 for meta in self._metadatas
                if meta.get("document_id") == document_id
            )
        return len(self._documents)

    def clear(self):
        """清空索引"""
        self._index = None
        self._documents = []
        self._metadatas = []
        self._chunk_ids = []
        self._tokenized_docs = []
        self._save_index()


# 全局单例
bm25_search = BM25Search()
