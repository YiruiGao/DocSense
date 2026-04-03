"""缓存工具"""
from cachetools import LRUCache
from typing import Optional, Any
import hashlib
import json


class EmbeddingCache:
    """Embedding 缓存"""

    def __init__(self, maxsize: int = 1000):
        self._cache = LRUCache(maxsize=maxsize)

    @staticmethod
    def _hash_text(text: str) -> str:
        """生成文本的哈希键"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[Any]:
        """获取缓存的embedding"""
        key = self._hash_text(text)
        return self._cache.get(key)

    def set(self, text: str, embedding: Any) -> None:
        """设置缓存"""
        key = self._hash_text(text)
        self._cache[key] = embedding

    def contains(self, text: str) -> bool:
        """检查是否缓存"""
        key = self._hash_text(text)
        return key in self._cache

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class QueryCache:
    """查询结果缓存"""

    def __init__(self, maxsize: int = 100):
        self._cache = LRUCache(maxsize=maxsize)

    @staticmethod
    def _hash_query(query: str, doc_id: Optional[str] = None) -> str:
        """生成查询的哈希键"""
        content = f"{query}:{doc_id or 'all'}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def get(self, query: str, doc_id: Optional[str] = None) -> Optional[dict]:
        """获取缓存的查询结果"""
        key = self._hash_query(query, doc_id)
        cached = self._cache.get(key)
        if cached:
            return json.loads(cached)
        return None

    def set(
        self,
        query: str,
        result: dict,
        doc_id: Optional[str] = None
    ) -> None:
        """设置缓存"""
        key = self._hash_query(query, doc_id)
        self._cache[key] = json.dumps(result, ensure_ascii=False)

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# 全局查询缓存实例
_query_cache = QueryCache()
