"""Embedding 模块 - 使用 sentence-transformers 生成向量"""
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np
from functools import lru_cache
import hashlib
from app.common.config import settings
from app.utils.cache import EmbeddingCache


class EmbeddingModel:
    """Embedding 模型封装"""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._model = SentenceTransformer(settings.embedding_model)
            self._cache = EmbeddingCache(maxsize=settings.embedding_cache_size)

    def encode(
        self,
        texts: str | List[str],
        use_cache: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        生成文本的向量表示

        Args:
            texts: 单个文本或文本列表
            use_cache: 是否使用缓存
            normalize: 是否归一化向量

        Returns:
            np.ndarray: 向量数组
        """
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # 检查缓存
        for i, text in enumerate(texts):
            if use_cache:
                cached = self._cache.get(text)
                if cached is not None:
                    embeddings.append((i, cached))
                    continue
            uncached_texts.append(text)
            uncached_indices.append(i)

        # 对未缓存的文本生成embedding
        if uncached_texts:
            new_embeddings = self._model.encode(
                uncached_texts,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )

            # 缓存新生成的embedding
            for text, embedding in zip(uncached_texts, new_embeddings):
                if use_cache:
                    self._cache.set(text, embedding)

            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, embedding))

        # 按原始顺序排序
        embeddings.sort(key=lambda x: x[0])
        result = np.array([e[1] for e in embeddings])

        if single_input:
            return result[0]
        return result

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        批量生成embedding

        Args:
            texts: 文本列表
            batch_size: 批次大小
            use_cache: 是否使用缓存

        Returns:
            np.ndarray: 向量数组
        """
        return self.encode(texts, use_cache=use_cache)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        ))

    @property
    def dimension(self) -> int:
        """返回向量维度"""
        return self._model.get_sentence_embedding_dimension()


class EmbeddingModelLazy:
    """延迟加载的 Embedding 模型"""

    def __init__(self):
        self._model = None

    @property
    def model(self) -> EmbeddingModel:
        if self._model is None:
            self._model = EmbeddingModel()
        return self._model

    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)


# 全局单例
embedding_model = EmbeddingModelLazy()
