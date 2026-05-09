"""向量存储模块 - 使用 ChromaDB"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from app.common.config import settings
from app.retrieval.embeddings import embedding_model
from app.models.schemas import Chunk
import uuid


class VectorStore:
    """向量存储"""

    def __init__(self):
        self._client = None
        self._collection = None

    @property
    def client(self) -> chromadb.Client:
        """获取 ChromaDB 客户端"""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(settings.chroma_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        return self._client

    @property
    def collection(self):
        """获取或创建默认集合"""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"description": "RAG document chunks"}
            )
        return self._collection

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        添加文档块到向量存储

        Args:
            chunks: 文档块列表

        Returns:
            int: 添加的数量
        """
        if not chunks:
            return 0

        # 生成 embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = embedding_model.encode(texts)

        # 准备元数据
        ids = [chunk.id for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "source": chunk.source,
                "namespace": chunk.namespace,
                "corpus_id": chunk.corpus_id or "",
            }
            for chunk in chunks
        ]

        # 添加到集合
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )

        return len(chunks)

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
        向量相似度搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            document_id: 限定文档ID
            min_score: 最小相似度分数

        Returns:
            List[Dict]: 搜索结果列表
        """
        # 生成查询向量
        query_embedding = embedding_model.encode(query)

        # 构建过滤条件
        where_filter = self._where_filter(
            document_id=document_id,
            namespace=namespace,
            corpus_id=corpus_id,
        )

        # 执行搜索
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # 格式化结果
        formatted_results = []
        if results and results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                # Chroma 返回的是距离，转换为相似度分数
                distance = results['distances'][0][i]
                score = 1 - distance  # 假设使用余弦距离

                if score < min_score:
                    continue

                formatted_results.append({
                    "chunk_id": chunk_id,
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": score
                })

        return formatted_results

    def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        corpus_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """通过embedding向量搜索"""
        where_filter = self._where_filter(
            document_id=document_id,
            namespace=namespace,
            corpus_id=corpus_id,
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i]
                score = 1 - distance

                formatted_results.append({
                    "chunk_id": chunk_id,
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": score
                })

        return formatted_results

    def delete_document(self, document_id: str) -> int:
        """删除文档的所有块"""
        # 获取该文档的所有chunk ID
        results = self.collection.get(
            where={"document_id": document_id}
        )

        if results and results['ids']:
            self.collection.delete(ids=results['ids'])
            return len(results['ids'])

        return 0

    def list_document_ids(self) -> List[str]:
        """List document ids currently present in the vector collection."""
        results = self.collection.get(include=["metadatas"])
        document_ids = {
            metadata.get("document_id")
            for metadata in results.get("metadatas", [])
            if metadata and metadata.get("document_id")
        }
        return sorted(document_ids)

    def delete_documents_not_in(self, valid_document_ids: set[str]) -> int:
        """Delete vector chunks whose document id is no longer in the metadata store."""
        removed = 0
        for document_id in self.list_document_ids():
            if document_id not in valid_document_ids:
                removed += self.delete_document(document_id)
        return removed

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """获取文档的所有块"""
        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )

        chunks = []
        if results and results['ids']:
            for i, chunk_id in enumerate(results['ids']):
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": results['documents'][i],
                    "metadata": results['metadatas'][i]
                })

        return chunks

    def update_document_metadata(
        self,
        document_id: str,
        namespace: str,
        corpus_id: Optional[str] = None,
    ) -> int:
        """Backfill namespace/corpus metadata for existing chunks."""
        results = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"]
        )
        if not results or not results.get("ids"):
            return 0

        metadatas = []
        changed = 0
        for metadata in results.get("metadatas", []):
            item = dict(metadata or {})
            next_corpus_id = corpus_id or ""
            if item.get("namespace") != namespace or item.get("corpus_id", "") != next_corpus_id:
                changed += 1
            item["namespace"] = namespace
            item["corpus_id"] = next_corpus_id
            metadatas.append(item)

        if changed:
            self.collection.update(ids=results["ids"], metadatas=metadatas)
        return changed

    def count(self, document_id: Optional[str] = None) -> int:
        """统计块数量"""
        if document_id:
            results = self.collection.get(where={"document_id": document_id})
            return len(results['ids']) if results and results['ids'] else 0
        return self.collection.count()

    @staticmethod
    def _where_filter(
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        corpus_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        filters = []
        if document_id:
            filters.append({"document_id": document_id})
        if namespace:
            filters.append({"namespace": namespace})
        if corpus_id:
            filters.append({"corpus_id": corpus_id})
        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    def reset(self) -> None:
        """重置向量存储"""
        self.client.delete_collection("documents")
        self._collection = None


# 全局单例
vector_store = VectorStore()
