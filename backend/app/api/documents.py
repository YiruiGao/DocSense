"""文档管理 API"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional, Tuple, List
from difflib import SequenceMatcher
import uuid
import time
import json
import hashlib
from pathlib import Path

from app.models.schemas import (
    DocumentUploadResponse,
    DocumentInfo,
    DocumentListResponse,
    ErrorResponse
)
from app.ingestion.chunker import SemanticChunker
from app.ingestion.pdf_processor import PDFProcessor, PDFProcessingError
from app.ingestion.text_processor import TextProcessor, TextProcessingError
from app.models.schemas import Chunk
from app.retrieval.bm25_search import bm25_search
from app.retrieval.vector_store import vector_store
from app.common.config import settings
from app.utils.cache import _query_cache
from app.common.logging import get_logger

logger = get_logger(__name__)

# 支持的文件类型
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}

router = APIRouter(tags=["documents"])

# 持久化文件路径
_STORE_FILE = settings.cache_dir / "documents_store.json"
_FULL_DIAGNOSTICS_CHUNK_LIMIT = 50
_DIAGNOSTICS_CHUNK_PREVIEW_LIMIT = 200


def _load_store() -> dict:
    """从磁盘加载文档元信息"""
    if _STORE_FILE.exists():
        try:
            with open(_STORE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 转换为 DocumentInfo 对象
                return {k: DocumentInfo(**v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"加载文档元信息失败: {e}")
    return {}


def _save_store() -> None:
    """保存文档元信息到磁盘"""
    try:
        with open(_STORE_FILE, "w", encoding="utf-8") as f:
            # 转换为 dict 以便 JSON 序列化
            data = {k: v.model_dump() for k, v in _documents_store.items()}
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"文档元信息已保存: {len(_documents_store)} 个文档")
    except Exception as e:
        logger.error(f"保存文档元信息失败: {e}")


# 全局状态 - 启动时从磁盘加载
_documents_store: dict = _load_store()
logger.info(f"已加载 {len(_documents_store)} 个文档元信息")


def _backfill_document_index_metadata() -> None:
    changed = 0
    for doc_id, doc_info in _documents_store.items():
        changed += vector_store.update_document_metadata(
            document_id=doc_id,
            namespace=doc_info.namespace,
            corpus_id=doc_info.corpus_id,
        )
    if changed:
        logger.info(f"已补齐 {changed} 个 chunk 的 namespace/corpus 元数据")


_backfill_document_index_metadata()


def _normalize_for_similarity(content: str) -> str:
    return "".join(content.lower().split())


def _chunk_payload(chunk: dict) -> dict:
    metadata = chunk.get("metadata", {})
    content = chunk.get("content", "")
    return {
        "chunk_id": chunk.get("chunk_id"),
        "document_id": metadata.get("document_id"),
        "document_name": metadata.get("source"),
        "page_number": metadata.get("page_number"),
        "chunk_index": metadata.get("chunk_index"),
        "namespace": metadata.get("namespace", "user"),
        "corpus_id": metadata.get("corpus_id") or None,
        "token_count": metadata.get("token_count"),
        "content": content,
        "char_count": len(content),
    }


def _diagnose_chunks(chunks: List[dict]) -> dict:
    token_counts = [
        int(chunk.get("metadata", {}).get("token_count") or 0)
        for chunk in chunks
    ]
    duplicate_pairs = []
    normalized = [
        (chunk.get("chunk_id"), _normalize_for_similarity(chunk.get("content", "")))
        for chunk in chunks
    ]
    for idx, (left_id, left_content) in enumerate(normalized):
        if not left_content:
            continue
        for right_id, right_content in normalized[idx + 1:]:
            if not right_content:
                continue
            ratio = SequenceMatcher(None, left_content[:1200], right_content[:1200]).ratio()
            if ratio >= 0.9:
                duplicate_pairs.append({
                    "left_chunk_id": left_id,
                    "right_chunk_id": right_id,
                    "similarity": round(ratio, 3),
                })

    empty_chunks = [
        chunk.get("chunk_id")
        for chunk in chunks
        if not chunk.get("content", "").strip()
    ]
    too_short = [
        chunk.get("chunk_id")
        for chunk in chunks
        if int(chunk.get("metadata", {}).get("token_count") or 0) < settings.chunk_min_tokens // 2
    ]
    too_long = [
        chunk.get("chunk_id")
        for chunk in chunks
        if int(chunk.get("metadata", {}).get("token_count") or 0) > settings.chunk_max_tokens
    ]

    code_block_cut_count = 0
    for chunk in chunks:
        if chunk.get("content", "").count("```") % 2 == 1:
            code_block_cut_count += 1

    return {
        "chunk_count": len(chunks),
        "avg_chunk_tokens": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
        "min_chunk_tokens": min(token_counts) if token_counts else 0,
        "max_chunk_tokens": max(token_counts) if token_counts else 0,
        "too_short_chunk_count": len(too_short),
        "too_long_chunk_count": len(too_long),
        "empty_chunk_count": len(empty_chunks),
        "duplicate_pair_count": len(duplicate_pairs),
        "duplicate_pairs": duplicate_pairs[:20],
        "code_block_cut_count": code_block_cut_count,
        "too_short_chunk_ids": too_short[:20],
        "too_long_chunk_ids": too_long[:20],
        "empty_chunk_ids": empty_chunks[:20],
    }


def _diagnose_chunks_fast(chunks: List[dict]) -> dict:
    token_counts = [
        int(chunk.get("metadata", {}).get("token_count") or 0)
        for chunk in chunks
    ]
    empty_chunks = [
        chunk.get("chunk_id")
        for chunk in chunks
        if not chunk.get("content", "").strip()
    ]
    too_short = [
        chunk.get("chunk_id")
        for chunk in chunks
        if int(chunk.get("metadata", {}).get("token_count") or 0) < settings.chunk_min_tokens // 2
    ]
    too_long = [
        chunk.get("chunk_id")
        for chunk in chunks
        if int(chunk.get("metadata", {}).get("token_count") or 0) > settings.chunk_max_tokens
    ]

    code_block_cut_count = sum(
        1
        for chunk in chunks
        if chunk.get("content", "").count("```") % 2 == 1
    )

    return {
        "chunk_count": len(chunks),
        "avg_chunk_tokens": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
        "min_chunk_tokens": min(token_counts) if token_counts else 0,
        "max_chunk_tokens": max(token_counts) if token_counts else 0,
        "too_short_chunk_count": len(too_short),
        "too_long_chunk_count": len(too_long),
        "empty_chunk_count": len(empty_chunks),
        "duplicate_pair_count": 0,
        "duplicate_pairs": [],
        "code_block_cut_count": code_block_cut_count,
        "too_short_chunk_ids": too_short[:20],
        "too_long_chunk_ids": too_long[:20],
        "empty_chunk_ids": empty_chunks[:20],
        "diagnostics_mode": "fast",
        "duplicate_scan_skipped": True,
    }


def _diagnose_chunks_auto(chunks: List[dict]) -> dict:
    if len(chunks) > _FULL_DIAGNOSTICS_CHUNK_LIMIT:
        return _diagnose_chunks_fast(chunks)
    diagnostics = _diagnose_chunks(chunks)
    diagnostics["diagnostics_mode"] = "full"
    diagnostics["duplicate_scan_skipped"] = False
    return diagnostics


def _hash_content(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _delete_document_artifacts(doc_id: str) -> dict:
    """Delete indexed data, uploads, and query cache for a document."""
    vector_deleted = vector_store.delete_document(doc_id)
    bm25_deleted = bm25_search.remove_document(doc_id)

    files_deleted = 0
    for f in settings.uploads_dir.glob(f"{doc_id}_*"):
        f.unlink()
        files_deleted += 1
        logger.debug(f"删除文件: {f}")

    # Query cache keys are hashed by question/document. For a demo app, clearing all
    # cached answers avoids stale citations after upload/delete without extra index state.
    _query_cache.clear()

    return {
        "vector_chunks_deleted": vector_deleted,
        "bm25_chunks_deleted": bm25_deleted,
        "files_deleted": files_deleted,
    }


def _find_existing_document(
    file_name: str,
    file_size: int,
    file_hash: str,
    namespace: str = "user",
    corpus_id: Optional[str] = None,
) -> Optional[str]:
    """Find a previously uploaded copy of the same file."""
    for doc_id, doc_info in _documents_store.items():
        if doc_info.namespace != namespace:
            continue
        if (doc_info.corpus_id or None) != (corpus_id or None):
            continue
        if doc_info.file_hash and doc_info.file_hash == file_hash:
            return doc_id
        if not doc_info.file_hash and doc_info.name == file_name and doc_info.file_size == file_size:
            return doc_id
    return None


def _index_document_content(
    *,
    content: bytes,
    file_name: str,
    doc_id: str,
    namespace: str,
    corpus_id: Optional[str] = None,
) -> DocumentInfo:
    """Extract, chunk, embed, and index one document."""
    file_ext = Path(file_name).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型，支持: {', '.join(SUPPORTED_EXTENSIONS)}")

    pages: List[Tuple[int, str]]
    total_pages: int

    if file_ext == ".pdf":
        logger.debug("开始提取PDF文本")
        processor = PDFProcessor(use_ocr=settings.use_ocr, ocr_lang=settings.ocr_lang)
        extraction_result = processor.extract_from_bytes(content, file_name)
        pages = processor.get_page_texts(extraction_result)
        total_pages = extraction_result.total_pages
        logger.info(f"PDF提取完成: {total_pages} 页")
    else:
        logger.debug(f"开始提取{file_ext}文本")
        text_processor = TextProcessor()
        extraction_result = text_processor.extract_from_bytes(content, file_name)
        pages = text_processor.get_page_texts(extraction_result)
        total_pages = extraction_result.total_pages
        logger.info(f"文本提取完成: {total_pages} 页")

    logger.debug("开始语义分块")
    chunker = SemanticChunker()
    chunk_results = chunker.chunk_pages(pages)
    logger.info(f"分块完成: {len(chunk_results)} 个分块")

    chunks = []
    texts = []
    metadatas = []
    chunk_ids = []

    for cr in chunk_results:
        chunk_id = str(uuid.uuid4())
        chunk = Chunk(
            id=chunk_id,
            document_id=doc_id,
            content=cr.content,
            page_number=cr.page_number,
            chunk_index=cr.chunk_index,
            token_count=cr.token_count,
            source=file_name,
            namespace=namespace,
            corpus_id=corpus_id,
        )
        chunks.append(chunk)
        texts.append(cr.content)
        metadatas.append({
            "document_id": doc_id,
            "page_number": cr.page_number,
            "chunk_index": cr.chunk_index,
            "token_count": cr.token_count,
            "source": file_name,
            "namespace": namespace,
            "corpus_id": corpus_id or "",
        })
        chunk_ids.append(chunk_id)

    doc_info = DocumentInfo(
        id=doc_id,
        name=file_name,
        chunk_count=len(chunks),
        pages=total_pages,
        file_size=len(content),
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        file_hash=_hash_content(content),
        namespace=namespace,
        corpus_id=corpus_id,
    )
    logger.debug("开始存储到向量数据库")
    vector_store.upsert_document(doc_info)
    vector_store.add_chunks(chunks)
    logger.info("向量存储完成")

    logger.debug("开始建立BM25索引")
    bm25_search.add_documents(texts, metadatas, chunk_ids)
    logger.info("BM25索引建立完成")

    return doc_info


def index_evaluation_document(file_path: Path, corpus_id: str) -> DocumentInfo:
    """Index a local evaluation corpus document without exposing it as user upload."""
    content = file_path.read_bytes()
    file_name = f"{corpus_id}/{file_path.name}"
    file_hash = _hash_content(content)
    existing_doc_id = _find_existing_document(
        file_name=file_name,
        file_size=len(content),
        file_hash=file_hash,
        namespace="evaluation",
        corpus_id=corpus_id,
    )
    if existing_doc_id:
        return _documents_store[existing_doc_id]

    for doc_id, doc_info in list(_documents_store.items()):
        if (
            doc_info.namespace == "evaluation"
            and doc_info.corpus_id == corpus_id
            and doc_info.name == file_name
        ):
            _delete_document_artifacts(doc_id)
            del _documents_store[doc_id]

    doc_id = str(uuid.uuid4())
    doc_info = _index_document_content(
        content=content,
        file_name=file_name,
        doc_id=doc_id,
        namespace="evaluation",
        corpus_id=corpus_id,
    )
    _documents_store[doc_id] = doc_info
    _save_store()
    _query_cache.clear()
    return doc_info


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档（支持 PDF、TXT、MD）

    流程:
    1. 保存文件
    2. 提取文本和页码
    3. 语义分块
    4. 生成embedding并存储
    5. 建立BM25索引
    """
    start_time = time.time()
    logger.info(f"开始处理上传文件: {file.filename}")

    # 获取文件扩展名
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""

    # 验证文件类型
    if file_ext not in SUPPORTED_EXTENSIONS:
        logger.warning(f"文件类型不支持: {file.filename}")
        raise HTTPException(status_code=400, detail=f"不支持的文件类型，支持: {', '.join(SUPPORTED_EXTENSIONS)}")

    content = await file.read()
    await file.seek(0)  # 重置文件指针

    logger.debug(f"读取文件大小: {len(content)} bytes")
    file_hash = _hash_content(content)

    existing_doc_id = _find_existing_document(file.filename, len(content), file_hash)
    if existing_doc_id:
        logger.info(f"检测到重复上传，替换旧文档: {existing_doc_id}")
        _delete_document_artifacts(existing_doc_id)
        del _documents_store[existing_doc_id]
        _save_store()

    # 生成文档ID
    doc_id = str(uuid.uuid4())
    logger.debug(f"生成文档ID: {doc_id}")

    # 保存文件
    file_path = settings.uploads_dir / f"{doc_id}_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(content)
    logger.debug(f"文件已保存: {file_path}")

    try:
        # 1. 根据文件类型提取文本
        pages: List[Tuple[int, str]]
        total_pages: int

        if file_ext == '.pdf':
            logger.debug("开始提取PDF文本")
            processor = PDFProcessor(use_ocr=settings.use_ocr, ocr_lang=settings.ocr_lang)
            extraction_result = processor.extract_from_bytes(content, file.filename)
            pages = processor.get_page_texts(extraction_result)
            total_pages = extraction_result.total_pages
            logger.info(f"PDF提取完成: {total_pages} 页")
        else:
            # txt 或 md 文件
            logger.debug(f"开始提取{file_ext}文本")
            text_processor = TextProcessor()
            extraction_result = text_processor.extract_from_bytes(content, file.filename)
            pages = text_processor.get_page_texts(extraction_result)
            total_pages = extraction_result.total_pages
            logger.info(f"文本提取完成: {total_pages} 页")

        # 2. 语义分块
        logger.debug("开始语义分块")
        chunker = SemanticChunker()
        chunk_results = chunker.chunk_pages(pages)
        logger.info(f"分块完成: {len(chunk_results)} 个分块")

        # 3. 创建Chunk对象
        chunks = []
        texts = []
        metadatas = []
        chunk_ids = []

        for cr in chunk_results:
            chunk_id = str(uuid.uuid4())
            chunk = Chunk(
                id=chunk_id,
                document_id=doc_id,
                content=cr.content,
                page_number=cr.page_number,
                chunk_index=cr.chunk_index,
                token_count=cr.token_count,
                source=file.filename
            )
            chunks.append(chunk)
            texts.append(cr.content)
            metadatas.append({
                "document_id": doc_id,
                "page_number": cr.page_number,
                "chunk_index": cr.chunk_index,
                "token_count": cr.token_count,
                "source": file.filename,
                "namespace": "user",
                "corpus_id": "",
            })
            chunk_ids.append(chunk_id)

        # 4. 存储到向量数据库
        doc_info = DocumentInfo(
            id=doc_id,
            name=file.filename,
            chunk_count=len(chunks),
            pages=total_pages,
            file_size=len(content),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            file_hash=file_hash,
        )
        logger.debug("开始存储到向量数据库")
        vector_store.upsert_document(doc_info)
        vector_store.add_chunks(chunks)
        logger.info("向量存储完成")

        # 5. 建立BM25索引
        logger.debug("开始建立BM25索引")
        bm25_search.add_documents(texts, metadatas, chunk_ids)
        logger.info("BM25索引建立完成")

        # 6. 保存文档信息
        _documents_store[doc_id] = doc_info
        _save_store()  # 持久化
        _query_cache.clear()

        elapsed = time.time() - start_time
        logger.info(f"文档处理完成: {doc_id}, 总耗时: {elapsed:.2f}秒")

        return DocumentUploadResponse(
            success=True,
            data=doc_info
        )

    except (PDFProcessingError, TextProcessingError) as e:
        logger.error(f"文档处理错误: {str(e)}")
        # 清理文件
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"文档处理失败: {str(e)}")
        # 清理
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """获取所有文档列表"""
    logger.debug("获取文档列表")
    documents = [
        document
        for document in _documents_store.values()
        if document.namespace == "user"
    ]
    logger.info(f"返回 {len(documents)} 个文档")
    return DocumentListResponse(
        success=True,
        data=documents
    )


@router.get("/{doc_id}/chunks")
async def get_document_chunks(doc_id: str):
    """查看文档的 chunks。"""
    if doc_id not in _documents_store:
        raise HTTPException(status_code=404, detail="文档不存在")

    chunks = vector_store.get_document_chunks(doc_id)
    chunks.sort(key=lambda item: item.get("metadata", {}).get("chunk_index", 0))
    return {
        "success": True,
        "data": [_chunk_payload(chunk) for chunk in chunks],
    }


@router.get("/{doc_id}/diagnostics")
async def get_document_diagnostics(doc_id: str):
    """返回文档 chunk 质量诊断。"""
    if doc_id not in _documents_store:
        raise HTTPException(status_code=404, detail="文档不存在")

    chunks = vector_store.get_document_chunks(doc_id)
    chunks.sort(key=lambda item: item.get("metadata", {}).get("chunk_index", 0))
    chunk_preview = chunks[:_DIAGNOSTICS_CHUNK_PREVIEW_LIMIT]
    return {
        "success": True,
        "data": {
            "document": _documents_store[doc_id].model_dump(),
            "diagnostics": _diagnose_chunks_auto(chunks),
            "chunks": [_chunk_payload(chunk) for chunk in chunk_preview],
            "chunk_preview_limit": _DIAGNOSTICS_CHUNK_PREVIEW_LIMIT,
            "chunk_preview_truncated": len(chunks) > len(chunk_preview),
        },
    }


@router.get("/{doc_id}", response_model=DocumentUploadResponse)
async def get_document(doc_id: str):
    """获取文档信息"""
    logger.debug(f"获取文档信息: {doc_id}")
    if doc_id not in _documents_store:
        logger.warning(f"文档不存在: {doc_id}")
        raise HTTPException(status_code=404, detail="文档不存在")

    return DocumentUploadResponse(
        success=True,
        data=_documents_store[doc_id]
    )


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档"""
    logger.info(f"删除文档: {doc_id}")
    if doc_id not in _documents_store:
        logger.warning(f"文档不存在: {doc_id}")
        raise HTTPException(status_code=404, detail="文档不存在")

    try:
        cleanup_result = _delete_document_artifacts(doc_id)

        # 从内存存储删除
        del _documents_store[doc_id]
        _save_store()  # 持久化

        logger.info(f"文档删除成功: {doc_id}, cleanup={cleanup_result}")
        return {"success": True, "message": "文档已删除", "data": cleanup_result}
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def cleanup_orphaned_indexes():
    """清理文档元信息中不存在的向量/BM25 孤儿索引。"""
    valid_document_ids = set(_documents_store.keys())
    vector_removed = vector_store.delete_documents_not_in(valid_document_ids)
    bm25_removed = bm25_search.remove_documents_not_in(valid_document_ids)
    _query_cache.clear()

    return {
        "success": True,
        "data": {
            "valid_documents": len(valid_document_ids),
            "vector_chunks_removed": vector_removed,
            "bm25_chunks_removed": bm25_removed,
            "query_cache_cleared": True,
        },
    }
