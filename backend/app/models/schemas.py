"""Pydantic 数据模型"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ==================== 请求模型 ====================

class QueryOptions(BaseModel):
    """查询选项"""
    use_hybrid_search: bool = True
    use_rerank: bool = True
    use_query_rewrite: bool = False
    top_k: int = Field(default=3, ge=1, le=10)
    namespace: str = "user"
    corpus_id: Optional[str] = None


class QueryRequest(BaseModel):
    """问答请求"""
    question: str = Field(..., min_length=1, max_length=2000)
    document_id: Optional[str] = None
    options: Optional[QueryOptions] = QueryOptions()


class EvaluationRequest(BaseModel):
    """评估请求"""
    methods: List[str] = Field(default=["baseline", "hybrid", "hybrid_rerank"])
    test_set_id: Optional[str] = "default"
    document_id: Optional[str] = None


# ==================== 响应模型 ====================

class Source(BaseModel):
    """引用来源"""
    chunk_id: str
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    page_number: int
    chunk_index: Optional[int] = None
    content: str
    score: float = Field(..., ge=0, le=1)


class QueryMetadata(BaseModel):
    """查询元数据"""
    retrieval_method: str
    total_candidates: int
    final_chunks: int
    response_time_seconds: float
    trace_id: Optional[str] = None
    query_rewrite_used: bool = False
    timings: Dict[str, float] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """问答响应"""
    success: bool = True
    data: Dict[str, Any]  # 包含 answer, sources, metadata


class DocumentInfo(BaseModel):
    """文档信息"""
    id: str
    name: str
    chunk_count: int
    pages: int
    file_size: int
    created_at: str  # 使用字符串格式 "YYYY-MM-DD HH:MM:SS"
    file_hash: Optional[str] = None
    namespace: str = "user"
    corpus_id: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    success: bool = True
    data: DocumentInfo


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    success: bool = True
    data: List[DocumentInfo]


class EvaluationRunResponse(BaseModel):
    """评估运行响应"""
    success: bool = True
    data: Dict[str, Any]  # 包含 run_id, results, comparison


class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = False
    error: str
    detail: Optional[str] = None


# ==================== 内部数据模型 ====================

class Chunk(BaseModel):
    """文档分块"""
    id: str
    document_id: str
    content: str
    page_number: int
    chunk_index: int
    token_count: int
    source: str
    namespace: str = "user"
    corpus_id: Optional[str] = None
    embedding: Optional[List[float]] = None

    class Config:
        from_attributes = True


class Document(BaseModel):
    """文档"""
    id: str
    name: str
    file_size: int
    chunk_count: int
    pages: int
    created_at: datetime
    chunks: List[Chunk] = []

    class Config:
        from_attributes = True
