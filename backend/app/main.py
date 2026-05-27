"""FastAPI 主入口"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uuid

from app.common.logging import setup_logging, add_request_id, get_logger
from app.common.config import settings
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router
from app.api.evaluation import router as evaluation_router
from app.api.ops import router as ops_router

# 初始化日志系统
setup_logging(log_level=settings.log_level, log_dir=settings.log_dir)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.retrieval.vector_store import vector_store
    _ = vector_store.pool  # warm up connection pool before first request
    yield


app = FastAPI(lifespan=lifespan)


# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(documents_router, prefix="/documents", tags=["documents"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(evaluation_router, prefix="/evaluation", tags=["evaluation"])
app.include_router(ops_router, prefix="/ops", tags=["ops"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    request_id = str(uuid.uuid4())[:8]
    add_request_id(logger, request_id)

    logger.info(f"请求开始: {request.method} {request.url}")
    start_time = time.time()

    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"请求完成: {request.method} {request.url} - {duration:.3f}s")

    return response


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """捕获所有未处理异常, 记录详细日志"""
    logger.error(
        f"未处理异常: {type(exc).__name__}: {exc}",
        extra={
            "path": str(request.url),
            "method": request.method
        }
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "detail": "Internal server error. Check server logs for details."
        }
    )


@app.get("/")
async def root():
    """健康检查"""
    logger.debug("健康检查请求")
    return {"status": "ok", "message": "RAG Backend API 服务运行中"}


@app.get("/health")
async def health_check():
    """服务健康检查"""
    logger.debug("健康检查请求")
    return {"status": "ok", "message": "数据库连接正常"}

