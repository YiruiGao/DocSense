"""应用配置"""
import os
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置类"""

    # 应用基础配置
    app_name: str = "RAG Backend"
    debug: bool = False

    # OCR 配置
    use_ocr: bool = False
    ocr_lang: str = "chi_sim+eng"

    # HuggingFace 配置
    hf_hub_offline: bool = True  # 强制使用本地缓存，避免每次访问 HuggingFace

    # 路径配置
    backend_dir: Path = Path(__file__).resolve().parents[2]
    base_dir: Path = Path(__file__).resolve().parents[3]
    data_dir: Path = base_dir / "data"
    uploads_dir: Path = data_dir / "uploads"
    eval_corpora_dir: Path = data_dir / "eval_corpora"
    cache_dir: Path = data_dir / "cache"
    logs_dir: Path = data_dir / "logs"

    # 日志配置
    log_level: str = "INFO"
    log_dir: Optional[Path] = None

    # LLM 配置
    llm_provider: str = "zai"
    llm_timeout_seconds: float = 30.0
    llm_max_tokens: int = 800
    llm_context_chars_per_chunk: int = 1200

    # z.ai API 配置
    zai_api_key: Optional[str] = None
    zai_base_url: str = "https://api.z.ai/api/coding/paas/v4"
    zai_model: str = "glm-5"

    # DeepSeek API 配置
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_reasoning_effort: str = "high"
    deepseek_thinking_enabled: bool = True

    # Embedding 配置
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_cache_size: int = 1000

    # 检索配置
    default_top_k: int = 10
    final_top_k: int = 3
    vector_weight: float = 0.6  # 混合检索中向量的权重
    retrieval_candidate_multiplier: int = 5
    duplicate_similarity_threshold: float = 0.82
    rerank_min_relevance_score: float = 0.05
    overview_context_chunks: int = 4

    # 分块配置
    chunk_min_tokens: int = 300
    chunk_max_tokens: int = 500
    chunk_overlap: int = 50

    # Rerank 配置
    rerank_model: str = "BAAI/bge-reranker-base"

    # Postgres 连接配置
    postgres_dsn: Optional[str] = None
    postgres_pool_min: int = 1
    postgres_pool_max: int = 10
    archived_retention_days: int = 7

    # 服务配置
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # Pydantic v2 配置：指定 .env 文件路径，extra="ignore" 忽略未定义的字段
    model_config = {
        "env_file": str(Path(__file__).resolve().parents[2] / ".env"),
        "extra": "ignore",
    }

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value):
        """Accept common deployment-mode strings for DEBUG."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production", "false", "0", "no"}:
                return False
            if normalized in {"debug", "dev", "development", "true", "1", "yes"}:
                return True
        return value

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 设置第三方库的环境变量
        if self.hf_hub_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"

        # 确保目录存在
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.eval_corpora_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if self.log_dir is None:
            self.log_dir = self.logs_dir
        elif not self.log_dir.is_absolute():
            self.log_dir = self.base_dir / self.log_dir


settings = Settings()
