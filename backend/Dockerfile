FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 创建数据目录
RUN mkdir -p /app/data/uploads /app/data/chroma /app/data/cache /app/data/logs

# 下载 embedding 模型(构建时下载，运行时加载)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
    model.save('/app/data/embedding_model')" || echo "Model download failed, but continuing..."

# 下载 rerank 模型
RUN python -c "from FlagEmbedding import FlagReranker; \
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True); \
    reranker.save('/app/data/reranker_model')" || echo "Reranker download failed, but continuing..."
# 复制应用代码
COPY app ./app

# 暴露端口
EXPOSE 8000

# 启动应用
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
