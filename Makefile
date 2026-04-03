# RAG 项目 Makefile
# 使用方法: make <命令>

.PHONY: help install dev build clean clean-cache db-reset chroma-reset frontend-check backend-static backend-smoke test-unit test-component test-integration ci

DATA_DIR ?= data
LEGACY_BACKEND_DATA_DIR ?= backend/data

# 默认显示帮助
help:
	@echo "RAG 项目命令:"
	@echo ""
	@echo "  make install        安装所有依赖"
	@echo "  make dev            启动开发环境（前后端同时启动）"
	@echo "  make dev-frontend   仅启动前端"
	@echo "  make dev-backend    仅启动后端"
	@echo "  make build          构建生产版本"
	@echo "  make docker         Docker 启动（生产）"
	@echo "  make docker-stop    Docker 停止"
	@echo "  make clean-cache    清理 Python/工具缓存"
	@echo "  make clean          清理缓存和前端构建产物"
	@echo "  make db-reset       重置全部本地数据（Chroma/BM25/文档元信息/上传文件）"
	@echo "  make chroma-reset   仅重置 Chroma 向量数据库"
	@echo "  make frontend-check 运行前端 lint 和生产构建"
	@echo "  make backend-static 运行后端静态检查"
	@echo "  make backend-smoke  导入后端 FastAPI 应用，验证依赖装配"
	@echo "  make test-unit      运行后端单元测试"
	@echo "  make test-component 运行后端组件测试"
	@echo "  make test-integration 运行后端集成测试"
	@echo "  make ci            运行本地 CI 基础门禁"
	@echo "  make check          检查环境"
	@echo ""

# 安装依赖
install:
	@echo "安装前端依赖..."
	cd frontend && bun install
	@echo "安装后端依赖..."
	cd backend && .venv/bin/pip install -r requirements.txt 2>/dev/null || pip install -r requirements.txt
	@echo "✅ 依赖安装完成"

# 开发环境（同时启动前后端）
dev:
	@echo "启动开发环境..."
	@trap 'kill 0' INT; \
	(cd frontend && bun dev) & \
	(cd backend && .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000) & \
	wait

dev-frontend:
	@echo "启动前端 (http://localhost:3000)..."
	cd frontend && bun dev

dev-backend:
	@echo "启动后端 (http://localhost:8000)..."
	cd backend && .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 生产构建
build:
	@echo "构建前端..."
	cd frontend && bun run build
	@echo "构建后端..."
	cd backend && .venv/bin/pip install -r requirements.txt 2>/dev/null || pip install -r requirements.txt
	@echo "✅ 构建完成"

# Docker
docker:
	docker-compose up --build -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# 清理
clean-cache:
	@echo "清理 Python 和工具缓存..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f backend/.coverage .coverage
	rm -rf backend/htmlcov htmlcov
	@echo "✅ 缓存清理完成"

clean: clean-cache
	@echo "清理前端构建缓存..."
	rm -rf frontend/.next
	rm -rf frontend/node_modules/.cache
	@echo "✅ 清理完成"

# 数据库
db-reset:
	@echo "重置全部本地数据..."
	@echo "请先停止后端服务，避免 Chroma/BM25 文件仍被进程占用。"
	rm -rf "$(DATA_DIR)/chroma" "$(DATA_DIR)/cache" "$(DATA_DIR)/uploads"
	rm -rf "$(LEGACY_BACKEND_DATA_DIR)/chroma" "$(LEGACY_BACKEND_DATA_DIR)/cache" "$(LEGACY_BACKEND_DATA_DIR)/uploads"
	mkdir -p "$(DATA_DIR)/chroma" "$(DATA_DIR)/cache" "$(DATA_DIR)/uploads"
	@echo "✅ 本地数据已重置"

chroma-reset:
	@echo "重置 Chroma 向量数据库..."
	@echo "请先停止后端服务，避免 Chroma 文件仍被进程占用。"
	rm -rf "$(DATA_DIR)/chroma"
	rm -rf "$(LEGACY_BACKEND_DATA_DIR)/chroma"
	mkdir -p "$(DATA_DIR)/chroma"
	@echo "✅ Chroma 已重置"

# 检查环境
check:
	@echo "检查环境..."
	@echo "Python: $$(python3 --version 2>/dev/null || echo '未安装')"
	@echo "Bun: $$(bun --version 2>/dev/null || echo '未安装')"
	@echo "Docker: $$(docker --version 2>/dev/null || echo '未安装')"
	@echo ""
	@echo "检查配置文件..."
	@test -d backend/.venv && echo "✅ backend/.venv 存在" || echo "❌ 运行: make venv"
	@test -f backend/.env && echo "✅ backend/.env 存在" || echo "❌ 请创建 backend/.env"
	@test -f frontend/node_modules/.bin/next && echo "✅ 前端依赖已安装" || echo "❌ 运行: make install"
	@echo ""
	@echo "✅ 检查完成"

# 快速测试
test-api:
	@echo "测试后端 API..."
	curl -s http://localhost:8000/health | python3 -m json.tool || echo "后端未启动"

frontend-check:
	@echo "运行前端 lint..."
	cd frontend && bun run lint
	@echo "构建前端生产版本..."
	cd frontend && bun run build

backend-static:
	@echo "运行后端静态检查..."
	cd backend && (.venv/bin/python -m compileall -q app tests || python -m compileall -q app tests)
	cd backend && (.venv/bin/ruff check app tests || ruff check app tests)

backend-smoke:
	@echo "导入后端 FastAPI 应用..."
	cd backend && (.venv/bin/python -c "from app.main import app; print(app.title)" || python -c "from app.main import app; print(app.title)")

test-unit:
	@echo "运行后端单元测试..."
	cd backend && (.venv/bin/python -m pytest tests/unit || python -m pytest tests/unit)

test-component:
	@echo "运行后端组件测试..."
	cd backend && (.venv/bin/python -m pytest tests/component || python -m pytest tests/component)

test-integration:
	@echo "运行后端集成测试..."
	cd backend && (.venv/bin/python -m pytest tests/integration || python -m pytest tests/integration)

ci: backend-static backend-smoke test-unit test-component frontend-check
	@echo "✅ 本地 CI 基础门禁完成"

# 安装 Python 虚拟环境
venv:
	@echo "创建 Python 虚拟环境..."
	cd backend && python3 -m venv .venv
	@echo "✅ 虚拟环境已创建"
	@echo "下一步: make install"
