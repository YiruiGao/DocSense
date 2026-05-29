.PHONY: help venv install db db-stop db-reset dev dev-frontend dev-backend \
        build docker docker-stop docker-logs \
        clean clean-cache check \
        frontend-check backend-static backend-smoke \
        test-unit test-component test-integration ci

DATA_DIR ?= data

# ── 帮助 ──────────────────────────────────────────────────────────────────────
help:
	@echo "开发流程:"
	@echo "  make venv             创建 Python 虚拟环境"
	@echo "  make install          安装前后端依赖"
	@echo "  make db               启动 Postgres（后台，仅首次或重启后需要）"
	@echo "  make db-stop          停止 Postgres"
	@echo "  make db-reset         清空 Postgres 数据 + BM25/上传文件（不可恢复）"
	@echo "  make dev              本地启动前后端（前提：make db 已运行）"
	@echo "  make dev-backend      仅启动后端"
	@echo "  make dev-frontend     仅启动前端"
	@echo ""
	@echo "构建 & 部署:"
	@echo "  make build            构建前端生产版本"
	@echo "  make docker           docker compose 完整部署（含 Postgres + backend + frontend）"
	@echo "  make docker-stop      停止所有容器"
	@echo "  make docker-logs      跟踪容器日志"
	@echo ""
	@echo "检查 & 测试:"
	@echo "  make check            检查本地环境（依赖/配置）"
	@echo "  make backend-static   后端静态检查（compile + ruff）"
	@echo "  make backend-smoke    后端 import 验证"
	@echo "  make frontend-check   前端 lint + 生产构建"
	@echo "  make test-unit        单元测试"
	@echo "  make test-component   组件测试"
	@echo "  make test-integration 集成测试（自动起停 test DB）"
	@echo "  make ci               本地 CI 门禁（静态检查 + smoke + 单元 + 前端）"
	@echo "  make clean            清理缓存和前端构建产物"

# ── 环境 ─────────────────────────────────────────────────────────────────────
venv:
	cd backend && python3 -m venv .venv
	@echo "虚拟环境已创建，下一步: make install"

install:
	cd frontend && bun install
	cd backend && (.venv/bin/pip install -r requirements.txt 2>/dev/null || pip install -r requirements.txt)

check:
	@echo "Python:  $$(python3 --version 2>/dev/null || echo '未安装')"
	@echo "Bun:     $$(bun --version 2>/dev/null || echo '未安装')"
	@echo "Docker:  $$(docker --version 2>/dev/null || echo '未安装')"
	@test -d backend/.venv      && echo "✅ backend/.venv"       || echo "❌ 运行: make venv"
	@test -f backend/.env       && echo "✅ backend/.env"        || echo "❌ 请创建 backend/.env（参考 .env.example）"
	@test -f frontend/node_modules/.bin/next && echo "✅ 前端依赖" || echo "❌ 运行: make install"

# ── 数据库 ────────────────────────────────────────────────────────────────────
db:
	docker compose up -d postgres

db-stop:
	docker compose stop postgres

db-reset:
	@echo "警告：将清空 Postgres 数据、BM25 索引和上传文件，确认请按 Enter，Ctrl-C 取消"
	@read _
	docker compose stop postgres
	docker compose rm -f postgres
	docker volume rm docsense_pgdata 2>/dev/null || true
	rm -rf "$(DATA_DIR)/cache" "$(DATA_DIR)/uploads"
	mkdir -p "$(DATA_DIR)/cache" "$(DATA_DIR)/uploads"
	@echo "已重置，重新运行: make db"

# ── 开发 ─────────────────────────────────────────────────────────────────────
dev:
	@trap 'kill 0' INT; \
	(cd frontend && bun dev) & \
	(cd backend && .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000) & \
	wait

dev-backend:
	cd backend && .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && bun dev

# ── 构建 & 部署 ───────────────────────────────────────────────────────────────
build:
	cd frontend && bun run build

docker:
	docker compose up --build -d

docker-stop:
	docker compose down

docker-logs:
	docker compose logs -f

# ── 清理 ─────────────────────────────────────────────────────────────────────
clean-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f backend/.coverage .coverage
	rm -rf backend/htmlcov htmlcov

clean: clean-cache
	rm -rf frontend/.next frontend/node_modules/.cache

# ── 静态检查 ──────────────────────────────────────────────────────────────────
backend-static:
	cd backend && (.venv/bin/python -m compileall -q app tests || python -m compileall -q app tests)
	cd backend && (.venv/bin/ruff check app tests || ruff check app tests)

backend-smoke:
	cd backend && (.venv/bin/python -c "from app.main import app; print(app.title)" || python -c "from app.main import app; print(app.title)")

frontend-check:
	cd frontend && bun run lint
	cd frontend && bun run build

# ── 测试 ─────────────────────────────────────────────────────────────────────
test-unit:
	cd backend && (.venv/bin/python -m pytest tests/unit || python -m pytest tests/unit)

test-component:
	cd backend && (.venv/bin/python -m pytest tests/component || python -m pytest tests/component)

test-integration:
	docker compose up -d --wait postgres-test
	@export POSTGRES_DSN="postgresql://test:test@localhost:5433/testdb"; \
	( cd backend && \
	  (.venv/bin/python -m pytest tests/integration -v || python -m pytest tests/integration -v) \
	); STATUS=$$?; docker compose stop postgres-test; exit $$STATUS

ci: backend-static backend-smoke test-unit test-component frontend-check
	@echo "✅ CI 完成"
