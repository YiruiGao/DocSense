# DocSense

DocSense is a document question-answering and RAGOps workbench for local
knowledge-base experiments. It provides a FastAPI backend for document
ingestion, hybrid retrieval, answer generation, tracing, and evaluation, plus a
Next.js frontend for uploading documents, asking questions, and inspecting RAG
quality.

The project is intended for iterating on practical RAG behavior: chunking,
retrieval, reranking, context selection, answer quality, and bad-case analysis.

## Features

- Upload and index PDF, Markdown, and plain text documents.
- Split documents into semantic chunks with persisted metadata.
- Search with vector retrieval, BM25, hybrid ranking, and reranking.
- Ask document-grounded questions with LLM-generated answers and source citations.
- Inspect retrieval candidates, selected context, and request traces.
- Run RAGOps evaluation datasets for retrieval and generation methods.
- Track bad cases for later analysis and regression testing.
- Run local checks through Make targets and GitHub Actions.

## Stack

### Backend

- Python 3.11
- FastAPI
- Chroma-backed vector storage
- BM25 lexical retrieval
- Ruff and pytest for static checks and tests

### Frontend

- Next.js 16 App Router
- React 19
- TypeScript
- Tailwind CSS
- Bun

### Infrastructure

- Docker Compose for local production-style startup
- GitHub Actions CI for frontend checks and backend tests

## Project Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes
│   │   ├── common/        # settings and logging
│   │   ├── evaluation/    # RAGOps datasets, runs, evaluators, metrics
│   │   ├── generation/    # LLM provider integration
│   │   ├── ingestion/     # PDF/text parsing and chunking
│   │   ├── ops/           # traces and bad-case records
│   │   └── retrieval/     # vector, BM25, hybrid search, reranking
│   ├── tests/
│   ├── requirements.txt
│   └── requirements-dev.txt
├── frontend/
│   └── app/
│       ├── page.tsx       # main document QA UI
│       └── ragops/        # RAGOps dashboard
├── .github/workflows/
│   └── ci.yml
├── docker-compose.yml
└── Makefile
```

## Prerequisites

- Python 3.11
- Bun
- Docker, optional
- An LLM API key for your selected provider

## Configuration

Create the backend environment file:

```bash
cp backend/.env.example backend/.env
```

Important settings:

```env
LLM_PROVIDER=zai
LLM_TIMEOUT_SECONDS=30
LLM_MAX_TOKENS=800
LLM_CONTEXT_CHARS_PER_CHUNK=1200

ZAI_API_KEY=...
ZAI_BASE_URL=https://api.z.ai/api/coding/paas/v4
ZAI_MODEL=glm-5

DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

Set `LLM_PROVIDER` to the provider you want to use. The repository currently
includes provider configuration examples for z.ai and DeepSeek.

Runtime data is stored under `data/` by default, including uploads, Chroma data,
cache files, and logs.

## Local Development

Install dependencies:

```bash
make install
```

Start frontend and backend together:

```bash
make dev
```

Then open:

```text
http://localhost:3000
```

Run only the frontend:

```bash
make dev-frontend
```

Run only the backend:

```bash
make dev-backend
```

Backend API health checks:

```text
http://localhost:8000/
http://localhost:8000/health
```

## Docker

Start both services:

```bash
make docker
```

Stop services:

```bash
make docker-stop
```

View logs:

```bash
make docker-logs
```

## Quality Gates

Run the local CI gate:

```bash
make ci
```

Individual checks:

```bash
make frontend-check
make backend-static
make backend-smoke
make test-unit
make test-component
make test-integration
```

The GitHub Actions workflow in `.github/workflows/ci.yml` runs frontend lint and
build checks, backend static checks, backend unit tests, backend component tests,
and a FastAPI import smoke test on pull requests and pushes to `main` or
`master`.

## Data Reset

Reset all local indexed data:

```bash
make db-reset
```

Reset only the Chroma vector database:

```bash
make chroma-reset
```

These commands remove local runtime data. Stop the backend before running them
so database files are not held open by a running process.

## Typical Workflow

1. Start the app with `make dev`.
2. Upload documents in the main UI.
3. Ask questions and inspect cited sources.
4. Review retrieval traces and candidate chunks.
5. Add bad cases or evaluation records when behavior is weak.
6. Run `make ci` before pushing changes.

## Notes

- Uploaded documents and indexes are local runtime state, not source code.
- Keep API keys in `backend/.env`; do not commit secrets.
- The frontend expects the backend at `http://localhost:8000` during local
  development.
