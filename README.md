# DocSense

DocSense is a local document question-answering and RAGOps workbench. It pairs a
FastAPI backend with a Next.js frontend so you can upload documents, ask
grounded questions, inspect retrieval behavior, and run repeatable RAG quality
evaluations.

The project is built for practical RAG iteration: chunking, hybrid retrieval,
reranking, context selection, answer generation, trace inspection, evaluation,
and bad-case analysis.

## Problem

RAG prototypes often fail in ways that are hard to diagnose:

- A generated answer looks plausible, but the supporting chunks are weak.
- Vector search misses exact terms, while keyword search misses semantic matches.
- Chunking changes improve one document but regress another.
- Retrieval quality, latency, and generation quality are mixed together, making
  it difficult to know what actually broke.
- Bad answers are noticed manually, but not captured as future regression cases.

DocSense addresses this by treating document QA as an observable workflow rather
than a black-box chat UI. Every query can expose retrieval candidates, selected
context, timings, sources, and traces; every weak case can be turned into a
bad-case record or evaluation case.

## Architecture

```text
User
  │
  ▼
Next.js frontend
  ├─ Document upload and management
  ├─ Document-scoped or full-library chat
  └─ RAGOps dashboard for traces, evaluations, and bad cases
  │
  ▼
FastAPI backend
  ├─ /documents    Upload, parse, chunk, index, list, delete
  ├─ /chat         Retrieve, rerank, generate, cite, trace
  ├─ /evaluation   Datasets, cases, evaluation runs, metrics
  └─ /ops          Dashboard summaries, traces, bad cases
  │
  ├─ Ingestion
  │   ├─ PDF / Markdown / text extraction
  │   └─ token-aware semantic chunking with metadata
  │
  ├─ Retrieval
  │   ├─ Chroma vector search
  │   ├─ BM25 lexical search
  │   ├─ Reciprocal Rank Fusion hybrid search
  │   └─ optional cross-encoder reranking
  │
  ├─ Generation
  │   └─ OpenAI-compatible chat completion providers
  │
  └─ Local runtime state
      ├─ uploaded files
      ├─ Chroma vector data
      ├─ BM25 index cache
      ├─ document metadata
      ├─ query traces
      ├─ evaluation runs
      └─ bad-case records
```

### Query Flow

1. The frontend sends a question, optional document scope, and retrieval options.
2. The backend checks the local query cache.
3. Retrieval runs through vector search, BM25, or hybrid search.
4. Hybrid search fuses vector and BM25 rankings with Reciprocal Rank Fusion.
5. Optional reranking reorders candidates with a reranker model.
6. Near-duplicate chunks are removed before context selection.
7. The LLM receives only selected document chunks and is instructed to cite
   sources.
8. The response returns the answer, citations, source previews, timings, and a
   trace ID.
9. RAGOps APIs persist traces for later inspection and bad-case capture.

### Ingestion Flow

1. A PDF, Markdown, or plain text file is uploaded to the backend.
2. The backend validates the file type and hashes content to detect duplicates.
3. Text is extracted with page and source metadata.
4. The semantic chunker creates token-bounded chunks with overlap.
5. Chunks are persisted into Chroma for vector retrieval.
6. The same chunks are indexed into a local BM25 index.
7. Document metadata is stored locally for listing, deletion, diagnostics, and
   evaluation scoping.

## Core Features

### Document QA

- Upload and index PDF, Markdown, and plain text documents.
- Ask questions against one selected document or the full local library.
- Generate grounded answers with source citations.
- Inspect cited chunks, chunk indexes, page numbers, scores, and response
  metadata.

### Hybrid Retrieval

- Vector retrieval for semantic matching.
- BM25 retrieval with `jieba` tokenization for lexical matching.
- Reciprocal Rank Fusion to combine vector and lexical candidates.
- Optional reranking for better final context ordering.
- Duplicate-context filtering to reduce repeated evidence sent to the LLM.

### RAGOps Dashboard

- Online quality summary: trace count, success rate, latency, candidates,
  selected chunks, token/cost fields when available.
- Knowledge-base diagnostics: document count, chunk count, short/long chunks,
  empty chunks, duplicate pairs, and code-block split issues.
- Trace explorer for retrieval stages, selected context, spans, and answers.
- Bad-case tracking with failure type, severity, expected behavior, and status.

### Offline Evaluation

- Create and manage evaluation datasets and test cases.
- Compare retrieval methods such as baseline, hybrid, and hybrid plus rerank.
- Track metrics including `hit@3`, `hit@5`, `hit@10`, MRR, latency, and errors.
- Persist evaluation runs for later comparison.

## Technical Decisions

### FastAPI Backend, Next.js Frontend

The backend owns ingestion, retrieval, generation, evaluation, and local
persistence because those workflows depend on Python ML/RAG libraries. The
frontend stays focused on interaction: document management, chat, source
inspection, and RAGOps views.

### Postgres + pgvector for Vector Storage

Chunks are stored in Postgres with an HNSW cosine index via pgvector. Using
Postgres as the single storage backend means ACL filtering, revision tracking,
and soft delete all live in the same transactional system as document metadata,
with no second store to keep in sync.

### BM25 Alongside Vector Search

Vector search handles semantic similarity, but exact terms, API names, security
categories, and configuration keys often need lexical matching. BM25 complements
embeddings and gives the system a strong baseline for technical documents.

### Reciprocal Rank Fusion for Hybrid Search

The hybrid retriever uses Reciprocal Rank Fusion instead of directly mixing raw
scores from different retrieval systems. Rank-based fusion avoids assuming that
vector distances and BM25 scores are calibrated on the same scale.

### Optional Reranking

Reranking is separated from first-stage retrieval. This keeps candidate recall
and final context precision independently tunable: retrieve broadly, then spend
reranker compute only on a smaller candidate set.

### Trace Everything Useful

Each query can record candidates, final context, timings, metadata, answer, and
sources. This makes debugging answer quality possible without reproducing the
exact same interaction from memory.

### Evaluation as a First-Class Workflow

The evaluation APIs and RAGOps dashboard are part of the application rather than
external scripts. This makes it easier to compare retrieval methods, capture
regressions, and turn bad user-visible behavior into repeatable test cases.

## Stack

### Backend

- Python 3.11
- FastAPI
- Postgres + pgvector for vector storage (HNSW cosine index)
- SentenceTransformers embeddings
- BM25 lexical retrieval with `rank-bm25` and `jieba`
- FlagEmbedding reranker
- Pydantic settings and schemas
- Ruff and pytest for static checks and tests

### Frontend

- Next.js 16 App Router
- React 19
- TypeScript
- Tailwind CSS
- shadcn/ui-style components
- Bun

### Infrastructure

- Docker Compose for local production-style startup
- GitHub Actions CI for frontend checks and backend tests
- Makefile targets for common local workflows

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
│   │   ├── models/        # API schemas and shared data models
│   │   ├── ops/           # traces and bad-case records
│   │   ├── retrieval/     # vector, BM25, hybrid search, reranking
│   │   └── utils/         # local cache helpers
│   ├── tests/
│   ├── requirements.txt
│   └── requirements-dev.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx       # main document QA UI
│   │   └── ragops/        # RAGOps dashboard
│   ├── components/
│   └── package.json
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

POSTGRES_DSN=postgresql://docsense:docsense@localhost:5432/docsense
```

Set `LLM_PROVIDER` to the provider you want to use. The repository currently
includes provider configuration examples for z.ai and DeepSeek.

Start the bundled Postgres service before running the backend:

```bash
docker compose up -d postgres
make dev
```

The schema in `backend/sql/schema.sql` is applied automatically on first connection.

Runtime data is stored under `data/` by default:

```text
data/
├── uploads/       # uploaded source files
├── chroma/        # persistent Chroma vector database
├── cache/         # BM25 index, metadata, traces, evaluations, bad cases
└── logs/          # backend logs
```

## Typical Workflow

1. Start the app with `make dev`.
2. Upload your own PDF, Markdown, or text documents.
3. Ask questions in document-scoped mode or full-library mode.
4. Inspect cited sources, retrieval metadata, and trace IDs.
5. Open the RAGOps dashboard to review traces and quality summaries.
6. Add bad cases or evaluation records when behavior is weak.
7. Compare retrieval methods through evaluation runs.
8. Run `make ci` before pushing changes.
