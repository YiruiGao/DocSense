-- DocSense Postgres + pgvector schema.
-- Applied automatically on first connection via PgVectorStore._ensure_schema.

-- =============================================================================
-- Extensions
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =============================================================================
-- Table: documents
-- =============================================================================
-- One row per (logical document, revision). Same source_uri across revisions
-- forms a chain via superseded_by. Retrieval defaults to status='published'.

CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    name            TEXT        NOT NULL,
    source_type     TEXT        NOT NULL,
    source_uri      TEXT        NOT NULL,
    file_hash       TEXT        NOT NULL,

    file_size       BIGINT      NOT NULL,
    pages           INT,
    chunk_count     INT         NOT NULL DEFAULT 0,

    revision        INT         NOT NULL DEFAULT 1,
    status          TEXT        NOT NULL DEFAULT 'published'
                    CHECK (status IN ('draft', 'published', 'archived')),
    published_at    TIMESTAMPTZ,
    superseded_by   UUID        REFERENCES documents(id) ON DELETE SET NULL,

    -- Empty array = visible to all users in this deployment.
    -- Cross-tenant isolation is handled at the deployment layer.
    acl_tags        TEXT[]      NOT NULL DEFAULT '{}',

    -- namespace: 'user' applies ACL; 'evaluation' bypasses ACL.
    -- corpus_id: evaluation-only dataset id; NULL for user docs;
    -- never exposed in user-facing query APIs.
    namespace       TEXT        NOT NULL DEFAULT 'user',
    corpus_id       TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by      TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at      TIMESTAMPTZ,

    UNIQUE (source_type, source_uri, revision)
);

CREATE INDEX IF NOT EXISTS documents_source_lookup_idx
    ON documents (source_type, source_uri);

-- Hot path: retrieval only sees published, non-deleted docs.
CREATE INDEX IF NOT EXISTS documents_active_idx
    ON documents (id)
    WHERE status = 'published' AND deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS documents_acl_tags_gin_idx
    ON documents USING GIN (acl_tags);

CREATE INDEX IF NOT EXISTS documents_namespace_idx
    ON documents (namespace);

CREATE INDEX IF NOT EXISTS documents_eval_corpus_idx
    ON documents (corpus_id)
    WHERE namespace = 'evaluation';

CREATE INDEX IF NOT EXISTS documents_deleted_at_idx
    ON documents (deleted_at);

-- =============================================================================
-- Table: chunks
-- =============================================================================

CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    document_id     UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    revision        INT         NOT NULL,

    chunk_index     INT         NOT NULL,
    content         TEXT        NOT NULL,
    token_count     INT         NOT NULL,
    source          TEXT,
    page_number     INT,

    -- Character offsets into the document's full extracted text, used for
    -- citation highlighting. TODO: computed by chunker; currently set to
    -- [0, len(content)] as a placeholder.
    char_start      INT         NOT NULL,
    char_end        INT         NOT NULL,
    CHECK (char_end > char_start),

    -- 384 dims matches paraphrase-multilingual-MiniLM-L12-v2.
    -- Nullable to allow writing chunks before backfilling embeddings
    -- within the same transaction.
    embedding       VECTOR(384),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (document_id, revision, chunk_index)
);

-- HNSW with cosine ops: better recall/latency than IVFFlat; cosine is more
-- stable than L2 for unnormalized SentenceTransformer output.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_doc_revision_idx
    ON chunks (document_id, revision);
