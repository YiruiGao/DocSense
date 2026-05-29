"""Postgres + pgvector vector store.

Uses psycopg3 sync connection pool to match the sync call sites in chat,
documents, hybrid_search, and evaluation. The pool is lazily initialized on
first use and applies schema.sql automatically.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from app.common.config import settings
from app.models.schemas import Chunk, DocumentInfo
from app.retrieval.embeddings import embedding_model

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"
_INITIAL_REVISION = 1

_DOCUMENT_COLUMNS = """\
    id::text, name, chunk_count, pages, file_size,
    to_char(created_at, 'YYYY-MM-DD HH24:MI:SS') AS created_at,
    file_hash, namespace, corpus_id,
    revision, status, superseded_by::text"""


def _row_to_document_info(row: dict) -> DocumentInfo:
    return DocumentInfo(
        id=row["id"],
        name=row["name"],
        chunk_count=row["chunk_count"],
        pages=row["pages"] or 0,
        file_size=row["file_size"],
        created_at=row["created_at"],
        file_hash=row.get("file_hash") or None,
        namespace=row["namespace"],
        corpus_id=row.get("corpus_id") or None,
        revision=row.get("revision", 1),
        status=row.get("status", "published"),
        superseded_by=row.get("superseded_by") or None,
    )


def _configure_connection(conn: Connection) -> None:
    register_vector(conn)


def _row_to_result(row: dict) -> Dict[str, Any]:
    return {
        "chunk_id": row["chunk_id"],
        "content": row["content"],
        "metadata": {
            "document_id": row["document_id"],
            "page_number": row["page_number"],
            "chunk_index": row["chunk_index"],
            "token_count": row["token_count"],
            "source": row["source"],
            "namespace": row["namespace"],
            "corpus_id": row["corpus_id"] or "",
        },
    }


class VectorStore:
    """Vector store backed by Postgres + pgvector."""

    def __init__(self) -> None:
        self._pool: Optional[ConnectionPool] = None

    @property
    def pool(self) -> ConnectionPool:
        if self._pool is None:
            if not settings.postgres_dsn:
                raise RuntimeError("POSTGRES_DSN is not configured")
            self._pool = ConnectionPool(
                conninfo=settings.postgres_dsn,
                min_size=settings.postgres_pool_min,
                max_size=settings.postgres_pool_max,
                configure=_configure_connection,
                open=False,
            )
            self._apply_schema()   # CREATE EXTENSION vector must run before pool opens
            self._pool.open(wait=True)
        return self._pool

    def _apply_schema(self) -> None:
        try:
            ddl = SCHEMA_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("schema.sql not found at %s; skipping bootstrap", SCHEMA_PATH)
            return
        import psycopg
        with psycopg.connect(settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    def upsert_document(self, doc: DocumentInfo, *, source_type: str = "upload") -> None:
        """Insert a new document row. source_uri uses name for stable revision chaining."""
        source_uri = f"{source_type}://{doc.name}"
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (
                        id, name, source_type, source_uri, file_hash,
                        file_size, pages, chunk_count, namespace, corpus_id,
                        revision, status, published_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE SET
                        name        = EXCLUDED.name,
                        chunk_count = EXCLUDED.chunk_count,
                        pages       = EXCLUDED.pages,
                        file_size   = EXCLUDED.file_size,
                        namespace   = EXCLUDED.namespace,
                        corpus_id   = EXCLUDED.corpus_id,
                        updated_at  = now()
                    """,
                    (
                        doc.id, doc.name, source_type, source_uri,
                        doc.file_hash or "", doc.file_size, doc.pages,
                        doc.chunk_count, doc.namespace, doc.corpus_id,
                        doc.revision, doc.status,
                    ),
                )
            conn.commit()

    def find_latest_by_source_uri(self, source_type: str, source_uri: str) -> Optional[Dict[str, Any]]:
        """Return the highest-revision document for a source_uri (any status, including soft-deleted), or None.

        Soft-deleted and archived rows still occupy slots in the
        `(source_type, source_uri, revision)` unique key, so callers must use this row's
        revision to compute the next one. Callers check `status` / `deleted_at` (via the
        extra column below) to decide hash-skip and predecessor archival.
        """
        with self.pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT {_DOCUMENT_COLUMNS},
                           deleted_at IS NOT NULL AS is_deleted
                    FROM documents
                    WHERE source_type = %s AND source_uri = %s
                    ORDER BY revision DESC
                    LIMIT 1
                    """,
                    (source_type, source_uri),
                )
                return cur.fetchone()

    def archive_document(self, doc_id: str, superseded_by_id: str) -> None:
        """Mark a document as archived and set its superseded_by pointer."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents
                       SET status = 'archived',
                           superseded_by = %s,
                           updated_at = now()
                     WHERE id = %s
                    """,
                    (superseded_by_id, doc_id),
                )
            conn.commit()

    def publish_document(self, doc_id: str, old_doc_id: Optional[str] = None) -> None:
        """Atomically flip a draft document to published and optionally archive the predecessor."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents
                       SET status = 'published', published_at = now(), updated_at = now()
                     WHERE id = %s AND status = 'draft'
                    """,
                    (doc_id,),
                )
                if old_doc_id:
                    cur.execute(
                        """
                        UPDATE documents
                           SET status = 'archived', superseded_by = %s, updated_at = now()
                         WHERE id = %s
                        """,
                        (doc_id, old_doc_id),
                    )
            conn.commit()

    def soft_delete_revision_chain(self, doc_id: str) -> List[str]:
        """Mark the document and all its revision siblings (same source_uri) as deleted.

        Returns the list of affected document ids. Chunks stay in place as tombstones
        and are physically removed later by `purge_archived_chunks` once retention expires.
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH anchor AS (
                        SELECT source_type, source_uri FROM documents WHERE id = %s
                    )
                    UPDATE documents d
                       SET deleted_at = now(), updated_at = now()
                      FROM anchor a
                     WHERE d.source_type = a.source_type
                       AND d.source_uri = a.source_uri
                       AND d.deleted_at IS NULL
                    RETURNING d.id::text
                    """,
                    (doc_id,),
                )
                affected = [r[0] for r in cur.fetchall()]
            conn.commit()
        return affected

    def delete_draft(self, doc_id: str) -> None:
        """Remove a draft document and its chunks (used to clean up after a failed ingest)."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE id = %s AND status = 'draft'", (doc_id,))
            conn.commit()

    def purge_archived_chunks(self, retention_days: int) -> Dict[str, Any]:
        """Physically delete chunks for archived or soft-deleted documents past retention.

        Document rows are kept as tombstones so revision history remains queryable.
        Returns `{"chunks_deleted": int, "document_ids": List[str]}` so callers can
        clean external artifacts (uploads, BM25) for the same docs.
        """
        cutoff = timedelta(days=retention_days)
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id::text FROM documents
                    WHERE (status = 'archived' OR deleted_at IS NOT NULL)
                      AND updated_at < now() - %s
                    """,
                    (cutoff,),
                )
                doc_ids = [r[0] for r in cur.fetchall()]
                if not doc_ids:
                    return {"chunks_deleted": 0, "document_ids": []}
                cur.execute("DELETE FROM chunks WHERE document_id = ANY(%s)", (doc_ids,))
                chunks_deleted = cur.rowcount or 0
            conn.commit()
        return {"chunks_deleted": chunks_deleted, "document_ids": doc_ids}

    def list_documents(
        self,
        namespace: Optional[str] = "user",
        include_archived: bool = False,
    ) -> List[Dict[str, Any]]:
        """List documents from Postgres, defaulting to published + not deleted only.

        Pass namespace=None to query all namespaces.
        """
        conditions = ["deleted_at IS NULL"]
        params: List[Any] = []
        if namespace is not None:
            conditions.append("namespace = %s")
            params.append(namespace)
        if not include_archived:
            conditions.append("status = 'published'")
        where = " AND ".join(conditions)
        with self.pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE {where} ORDER BY created_at DESC",
                    params,
                )
                return cur.fetchall()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single document row by ID."""
        with self.pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE id = %s",
                    (doc_id,),
                )
                return cur.fetchone()

    def get_document_revisions(self, doc_id: str) -> List[Dict[str, Any]]:
        """Return all revisions sharing the same (source_type, source_uri) chain as doc_id."""
        with self.pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    WITH anchor AS (
                        SELECT source_type, source_uri FROM documents WHERE id = %s
                    )
                    SELECT {_DOCUMENT_COLUMNS}
                    FROM documents
                    JOIN anchor USING (source_type, source_uri)
                    ORDER BY revision
                    """,
                    (doc_id,),
                )
                return cur.fetchall()

    def add_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        embeddings = embedding_model.encode(texts)
        if isinstance(embeddings, np.ndarray):
            embeddings_list = list(embeddings)
        else:
            embeddings_list = [np.asarray(e, dtype=np.float32) for e in embeddings]

        rows = []
        for chunk, emb in zip(chunks, embeddings_list):
            # TODO: replace with real chunker-computed offsets when chunker
            # starts tracking char positions.
            rows.append((
                chunk.id, chunk.document_id, _INITIAL_REVISION, chunk.chunk_index,
                chunk.content, chunk.token_count, chunk.source, chunk.page_number,
                0, max(1, len(chunk.content)), emb,
            ))

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks (
                        id, document_id, revision, chunk_index, content,
                        token_count, source, page_number,
                        char_start, char_end, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content     = EXCLUDED.content,
                        token_count = EXCLUDED.token_count,
                        embedding   = EXCLUDED.embedding
                    """,
                    rows,
                )
            conn.commit()
        return len(rows)

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        corpus_id: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        query_embedding = embedding_model.encode(query)
        if isinstance(query_embedding, np.ndarray) and query_embedding.ndim == 2:
            query_embedding = query_embedding[0]
        return self.search_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
            namespace=namespace,
            corpus_id=corpus_id,
            min_score=min_score,
        )

    def search_by_embedding(
        self,
        query_embedding,
        top_k: int = 10,
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        corpus_id: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        vec = np.asarray(query_embedding, dtype=np.float32)

        conditions: List[str] = ["c.embedding IS NOT NULL", "d.status = 'published'", "d.deleted_at IS NULL"]
        params: List[Any] = []

        if document_id:
            conditions.append("c.document_id = %s")
            params.append(document_id)
        if namespace:
            conditions.append("d.namespace = %s")
            params.append(namespace)
        if corpus_id:
            conditions.append("d.corpus_id = %s")
            params.append(corpus_id)

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT
                c.id::text          AS chunk_id,
                c.document_id::text AS document_id,
                c.content,
                c.page_number,
                c.chunk_index,
                c.token_count,
                c.source,
                d.namespace,
                d.corpus_id,
                1 - (c.embedding <=> %s) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {where_clause}
            ORDER BY c.embedding <=> %s
            LIMIT %s
        """

        with self.pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, [vec, *params, vec, top_k])
                rows = cur.fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            score = float(row["score"])
            if score < min_score:
                continue
            result = _row_to_result(row)
            result["score"] = score
            results.append(result)
        return results

    def delete_document(self, document_id: str) -> int:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
                deleted = cur.rowcount or 0
                cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            conn.commit()
        return deleted

    def list_document_ids(self) -> List[str]:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT document_id::text FROM chunks")
                rows = cur.fetchall()
        return sorted(r[0] for r in rows)

    def delete_documents_not_in(self, valid_document_ids: set[str]) -> int:
        all_ids = self.list_document_ids()
        stale = [doc_id for doc_id in all_ids if doc_id not in valid_document_ids]
        if not stale:
            return 0
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks   WHERE document_id = ANY(%s)", (stale,))
                deleted = cur.rowcount or 0
                cur.execute("DELETE FROM documents WHERE id          = ANY(%s)", (stale,))
            conn.commit()
        return deleted

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        with self.pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        c.id::text          AS chunk_id,
                        c.document_id::text AS document_id,
                        c.content,
                        c.page_number,
                        c.chunk_index,
                        c.token_count,
                        c.source,
                        d.namespace,
                        d.corpus_id
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.document_id = %s
                    ORDER BY c.chunk_index
                    """,
                    (document_id,),
                )
                rows = cur.fetchall()
        return [_row_to_result(row) for row in rows]

    def update_document_metadata(
        self,
        document_id: str,
        namespace: str,
        corpus_id: Optional[str] = None,
    ) -> int:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents
                       SET namespace  = %s,
                           corpus_id  = %s,
                           updated_at = now()
                     WHERE id = %s
                       AND (namespace IS DISTINCT FROM %s
                            OR corpus_id IS DISTINCT FROM %s)
                    """,
                    (namespace, corpus_id, document_id, namespace, corpus_id),
                )
                changed = cur.rowcount or 0
            conn.commit()
        return changed

    def count(self, document_id: Optional[str] = None) -> int:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                if document_id:
                    cur.execute(
                        "SELECT COUNT(*) FROM chunks WHERE document_id = %s",
                        (document_id,),
                    )
                else:
                    cur.execute("SELECT COUNT(*) FROM chunks")
                row = cur.fetchone()
        return int(row[0]) if row else 0

    def reset(self) -> None:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")
            conn.commit()


vector_store = VectorStore()
