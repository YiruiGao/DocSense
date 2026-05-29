"""Integration tests for VectorStore (Postgres + pgvector backend).

Requires a live Postgres+pgvector instance. Skipped unless POSTGRES_DSN is set
(e.g. via the docker-compose `postgres` service).
"""
from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def store():
    if not os.environ.get("POSTGRES_DSN"):
        pytest.skip("POSTGRES_DSN not set; skipping pgvector integration tests")

    from app.common.config import settings
    from app.retrieval.vector_store import VectorStore

    settings.postgres_dsn = os.environ["POSTGRES_DSN"]
    s = VectorStore()
    s.reset()
    yield s
    s.reset()


def _chunk(doc_id: str, idx: int, content: str, namespace: str = "user",
           corpus_id: str | None = None):
    from app.models.schemas import Chunk
    return Chunk(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        content=content,
        page_number=1,
        chunk_index=idx,
        token_count=max(1, len(content) // 4),
        source="test.pdf",
        namespace=namespace,
        corpus_id=corpus_id,
    )


def _add(store, chunks: list):
    """upsert document rows then add chunks."""
    from app.models.schemas import DocumentInfo
    import time
    seen = {c.document_id for c in chunks}
    for doc_id in seen:
        store.upsert_document(DocumentInfo(
            id=doc_id, name=doc_id, chunk_count=0, pages=1,
            file_size=0, created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            file_hash="", namespace=chunks[0].namespace,
            corpus_id=chunks[0].corpus_id,
        ))
    return store.add_chunks(chunks)


def test_add_and_count(store):
    doc_id = str(uuid.uuid4())
    assert _add(store,[_chunk(doc_id, i, f"hello world chunk {i}") for i in range(3)]) == 3
    assert store.count(document_id=doc_id) == 3
    assert store.count() >= 3


def test_search_returns_relevant(store):
    doc_id = str(uuid.uuid4())
    _add(store,[
        _chunk(doc_id, 0, "Postgres pgvector enables similarity search."),
        _chunk(doc_id, 1, "The quick brown fox jumps over the lazy dog."),
        _chunk(doc_id, 2, "Vector databases support nearest neighbor queries."),
    ])
    results = store.search("vector database similarity", top_k=2, document_id=doc_id)
    assert len(results) == 2
    assert all(r["metadata"]["document_id"] == doc_id for r in results)
    assert all(0.0 <= r["score"] <= 1.0 for r in results)


def test_namespace_filter(store):
    user_doc = str(uuid.uuid4())
    eval_doc = str(uuid.uuid4())
    _add(store,[_chunk(user_doc, 0, "user namespace content")])
    _add(store,[_chunk(eval_doc, 0, "evaluation namespace content",
                             namespace="evaluation", corpus_id="suite-a")])

    user_ids = {r["metadata"]["document_id"] for r in store.search("content", top_k=20, namespace="user")}
    eval_ids = {r["metadata"]["document_id"] for r in store.search("content", top_k=20, namespace="evaluation")}

    assert user_doc in user_ids and eval_doc not in user_ids
    assert eval_doc in eval_ids and user_doc not in eval_ids


def test_delete_document(store):
    doc_id = str(uuid.uuid4())
    _add(store,[_chunk(doc_id, i, f"to be deleted {i}") for i in range(2)])
    assert store.count(document_id=doc_id) == 2
    assert store.delete_document(doc_id) == 2
    assert store.count(document_id=doc_id) == 0


def test_prune_orphans(store):
    keep = str(uuid.uuid4())
    drop = str(uuid.uuid4())
    _add(store,[_chunk(keep, 0, "keep me")])
    _add(store,[_chunk(drop, 0, "drop me")])

    assert drop in set(store.list_document_ids())
    assert store.delete_documents_not_in({keep}) >= 1
    assert drop not in set(store.list_document_ids())
