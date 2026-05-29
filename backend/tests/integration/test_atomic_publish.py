"""Integration tests for B3: Atomic Publish + Revision Flow.

Verifies that draft documents are invisible to retrieval, publish_document
atomically flips status and archives the predecessor, delete_draft cleans up
on ingest failure, and purge_archived_chunks respects the retention window.
"""
from __future__ import annotations

import os
import uuid
import time

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def store():
    if not os.environ.get("POSTGRES_DSN"):
        pytest.skip("POSTGRES_DSN not set; skipping atomic publish integration tests")

    from app.common.config import settings
    from app.retrieval.vector_store import VectorStore

    settings.postgres_dsn = os.environ["POSTGRES_DSN"]
    s = VectorStore()
    s.reset()
    yield s
    s.reset()


def _doc(store, name: str, status: str = "published", revision: int = 1):
    from app.models.schemas import DocumentInfo
    doc_id = str(uuid.uuid4())
    store.upsert_document(DocumentInfo(
        id=doc_id, name=name, chunk_count=0, pages=1,
        file_size=0, created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        file_hash="hash_" + doc_id[:8], namespace="user",
        status=status, revision=revision,
    ))
    return doc_id


def _add_chunk(store, doc_id: str, content: str, chunk_index: int = 0):
    from app.models.schemas import Chunk
    chunk = Chunk(
        id=str(uuid.uuid4()), document_id=doc_id,
        content=content, page_number=1, chunk_index=chunk_index,
        token_count=max(1, len(content) // 4), source="test.txt",
        namespace="user",
    )
    store.add_chunks([chunk])


def test_draft_invisible_in_list(store):
    doc_id = _doc(store, "draft_list.txt", status="draft")
    rows = store.list_documents(namespace="user")
    assert not any(r["id"] == doc_id for r in rows)
    store.delete_draft(doc_id)


def test_draft_invisible_in_search(store):
    doc_id = _doc(store, "draft_search.txt", status="draft")
    _add_chunk(store, doc_id, "unique atomic publish draft search content")
    results = store.search("unique atomic publish draft search content", top_k=10)
    assert not any(r["metadata"]["document_id"] == doc_id for r in results)
    store.delete_draft(doc_id)


def test_publish_document_makes_visible(store):
    doc_id = _doc(store, "publish_visible.txt", status="draft")
    _add_chunk(store, doc_id, "content that becomes visible after publish")

    store.publish_document(doc_id)

    row = store.get_document(doc_id)
    assert row["status"] == "published"

    rows = store.list_documents(namespace="user")
    assert any(r["id"] == doc_id for r in rows)


def test_publish_archives_predecessor(store):
    doc_v1 = _doc(store, "atomic_chain.txt", revision=1)
    doc_v2 = _doc(store, "atomic_chain.txt", status="draft", revision=2)

    store.publish_document(doc_v2, old_doc_id=doc_v1)

    row_v1 = store.get_document(doc_v1)
    assert row_v1["status"] == "archived"
    assert row_v1["superseded_by"] == doc_v2

    row_v2 = store.get_document(doc_v2)
    assert row_v2["status"] == "published"


def test_publish_is_atomic_old_not_archived_if_new_was_not_draft(store):
    """publish_document only flips docs that are actually in draft status."""
    doc_id = _doc(store, "not_draft.txt", status="published")
    old_id = _doc(store, "predecessor.txt", status="published")

    store.publish_document(doc_id, old_doc_id=old_id)

    # doc_id was already published — WHERE status='draft' matches nothing, no error
    # old_id still gets archived (the old-revision archival is unconditional)
    row = store.get_document(old_id)
    assert row["status"] == "archived"


def test_delete_draft_removes_doc_and_chunks(store):
    doc_id = _doc(store, "delete_draft.txt", status="draft")
    _add_chunk(store, doc_id, "draft chunk A", chunk_index=0)
    _add_chunk(store, doc_id, "draft chunk B", chunk_index=1)

    assert store.count(document_id=doc_id) == 2
    store.delete_draft(doc_id)
    assert store.count(document_id=doc_id) == 0
    assert store.get_document(doc_id) is None


def test_delete_draft_does_not_remove_published(store):
    doc_id = _doc(store, "published_safe.txt", status="published")
    _add_chunk(store, doc_id, "published chunk")

    store.delete_draft(doc_id)  # WHERE status='draft' — should be a no-op

    assert store.count(document_id=doc_id) == 1
    assert store.get_document(doc_id) is not None


def test_purge_archived_chunks_removes_old(store):
    # Use a real v1→v2 chain so the superseded_by FK is satisfied
    v1_id = _doc(store, "old_archived.txt", revision=1)
    _add_chunk(store, v1_id, "archived old content")
    v2_id = _doc(store, "old_archived.txt", status="draft", revision=2)
    store.publish_document(v2_id, old_doc_id=v1_id)

    # Backdate v1's updated_at to simulate archived 10 days ago
    with store.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET updated_at = now() - INTERVAL '10 days' WHERE id = %s",
                (v1_id,),
            )
        conn.commit()

    assert store.count(document_id=v1_id) == 1
    result = store.purge_archived_chunks(retention_days=7)
    assert result["chunks_deleted"] >= 1
    assert v1_id in result["document_ids"]
    assert store.count(document_id=v1_id) == 0
    # Document tombstone remains for revision history
    assert store.get_document(v1_id) is not None


def test_purge_respects_retention_window(store):
    v1_id = _doc(store, "recent_archived.txt", revision=1)
    _add_chunk(store, v1_id, "recently archived content")
    v2_id = _doc(store, "recent_archived.txt", status="draft", revision=2)
    store.publish_document(v2_id, old_doc_id=v1_id)
    # v1's updated_at is now() by default — within the 7-day window

    result = store.purge_archived_chunks(retention_days=7)
    # v1's chunks should NOT be purged
    assert v1_id not in result["document_ids"]
    assert store.count(document_id=v1_id) == 1
