"""Integration tests for B2 Document Registry.

Verifies revision tracking, status filtering, and the revisions endpoint.
Requires a live Postgres+pgvector instance. Skipped unless POSTGRES_DSN is set.
"""
from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def store():
    if not os.environ.get("POSTGRES_DSN"):
        pytest.skip("POSTGRES_DSN not set; skipping document registry integration tests")

    from app.common.config import settings
    from app.retrieval.vector_store import VectorStore

    settings.postgres_dsn = os.environ["POSTGRES_DSN"]
    s = VectorStore()
    s.reset()
    yield s
    s.reset()


def _doc(name: str, file_hash: str, namespace: str = "user", revision: int = 1):
    from app.models.schemas import DocumentInfo
    import time
    return DocumentInfo(
        id=str(uuid.uuid4()),
        name=name,
        chunk_count=0,
        pages=1,
        file_size=100,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        file_hash=file_hash,
        namespace=namespace,
        revision=revision,
    )


def test_first_upload_creates_revision_1(store):
    doc = _doc("report.pdf", "hash_v1")
    store.upsert_document(doc)

    row = store.get_document(doc.id)
    assert row is not None
    assert row["revision"] == 1
    assert row["status"] == "published"
    assert row["superseded_by"] is None


def test_same_content_skipped_via_find(store):
    """find_latest_by_source_uri returns same hash → caller should skip re-index."""
    doc = _doc("same_content.pdf", "stable_hash")
    store.upsert_document(doc)

    existing = store.find_latest_by_source_uri("upload", "upload://same_content.pdf")
    assert existing is not None
    assert existing["file_hash"] == "stable_hash"
    assert existing["id"] == doc.id


def test_new_content_creates_revision_2(store):
    """Uploading same filename with different hash should create revision 2."""
    doc_v1 = _doc("evolving.txt", "hash_v1", revision=1)
    store.upsert_document(doc_v1)

    existing = store.find_latest_by_source_uri("upload", "upload://evolving.txt")
    assert existing is not None

    doc_v2 = _doc("evolving.txt", "hash_v2", revision=existing["revision"] + 1)
    store.upsert_document(doc_v2)
    store.archive_document(existing["id"], doc_v2.id)

    # v1 is archived, superseded_by v2
    row_v1 = store.get_document(existing["id"])
    assert row_v1["status"] == "archived"
    assert row_v1["superseded_by"] == doc_v2.id

    # v2 is published
    row_v2 = store.get_document(doc_v2.id)
    assert row_v2["status"] == "published"
    assert row_v2["revision"] == 2


def test_list_documents_excludes_archived(store):
    """list_documents should only return published documents by default."""
    doc_v1 = _doc("filtered.md", "hash_a", revision=1)
    store.upsert_document(doc_v1)
    doc_v2 = _doc("filtered.md", "hash_b", revision=2)
    store.upsert_document(doc_v2)
    store.archive_document(doc_v1.id, doc_v2.id)

    rows = store.list_documents(namespace="user")
    ids = {r["id"] for r in rows}
    assert doc_v1.id not in ids
    assert doc_v2.id in ids


def test_list_documents_include_archived(store):
    """include_archived=True should return both."""
    doc_v1 = _doc("both.txt", "hash_x", revision=1)
    store.upsert_document(doc_v1)
    doc_v2 = _doc("both.txt", "hash_y", revision=2)
    store.upsert_document(doc_v2)
    store.archive_document(doc_v1.id, doc_v2.id)

    rows = store.list_documents(namespace="user", include_archived=True)
    ids = {r["id"] for r in rows}
    assert doc_v1.id in ids
    assert doc_v2.id in ids


def test_get_document_revisions(store):
    """get_document_revisions returns all revisions in order."""
    doc_v1 = _doc("chain.pdf", "chain_hash_1", revision=1)
    store.upsert_document(doc_v1)
    doc_v2 = _doc("chain.pdf", "chain_hash_2", revision=2)
    store.upsert_document(doc_v2)
    store.archive_document(doc_v1.id, doc_v2.id)

    revisions = store.get_document_revisions(doc_v2.id)
    assert len(revisions) == 2
    assert revisions[0]["revision"] == 1
    assert revisions[1]["revision"] == 2
    # Can also look up from v1's id
    revisions_from_v1 = store.get_document_revisions(doc_v1.id)
    assert len(revisions_from_v1) == 2


def test_get_document_revisions_unknown_id(store):
    assert store.get_document_revisions(str(uuid.uuid4())) == []
