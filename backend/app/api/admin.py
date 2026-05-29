"""Admin maintenance endpoints."""
from fastapi import APIRouter

from app.common.config import settings
from app.retrieval.vector_store import vector_store

router = APIRouter(tags=["admin"])


@router.post("/cleanup_archived")
async def cleanup_archived():
    """Reclaim storage for archived/soft-deleted documents past retention.

    Deletes PG chunks and uploads/ files for the same doc ids in one pass so they
    can't drift apart. Document tombstones in PG stay for revision history.
    """
    result = vector_store.purge_archived_chunks(settings.archived_retention_days)
    files_deleted = 0
    for doc_id in result["document_ids"]:
        for f in settings.uploads_dir.glob(f"{doc_id}_*"):
            f.unlink()
            files_deleted += 1
    return {
        "success": True,
        "data": {
            "chunks_deleted": result["chunks_deleted"],
            "documents_purged": len(result["document_ids"]),
            "files_deleted": files_deleted,
            "retention_days": settings.archived_retention_days,
        },
    }
