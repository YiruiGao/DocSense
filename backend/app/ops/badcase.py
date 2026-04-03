"""Local persistence for RAG badcases."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.common.config import settings
from app.common.logging import get_logger

logger = get_logger(__name__)

_BADCASES_FILE = settings.cache_dir / "badcases.json"
_MAX_BADCASES = 500


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_badcases() -> List[Dict[str, Any]]:
    if not _BADCASES_FILE.exists():
        return []
    try:
        with open(_BADCASES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning(f"加载 badcases 失败: {exc}")
        return []


def _save_badcases(badcases: List[Dict[str, Any]]) -> None:
    try:
        _BADCASES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_BADCASES_FILE, "w", encoding="utf-8") as f:
            json.dump(badcases[-_MAX_BADCASES:], f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error(f"保存 badcases 失败: {exc}")


def list_badcases(
    status: Optional[str] = None,
    failure_type: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    items = list(reversed(_load_badcases()))
    if status:
        items = [item for item in items if item.get("status") == status]
    if failure_type:
        items = [item for item in items if item.get("failure_type") == failure_type]
    return items[:limit]


def get_badcase(badcase_id: str) -> Optional[Dict[str, Any]]:
    for badcase in _load_badcases():
        if badcase.get("badcase_id") == badcase_id:
            return badcase
    return None


def create_badcase(payload: Dict[str, Any]) -> Dict[str, Any]:
    badcases = _load_badcases()
    now = _now()
    badcase = {
        "badcase_id": f"badcase_{uuid.uuid4().hex[:12]}",
        "trace_id": payload.get("trace_id"),
        "question": payload.get("question") or "",
        "answer": payload.get("answer") or "",
        "document_id": payload.get("document_id"),
        "document_name": payload.get("document_name"),
        "failure_type": payload.get("failure_type") or "other",
        "severity": payload.get("severity") or "medium",
        "expected_behavior": payload.get("expected_behavior") or "",
        "note": payload.get("note") or "",
        "status": payload.get("status") or "open",
        "trace_snapshot": payload.get("trace_snapshot") or {},
        "created_at": now,
        "updated_at": now,
    }
    badcases.append(badcase)
    _save_badcases(badcases)
    return badcase


def update_badcase(badcase_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    badcases = _load_badcases()
    for badcase in badcases:
        if badcase.get("badcase_id") != badcase_id:
            continue
        for key in [
            "failure_type",
            "severity",
            "expected_behavior",
            "note",
            "status",
            "question",
            "answer",
            "document_id",
            "document_name",
        ]:
            if key in payload:
                badcase[key] = payload[key]
        badcase["updated_at"] = _now()
        _save_badcases(badcases)
        return badcase
    return None


def delete_badcase(badcase_id: str) -> bool:
    badcases = _load_badcases()
    next_badcases = [item for item in badcases if item.get("badcase_id") != badcase_id]
    if len(next_badcases) == len(badcases):
        return False
    _save_badcases(next_badcases)
    return True


def clear_badcases() -> None:
    _save_badcases([])


def badcases_file_path() -> Path:
    return _BADCASES_FILE
