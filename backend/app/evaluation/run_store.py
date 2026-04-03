"""Local persistence for evaluation runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.common.config import settings
from app.common.logging import get_logger

logger = get_logger(__name__)

_RUNS_FILE = settings.cache_dir / "evaluation_runs.json"
_MAX_RUNS = 100


def _load_runs() -> List[Dict[str, Any]]:
    if not _RUNS_FILE.exists():
        return []
    try:
        with open(_RUNS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning(f"加载 evaluation runs 失败: {exc}")
        return []


def _save_runs(runs: List[Dict[str, Any]]) -> None:
    try:
        _RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_RUNS_FILE, "w", encoding="utf-8") as f:
            json.dump(runs[-_MAX_RUNS:], f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error(f"保存 evaluation runs 失败: {exc}")


def add_run(run: Dict[str, Any]) -> Dict[str, Any]:
    runs = _load_runs()
    runs.append(run)
    _save_runs(runs)
    return run


def list_runs(limit: int = 20) -> List[Dict[str, Any]]:
    runs = list(reversed(_load_runs()))
    summaries: List[Dict[str, Any]] = []
    for run in runs[:limit]:
        summaries.append({
            "run_id": run.get("run_id"),
            "timestamp": run.get("timestamp"),
            "dataset_id": run.get("dataset_id"),
            "dataset_name": run.get("dataset_name"),
            "document_id": run.get("document_id"),
            "methods": run.get("methods", []),
            "metrics": run.get("metrics", {}),
            "case_count": run.get("case_count", 0),
            "status": run.get("status", "success"),
        })
    return summaries


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    for run in _load_runs():
        if run.get("run_id") == run_id:
            return run
    return None


def clear_runs() -> None:
    _save_runs([])


def runs_file_path() -> Path:
    return _RUNS_FILE
