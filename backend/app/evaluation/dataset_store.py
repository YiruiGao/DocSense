"""Local persistence for evaluation datasets and test cases."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.common.config import settings
from app.common.logging import get_logger
from app.evaluation.test_cases import DEFAULT_TEST_CASES, QuestionDifficulty, TestCase

logger = get_logger(__name__)

_DATASETS_FILE = settings.cache_dir / "evaluation_datasets.json"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _case_to_dict(test_case: TestCase, dataset_id: str) -> Dict[str, Any]:
    return {
        "case_id": test_case.id,
        "dataset_id": dataset_id,
        "question": test_case.question,
        "document_id": test_case.document_id,
        "category": test_case.category,
        "difficulty": test_case.difficulty.value,
        "expected_keywords": list(test_case.expected_chunks or []),
        "expected_chunk_ids": [],
        "expected_page_numbers": list(test_case.expected_page_numbers or []),
        "should_answer": True,
        "notes": "",
        "enabled": True,
        "source": "seed",
        "created_at": _now(),
        "updated_at": _now(),
    }


def _default_dataset() -> Dict[str, Any]:
    now = _now()
    dataset_id = DEFAULT_TEST_CASES.id
    return {
        "dataset_id": dataset_id,
        "name": DEFAULT_TEST_CASES.name,
        "description": DEFAULT_TEST_CASES.description,
        "document_id": DEFAULT_TEST_CASES.document_id,
        "enabled": True,
        "created_at": now,
        "updated_at": now,
        "cases": [
            _case_to_dict(test_case, dataset_id)
            for test_case in DEFAULT_TEST_CASES.test_cases
        ],
    }


def _load_datasets() -> List[Dict[str, Any]]:
    if not _DATASETS_FILE.exists():
        return [_default_dataset()]
    try:
        with open(_DATASETS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [_default_dataset()]
    except Exception as exc:
        logger.warning(f"加载 evaluation datasets 失败: {exc}")
        return [_default_dataset()]


def _save_datasets(datasets: List[Dict[str, Any]]) -> None:
    try:
        _DATASETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_DATASETS_FILE, "w", encoding="utf-8") as f:
            json.dump(datasets, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error(f"保存 evaluation datasets 失败: {exc}")


def list_datasets(include_cases: bool = False) -> List[Dict[str, Any]]:
    datasets = _load_datasets()
    if include_cases:
        return datasets

    summaries = []
    for dataset in datasets:
        cases = dataset.get("cases", [])
        summaries.append({
            "dataset_id": dataset.get("dataset_id"),
            "name": dataset.get("name"),
            "description": dataset.get("description"),
            "document_id": dataset.get("document_id"),
            "enabled": dataset.get("enabled", True),
            "case_count": len(cases),
            "enabled_case_count": sum(1 for case in cases if case.get("enabled", True)),
            "created_at": dataset.get("created_at"),
            "updated_at": dataset.get("updated_at"),
        })
    return summaries


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    for dataset in _load_datasets():
        if dataset.get("dataset_id") == dataset_id:
            return dataset
    return None


def create_dataset(payload: Dict[str, Any]) -> Dict[str, Any]:
    datasets = _load_datasets()
    now = _now()
    dataset = {
        "dataset_id": f"dataset_{uuid.uuid4().hex[:12]}",
        "name": payload.get("name") or "未命名评测集",
        "description": payload.get("description") or "",
        "document_id": payload.get("document_id"),
        "enabled": payload.get("enabled", True),
        "created_at": now,
        "updated_at": now,
        "cases": [],
    }
    datasets.append(dataset)
    _save_datasets(datasets)
    return dataset


def update_dataset(dataset_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    datasets = _load_datasets()
    for dataset in datasets:
        if dataset.get("dataset_id") != dataset_id:
            continue
        for key in ["name", "description", "document_id", "enabled"]:
            if key in payload:
                dataset[key] = payload[key]
        dataset["updated_at"] = _now()
        _save_datasets(datasets)
        return dataset
    return None


def delete_dataset(dataset_id: str) -> bool:
    datasets = _load_datasets()
    next_datasets = [dataset for dataset in datasets if dataset.get("dataset_id") != dataset_id]
    if len(next_datasets) == len(datasets):
        return False
    _save_datasets(next_datasets)
    return True


def add_case(dataset_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    datasets = _load_datasets()
    for dataset in datasets:
        if dataset.get("dataset_id") != dataset_id:
            continue
        now = _now()
        case = {
            "case_id": f"case_{uuid.uuid4().hex[:12]}",
            "dataset_id": dataset_id,
            "question": payload.get("question") or "",
            "document_id": payload.get("document_id") or dataset.get("document_id"),
            "category": payload.get("category") or "fact_lookup",
            "difficulty": payload.get("difficulty") or "medium",
            "expected_keywords": payload.get("expected_keywords") or [],
            "expected_chunk_ids": payload.get("expected_chunk_ids") or [],
            "expected_page_numbers": payload.get("expected_page_numbers") or [],
            "should_answer": payload.get("should_answer", True),
            "notes": payload.get("notes") or "",
            "enabled": payload.get("enabled", True),
            "source": payload.get("source") or "manual",
            "created_at": now,
            "updated_at": now,
        }
        dataset.setdefault("cases", []).append(case)
        dataset["updated_at"] = now
        _save_datasets(datasets)
        return case
    return None


def update_case(dataset_id: str, case_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    datasets = _load_datasets()
    for dataset in datasets:
        if dataset.get("dataset_id") != dataset_id:
            continue
        for case in dataset.get("cases", []):
            if case.get("case_id") != case_id:
                continue
            for key in [
                "question",
                "document_id",
                "category",
                "difficulty",
                "expected_keywords",
                "expected_chunk_ids",
                "expected_page_numbers",
                "should_answer",
                "notes",
                "enabled",
                "source",
            ]:
                if key in payload:
                    case[key] = payload[key]
            case["updated_at"] = _now()
            dataset["updated_at"] = case["updated_at"]
            _save_datasets(datasets)
            return case
    return None


def delete_case(dataset_id: str, case_id: str) -> bool:
    datasets = _load_datasets()
    for dataset in datasets:
        if dataset.get("dataset_id") != dataset_id:
            continue
        cases = dataset.get("cases", [])
        next_cases = [case for case in cases if case.get("case_id") != case_id]
        if len(next_cases) == len(cases):
            return False
        dataset["cases"] = next_cases
        dataset["updated_at"] = _now()
        _save_datasets(datasets)
        return True
    return False


def cases_as_test_cases(dataset: Dict[str, Any], document_id: Optional[str] = None) -> List[TestCase]:
    test_cases: List[TestCase] = []
    for case in dataset.get("cases", []):
        if not case.get("enabled", True):
            continue
        expected_chunks = list(case.get("expected_chunk_ids") or [])
        expected_chunks.extend(case.get("expected_keywords") or [])
        difficulty_value = case.get("difficulty") or QuestionDifficulty.EASY.value
        try:
            difficulty = QuestionDifficulty(difficulty_value)
        except ValueError:
            difficulty = QuestionDifficulty.EASY
        test_cases.append(
            TestCase(
                id=case.get("case_id"),
                question=case.get("question", ""),
                expected_chunks=expected_chunks,
                expected_page_numbers=case.get("expected_page_numbers") or [],
                difficulty=difficulty,
                category=case.get("category"),
                document_id=case.get("document_id") or document_id or dataset.get("document_id"),
            )
        )
    return test_cases


def datasets_file_path() -> Path:
    return _DATASETS_FILE
