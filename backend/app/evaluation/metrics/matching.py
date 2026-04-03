"""Reusable matching metrics for RAG evaluation stages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class MatchResult:
    """Generic exact/substring match result."""

    matched: bool
    expected: List[str]
    matched_items: List[str]
    missing_items: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "expected": self.expected,
            "matched_items": self.matched_items,
            "missing_items": self.missing_items,
        }


def keyword_match(text: str, expected_keywords: Iterable[str]) -> MatchResult:
    """Check whether all expected keywords appear in text."""
    expected = [keyword for keyword in expected_keywords if keyword]
    normalized_text = text.lower()
    matched = [keyword for keyword in expected if keyword.lower() in normalized_text]
    missing = [keyword for keyword in expected if keyword.lower() not in normalized_text]
    return MatchResult(
        matched=not missing,
        expected=expected,
        matched_items=matched,
        missing_items=missing,
    )


def source_match(actual_sources: Iterable[str], expected_source: Optional[str]) -> MatchResult:
    """Check whether a source list contains the expected source."""
    expected = [expected_source] if expected_source else []
    sources = [source for source in actual_sources if source]
    if not expected:
        return MatchResult(matched=True, expected=[], matched_items=[], missing_items=[])

    wanted = expected[0]
    matched = [
        source for source in sources
        if source == wanted or source.endswith(f"_{wanted}")
    ]
    return MatchResult(
        matched=bool(matched),
        expected=expected,
        matched_items=matched,
        missing_items=[] if matched else expected,
    )
