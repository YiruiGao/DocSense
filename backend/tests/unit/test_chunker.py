import pytest

from app.ingestion.chunker import SemanticChunker


@pytest.mark.unit
def test_chunk_text_ignores_blank_input():
    chunker = SemanticChunker(min_tokens=1, max_tokens=20, overlap=0)

    assert chunker.chunk_text(" \n\t ", page_number=1) == []


@pytest.mark.unit
def test_chunk_pages_keeps_global_chunk_indexes_across_pages():
    chunker = SemanticChunker(min_tokens=1, max_tokens=80, overlap=0)

    chunks = chunker.chunk_pages([
        (1, "First page policy paragraph."),
        (2, "Second page onboarding paragraph."),
    ])

    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert chunks[0].content == "First page policy paragraph."
    assert chunks[1].content == "Second page onboarding paragraph."
