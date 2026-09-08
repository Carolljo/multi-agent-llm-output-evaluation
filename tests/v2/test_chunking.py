import pytest

from src.v2.rag.chunking import TextChunker


def test_empty_text_returns_no_chunks():
    chunker = TextChunker()

    assert chunker.split("") == []
    assert chunker.split("   ") == []


def test_text_is_split_into_chunks():
    chunker = TextChunker(
        chunk_size=20,
        chunk_overlap=5,
    )

    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)


def test_chunks_have_overlap():
    chunker = TextChunker(
        chunk_size=20,
        chunk_overlap=5,
    )

    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chunks = chunker.split(text)

    assert chunks[0][-5:] == chunks[1][:5]


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=20,
            chunk_overlap=20,
        )