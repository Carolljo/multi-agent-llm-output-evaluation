
import pytest

from src.v2.models.schemas import Evidence
from src.v2.rag.bm25_retriever import BM25Retriever


@pytest.fixture
def evidence():
    return [
        Evidence(
            evidence_id="E1",
            document_id="doc1",
            chunk_id="chunk1",
            content=(
                "Retrieval augmented generation retrieves "
                "external information before generating answers."
            ),
            source="rag.txt",
        ),
        Evidence(
            evidence_id="E2",
            document_id="doc2",
            chunk_id="chunk1",
            content=(
                "Neural networks learn patterns from "
                "training examples."
            ),
            source="ml.txt",
        ),
    ]


def test_bm25_retrieves_relevant_evidence(evidence):
    retriever = BM25Retriever()
    retriever.add_documents(evidence)

    results = retriever.search(
        "retrieval augmented generation external information",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].evidence_id == "E1"


def test_bm25_preserves_evidence_metadata(evidence):
    retriever = BM25Retriever()
    retriever.add_documents(evidence)

    results = retriever.search("neural networks training")

    result = next(item for item in results if item.evidence_id == "E2")

    assert result.document_id == "doc2"
    assert result.chunk_id == "chunk1"
    assert result.source == "ml.txt"


def test_bm25_returns_empty_for_unknown_terms(evidence):
    retriever = BM25Retriever()
    retriever.add_documents(evidence)

    results = retriever.search("quantum teleportation")

    assert results == []


def test_bm25_returns_empty_for_empty_query(evidence):
    retriever = BM25Retriever()
    retriever.add_documents(evidence)

    assert retriever.search("   ") == []


def test_bm25_returns_empty_when_index_is_empty():
    retriever = BM25Retriever()

    assert retriever.search("retrieval") == []


def test_bm25_rejects_invalid_top_k():
    retriever = BM25Retriever()

    with pytest.raises(ValueError):
        retriever.search("retrieval", top_k=0)


def test_bm25_updates_existing_evidence():
    retriever = BM25Retriever()

    original = Evidence(
        evidence_id="E1",
        document_id="doc1",
        chunk_id="chunk1",
        content="Cats are domestic animals.",
    )

    updated = original.model_copy(
        update={"content": "Satellite communication uses radio waves."}
    )

    retriever.add_documents([original])
    retriever.add_documents([updated])

    results = retriever.search("satellite communication radio waves")

    assert len(results) == 1
    assert results[0].content == updated.content

