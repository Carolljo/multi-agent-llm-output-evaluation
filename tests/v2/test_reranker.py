
import pytest

from src.v2.models.schemas import Evidence
from src.v2.rag.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.last_pairs = None

    def predict(self, pairs, show_progress_bar=False):
        self.last_pairs = pairs
        return self.scores


def make_evidence(evidence_id, content):
    return Evidence(
        evidence_id=evidence_id,
        document_id="doc1",
        chunk_id=evidence_id,
        content=content,
        source="test.txt",
    )


@pytest.fixture
def reranker():
    instance = CrossEncoderReranker.__new__(
        CrossEncoderReranker
    )
    instance.model = FakeCrossEncoder([0.1, 0.9, 0.4])
    return instance


def test_reranker_orders_evidence_by_score(reranker):
    evidence = [
        make_evidence("A", "First passage"),
        make_evidence("B", "Second passage"),
        make_evidence("C", "Third passage"),
    ]

    results = reranker.rerank("test query", evidence)

    assert [item.evidence_id for item in results] == [
        "B", "C", "A"
    ]


def test_reranker_respects_top_k(reranker):
    evidence = [
        make_evidence("A", "First passage"),
        make_evidence("B", "Second passage"),
        make_evidence("C", "Third passage"),
    ]

    results = reranker.rerank(
        "test query",
        evidence,
        top_k=2,
    )

    assert len(results) == 2
    assert [item.evidence_id for item in results] == ["B", "C"]


def test_reranker_preserves_evidence_metadata(reranker):
    evidence = make_evidence("A", "Example passage")
    evidence.metadata = {"source_type": "local"}

    result = reranker.rerank(
        "test query",
        [evidence],
    )[0]

    assert result.metadata["source_type"] == "local"
    assert result.metadata["reranker_score"] == 0.1

    # The original object must not be modified.
    assert evidence.metadata == {"source_type": "local"}


def test_reranker_returns_empty_for_empty_input(reranker):
    assert reranker.rerank("query", []) == []
    assert reranker.rerank("   ", [make_evidence("A", "Text")]) == []


def test_reranker_rejects_invalid_top_k(reranker):
    with pytest.raises(ValueError):
        reranker.rerank("query", [], top_k=0)


def test_reranker_rejects_invalid_max_length():
    with pytest.raises(ValueError):
        CrossEncoderReranker.__init__(
            CrossEncoderReranker.__new__(CrossEncoderReranker),
            max_length=0,
        )

