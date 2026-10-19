
import pytest

from src.v2.models.schemas import Evidence
from src.v2.rag.fusion import ReciprocalRankFusion


def make_evidence(evidence_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        document_id="doc1",
        chunk_id=evidence_id,
        content=f"Evidence content for {evidence_id}",
        source="test.txt",
    )


def test_fusion_combines_rankings():
    fusion = ReciprocalRankFusion(k=60)

    dense = [
        make_evidence("A"),
        make_evidence("B"),
    ]

    bm25 = [
        make_evidence("B"),
        make_evidence("A"),
    ]

    results = fusion.fuse(dense, bm25, top_k=2)

    # Both evidence items have the same combined score.
    assert {item.evidence_id for item in results} == {"A", "B"}


def test_fusion_rewards_consistent_high_rank():
    fusion = ReciprocalRankFusion(k=60)

    dense = [
        make_evidence("A"),
        make_evidence("B"),
        make_evidence("C"),
    ]

    bm25 = [
        make_evidence("A"),
        make_evidence("C"),
        make_evidence("B"),
    ]

    results = fusion.fuse(dense, bm25, top_k=3)

    assert results[0].evidence_id == "A"


def test_fusion_deduplicates_evidence():
    fusion = ReciprocalRankFusion(k=60)

    results = fusion.fuse(
        [
            make_evidence("A"),
            make_evidence("A"),
            make_evidence("B"),
        ],
        top_k=5,
    )

    ids = [item.evidence_id for item in results]

    assert len(ids) == len(set(ids))
    assert set(ids) == {"A", "B"}


def test_fusion_preserves_evidence_metadata():
    fusion = ReciprocalRankFusion()

    evidence = make_evidence("A")
    evidence.metadata = {"origin": "local"}

    results = fusion.fuse([evidence])

    assert results[0].metadata["origin"] == "local"
    assert "rrf_score" in results[0].metadata
    assert evidence.metadata == {"origin": "local"}


def test_fusion_handles_empty_results():
    fusion = ReciprocalRankFusion()

    assert fusion.fuse([], []) == []


def test_fusion_rejects_invalid_top_k():
    fusion = ReciprocalRankFusion()

    with pytest.raises(ValueError):
        fusion.fuse([], top_k=0)


def test_fusion_rejects_invalid_k():
    with pytest.raises(ValueError):
        ReciprocalRankFusion(k=0)

