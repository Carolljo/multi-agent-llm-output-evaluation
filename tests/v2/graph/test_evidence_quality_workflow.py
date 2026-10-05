from src.v2.evidence.quality import EvidenceQualityScorer
from src.v2.graph.workflow import score_evidence_quality_node
from src.v2.models.schemas import EvaluationInput, Evidence


def build_state():
    evidence = [
        Evidence(
            evidence_id="evidence-1",
            document_id="01_rag",
            chunk_id="chunk-1",
            content="Machine learning learns patterns from data.",
            source="01_rag.txt",
            relevance_score=0.95,
        ),
        Evidence(
            evidence_id="evidence-2",
            document_id="02_large_language_models",
            chunk_id="chunk-1",
            content="Large language models learn statistical patterns.",
            source="02_large_language_models.txt",
            relevance_score=0.80,
        ),
    ]

    return {
        "evaluation_input": EvaluationInput(
            question="What is machine learning?",
            response="Test response.",
        ),
        "evidence": evidence,
    }


def test_evidence_quality_is_added_to_shared_state():
    state = build_state()

    result = score_evidence_quality_node(
        state,
        EvidenceQualityScorer(),
    )

    assert "evidence_quality" in result
    assert len(result["evidence_quality"]) == 2


def test_each_evidence_item_gets_quality_score():
    state = build_state()

    result = score_evidence_quality_node(
        state,
        EvidenceQualityScorer(),
    )

    first = result["evidence_quality"][0]

    assert first.evidence_id == "evidence-1"
    assert 0.0 <= first.overall_score <= 1.0
    assert first.quality_level in {
        "high",
        "medium",
        "low",
    }


def test_quality_scores_preserve_evidence_identity():
    state = build_state()

    result = score_evidence_quality_node(
        state,
        EvidenceQualityScorer(),
    )

    evidence_ids = [
        item.evidence_id
        for item in result["evidence_quality"]
    ]

    assert evidence_ids == [
        "evidence-1",
        "evidence-2",
    ]


def test_empty_evidence_pool_is_rejected():
    state = build_state()
    state["evidence"] = []

    try:
        score_evidence_quality_node(
            state,
            EvidenceQualityScorer(),
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == (
            "Cannot score evidence quality because the evidence pool is empty."
        )