from src.v2.graph.workflow import (
    route_after_memory,
    use_memory_evidence_node,
)
from src.v2.models.schemas import EvaluationInput, Evidence


def build_state(
    *,
    memory_hit=False,
    memory_evidence=None,
):
    return {
        "evaluation_input": EvaluationInput(
            question="What is machine learning?",
            response="Test response.",
        ),
        "memory_matches": [],
        "memory_hit": memory_hit,
        "memory_decision": None,
        "memory_decision_reason": None,
        "memory_similarity": None,
        "memory_validation_score": None,
        "memory_validation_confidence": None,
        "memory_decision_score": None,
        "memory_answer": None,
        "memory_evidence": memory_evidence or [],
        "evidence": [],
    }


def build_evidence():
    return [
        Evidence(
            evidence_id="memory-evidence-1",
            document_id="01_rag",
            chunk_id="chunk-1",
            content="Machine learning learns patterns from data.",
            source="01_rag.txt",
            relevance_score=0.95,
        )
    ]


def test_memory_hit_routes_to_memory():
    state = build_state(memory_hit=True)

    result = route_after_memory(state)

    assert result == "use_memory"


def test_memory_miss_routes_to_fresh_retrieval():
    state = build_state(memory_hit=False)

    result = route_after_memory(state)

    assert result == "fresh_retrieval"


def test_memory_evidence_becomes_shared_evidence():
    evidence = build_evidence()

    state = build_state(
        memory_hit=True,
        memory_evidence=evidence,
    )

    result = use_memory_evidence_node(state)

    assert result["evidence"] == evidence
    assert len(result["evidence"]) == 1
    assert result["evidence"][0].evidence_id == "memory-evidence-1"


def test_memory_evidence_node_rejects_empty_memory_evidence():
    state = build_state(
        memory_hit=True,
        memory_evidence=[],
    )

    try:
        use_memory_evidence_node(state)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == (
            "Memory hit occurred but no memory evidence was available."
        )