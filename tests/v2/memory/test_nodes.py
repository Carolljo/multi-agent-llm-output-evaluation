from dataclasses import dataclass

from src.v2.memory.nodes import answer_memory_node
from src.v2.models.schemas import EvaluationInput, ReferenceValidation


@dataclass
class FakeMemory:
    answer: str
    validation_score: float
    validation_confidence: float
    evidence: list
    reference_validation: ReferenceValidation


@dataclass
class FakeMatch:
    memory: FakeMemory
    similarity: float


class FakeMemoryRetriever:
    def __init__(self, matches):
        self.matches = matches

    def search(self, question):
        return self.matches


def build_state(question="What is machine learning?"):
    return {
        "evaluation_input": EvaluationInput(
            question=question,
            response="Test response.",
        )
    }


def build_match(
    similarity=0.94,
    validation_score=0.96,
    validation_confidence=0.95,
):
    reference_validation = ReferenceValidation(
        status="valid",
        score=validation_score,
        confidence=validation_confidence,
        supported_claims=["C1"],
        reasoning="The reference is supported by the evidence.",
    )

    memory = FakeMemory(
        answer="Machine learning learns patterns from data.",
        validation_score=validation_score,
        validation_confidence=validation_confidence,
        evidence=["evidence-1", "evidence-2"],
        reference_validation=reference_validation,
    )

    return FakeMatch(
        memory=memory,
        similarity=similarity,
    )


def test_strong_memory_is_selected():
    retriever = FakeMemoryRetriever([build_match()])

    result = answer_memory_node(
        build_state(),
        memory_retriever=retriever,
    )

    assert result["memory_hit"] is True
    assert result["memory_decision"].decision == "use_memory"
    assert result["memory_answer"] == (
        "Machine learning learns patterns from data."
    )
    assert result["memory_evidence"] == [
        "evidence-1",
        "evidence-2",
    ]


def test_no_memory_forces_fresh_retrieval():
    retriever = FakeMemoryRetriever([])

    result = answer_memory_node(
        build_state(),
        memory_retriever=retriever,
    )

    assert result["memory_hit"] is False
    assert result["memory_decision"] is None
    assert result["memory_answer"] is None
    assert result["memory_evidence"] == []


def test_weak_memory_forces_fresh_retrieval():
    retriever = FakeMemoryRetriever(
        [build_match(similarity=0.70)]
    )

    result = answer_memory_node(
        build_state(),
        memory_retriever=retriever,
    )

    assert result["memory_hit"] is False
    assert result["memory_decision"].decision == "fresh_retrieval"
    assert result["memory_answer"] is None


def test_low_validation_score_forces_fresh_retrieval():
    retriever = FakeMemoryRetriever(
        [build_match(validation_score=0.60)]
    )

    result = answer_memory_node(
        build_state(),
        memory_retriever=retriever,
    )

    assert result["memory_hit"] is False
    assert result["memory_decision"].decision == "fresh_retrieval"


def test_low_validation_confidence_forces_fresh_retrieval():
    retriever = FakeMemoryRetriever(
        [build_match(validation_confidence=0.60)]
    )

    result = answer_memory_node(
        build_state(),
        memory_retriever=retriever,
    )

    assert result["memory_hit"] is False
    assert result["memory_decision"].decision == "fresh_retrieval"


def test_node_preserves_memory_quality_signals():
    retriever = FakeMemoryRetriever(
        [
            build_match(
                similarity=0.94,
                validation_score=0.96,
                validation_confidence=0.95,
            )
        ]
    )

    result = answer_memory_node(
        build_state(),
        memory_retriever=retriever,
    )

    assert result["memory_similarity"] == 0.94
    assert result["memory_validation_score"] == 0.96
    assert result["memory_validation_confidence"] == 0.95
    assert result["memory_decision_score"] == (
        0.50 * 0.94
        + 0.30 * 0.96
        + 0.20 * 0.95
    )


def test_node_uses_best_match():
    weak_match = build_match(
        similarity=0.88,
        validation_score=0.80,
        validation_confidence=0.80,
    )

    strong_match = build_match(
        similarity=0.95,
        validation_score=0.96,
        validation_confidence=0.97,
    )

    retriever = FakeMemoryRetriever(
        [
            strong_match,
            weak_match,
        ]
    )

    result = answer_memory_node(
        build_state(),
        memory_retriever=retriever,
    )

    assert result["memory_hit"] is True
    assert result["memory_similarity"] == 0.95
