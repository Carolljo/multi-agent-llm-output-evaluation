from src.v2.memory.decision import MemoryDecisionPolicy
from src.v2.memory.schemas import AnswerMemoryMatch, VerifiedAnswer


def build_match(
    similarity=0.94,
    validation_score=0.96,
    validation_confidence=0.95,
):
    memory = VerifiedAnswer(
        memory_id="answer_test",
        question="What is machine learning?",
        answer="Machine learning learns patterns from data.",
        claims=[],
        evidence_refs=[],
        source_documents=[],
        evidence=[],
        validation_status="valid",
        validation_score=validation_score,
        validation_confidence=validation_confidence,
        created_at="2026-10-04T00:00:00+00:00",
        updated_at="2026-10-04T00:00:00+00:00",
        version=1,
    )

    return AnswerMemoryMatch(
        memory=memory,
        similarity=similarity,
    )


def test_strong_memory_is_used():
    policy = MemoryDecisionPolicy()

    result = policy.evaluate(build_match())

    assert result.decision == "use_memory"
    assert result.decision_score >= 0.85


def test_low_similarity_forces_fresh_retrieval():
    policy = MemoryDecisionPolicy()

    result = policy.evaluate(
        build_match(similarity=0.70)
    )

    assert result.decision == "fresh_retrieval"


def test_low_validation_forces_fresh_retrieval():
    policy = MemoryDecisionPolicy()

    result = policy.evaluate(
        build_match(validation_score=0.60)
    )

    assert result.decision == "fresh_retrieval"


def test_low_confidence_forces_fresh_retrieval():
    policy = MemoryDecisionPolicy()

    result = policy.evaluate(
        build_match(validation_confidence=0.60)
    )

    assert result.decision == "fresh_retrieval"


def test_combined_score_is_calculated_correctly():
    policy = MemoryDecisionPolicy()

    result = policy.evaluate(
        build_match(
            similarity=0.90,
            validation_score=0.90,
            validation_confidence=0.90,
        )
    )

    assert result.decision_score == 0.90
    assert result.decision == "use_memory"


def test_boundary_thresholds_are_accepted():
    policy = MemoryDecisionPolicy()

    result = policy.evaluate(
        build_match(
            similarity=0.85,
            validation_score=0.80,
            validation_confidence=0.80,
        )
    )

    expected_score = (
        0.50 * 0.85
        + 0.30 * 0.80
        + 0.20 * 0.80
    )

    assert result.decision_score == expected_score
    assert result.decision == "fresh_retrieval"