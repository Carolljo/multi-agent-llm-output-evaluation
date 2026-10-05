from src.v2.memory.write_policy import MemoryWritePolicy


def evaluate_valid_answer(**overrides):
    values = {
        "verdict": "ACCEPT",
        "validation_score": 0.95,
        "validation_confidence": 0.94,
        "aggregated_confidence": 0.92,
        "evidence_count": 5,
        "unresolved_disagreement": False,
    }

    values.update(overrides)

    return MemoryWritePolicy().evaluate(**values)


def test_strong_accepted_answer_is_stored():
    result = evaluate_valid_answer()

    assert result.decision == "store"
    assert "passed" in result.reason


def test_reject_verdict_is_not_stored():
    result = evaluate_valid_answer(verdict="REJECT")

    assert result.decision == "do_not_store"


def test_partial_verdict_is_not_stored():
    result = evaluate_valid_answer(verdict="PARTIAL")

    assert result.decision == "do_not_store"


def test_low_validation_score_is_not_stored():
    result = evaluate_valid_answer(validation_score=0.70)

    assert result.decision == "do_not_store"


def test_low_validation_confidence_is_not_stored():
    result = evaluate_valid_answer(validation_confidence=0.70)

    assert result.decision == "do_not_store"


def test_low_aggregated_confidence_is_not_stored():
    result = evaluate_valid_answer(aggregated_confidence=0.70)

    assert result.decision == "do_not_store"


def test_missing_evidence_is_not_stored():
    result = evaluate_valid_answer(evidence_count=0)

    assert result.decision == "do_not_store"


def test_unresolved_disagreement_is_not_stored():
    result = evaluate_valid_answer(unresolved_disagreement=True)

    assert result.decision == "do_not_store"


def test_boundary_values_are_accepted():
    result = evaluate_valid_answer(
        validation_score=0.80,
        validation_confidence=0.80,
        aggregated_confidence=0.85,
    )

    assert result.decision == "store"


def test_decision_contains_all_quality_signals():
    result = evaluate_valid_answer()

    assert result.validation_score == 0.95
    assert result.validation_confidence == 0.94
    assert result.aggregated_confidence == 0.92
    assert result.evidence_count == 5
    assert result.unresolved_disagreement is False