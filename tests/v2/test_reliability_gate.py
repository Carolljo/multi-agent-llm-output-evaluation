from src.v2.aggregation.reliability_gate import check_reliability
from src.v2.models.schemas import (
    EvaluatorResult,
    ReferenceValidation,
)


def make_reference_validation(
    *,
    status="valid",
    score=0.95,
    confidence=0.95,
):
    return ReferenceValidation(
        status=status,
        score=score,
        confidence=confidence,
        supported_claims=["claim-1"],
        unsupported_claims=[],
        missing_information=[],
        reasoning="Test validation.",
    )


def make_grounding_result(score=0.95):
    return EvaluatorResult(
        evaluator="evidence_grounding",
        score=score,
        confidence=0.95,
        claims_evaluated=1,
        errors=[],
        reasoning="Test grounding evaluation.",
        evidence_refs=["evidence-1"],
    )


def test_reliable_evaluation_passes_gate():
    result = check_reliability(
        reference_validation=make_reference_validation(),
        evaluations=[make_grounding_result()],
        evidence_count=1,
        aggregated_confidence=0.95,
        unresolved_disagreement=False,
    )

    assert result.allowed is True
    assert result.reasons == ()


def test_missing_reference_validation_fails_gate():
    result = check_reliability(
        reference_validation=None,
        evaluations=[make_grounding_result()],
        evidence_count=1,
        aggregated_confidence=0.95,
        unresolved_disagreement=False,
    )

    assert result.allowed is False
    assert any("missing" in reason.lower() for reason in result.reasons)


def test_invalid_reference_fails_gate():
    result = check_reliability(
        reference_validation=make_reference_validation(
            status="needs_review"
        ),
        evaluations=[make_grounding_result()],
        evidence_count=1,
        aggregated_confidence=0.95,
        unresolved_disagreement=False,
    )

    assert result.allowed is False


def test_insufficient_grounding_fails_gate():
    result = check_reliability(
        reference_validation=make_reference_validation(),
        evaluations=[make_grounding_result(score=0.40)],
        evidence_count=1,
        aggregated_confidence=0.95,
        unresolved_disagreement=False,
    )

    assert result.allowed is False


def test_missing_evidence_fails_gate():
    result = check_reliability(
        reference_validation=make_reference_validation(),
        evaluations=[make_grounding_result()],
        evidence_count=0,
        aggregated_confidence=0.95,
        unresolved_disagreement=False,
    )

    assert result.allowed is False


def test_low_aggregated_confidence_fails_gate():
    result = check_reliability(
        reference_validation=make_reference_validation(),
        evaluations=[make_grounding_result()],
        evidence_count=1,
        aggregated_confidence=0.60,
        unresolved_disagreement=False,
    )

    assert result.allowed is False


def test_unresolved_disagreement_fails_gate():
    result = check_reliability(
        reference_validation=make_reference_validation(),
        evaluations=[make_grounding_result()],
        evidence_count=1,
        aggregated_confidence=0.95,
        unresolved_disagreement=True,
    )

    assert result.allowed is False