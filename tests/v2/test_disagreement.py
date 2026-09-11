import pytest

from src.v2.aggregation.disagreement import detect_disagreement
from src.v2.models.schemas import EvaluatorResult


def make_result(
    evaluator: str,
    score: float,
    confidence: float = 0.9,
) -> EvaluatorResult:
    return EvaluatorResult(
        evaluator=evaluator,
        score=score,
        confidence=confidence,
        claims_evaluated=3,
        errors=[],
        reasoning=f"{evaluator} reasoning",
        evidence_refs=[],
    )


def test_no_disagreement_when_scores_are_close():
    evaluations = [
        make_result("accuracy", 0.90),
        make_result("completeness", 0.85),
        make_result("logical_consistency", 0.80),
        make_result("evidence_grounding", 0.75),
    ]

    result = detect_disagreement(evaluations)

    assert result.detected is False
    assert result.score_range == pytest.approx(0.15)
    assert result.requires_adjudication is False


def test_disagreement_detected_when_score_range_exceeds_threshold():
    evaluations = [
        make_result("accuracy", 0.95),
        make_result("completeness", 0.90),
        make_result("logical_consistency", 0.85),
        make_result("evidence_grounding", 0.40),
    ]

    result = detect_disagreement(evaluations)

    assert result.detected is True
    assert result.score_range == pytest.approx(0.55)
    assert result.requires_adjudication is True


def test_disagreement_identifies_high_and_low_evaluators():
    evaluations = [
        make_result("accuracy", 0.95),
        make_result("completeness", 0.80),
        make_result("logical_consistency", 0.85),
        make_result("evidence_grounding", 0.40),
    ]

    result = detect_disagreement(evaluations)

    assert result.detected is True
    assert "accuracy" in result.evaluators
    assert "evidence_grounding" in result.evaluators


def test_threshold_boundary_is_treated_as_disagreement():
    evaluations = [
        make_result("accuracy", 0.90),
        make_result("completeness", 0.80),
        make_result("logical_consistency", 0.70),
        make_result("evidence_grounding", 0.60),
    ]

    result = detect_disagreement(evaluations)

    assert result.score_range == pytest.approx(0.30)
    assert result.detected is True
    assert result.requires_adjudication is True


def test_no_disagreement_when_all_scores_are_identical():
    evaluations = [
        make_result("accuracy", 0.80),
        make_result("completeness", 0.80),
        make_result("logical_consistency", 0.80),
        make_result("evidence_grounding", 0.80),
    ]

    result = detect_disagreement(evaluations)

    assert result.detected is False
    assert result.score_range == pytest.approx(0.0)
    assert result.requires_adjudication is False


def test_disagreement_reason_is_populated():
    evaluations = [
        make_result("accuracy", 1.0),
        make_result("completeness", 0.90),
        make_result("logical_consistency", 0.85),
        make_result("evidence_grounding", 0.30),
    ]

    result = detect_disagreement(evaluations)

    assert result.detected is True
    assert result.reason
    assert "0.70" in result.reason