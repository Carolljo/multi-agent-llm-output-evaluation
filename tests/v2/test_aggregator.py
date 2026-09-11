import pytest

from src.v2.aggregation.aggregator import aggregate_evaluations
from src.v2.models.schemas import EvaluatorResult


def make_result(
    evaluator: str,
    score: float,
    confidence: float,
    errors: list[str] | None = None,
    evidence_refs: list[str] | None = None,
) -> EvaluatorResult:
    return EvaluatorResult(
        evaluator=evaluator,
        score=score,
        confidence=confidence,
        claims_evaluated=3,
        errors=errors or [],
        reasoning=f"{evaluator} reasoning",
        evidence_refs=evidence_refs or [],
    )


def test_aggregate_scores_and_confidence():
    evaluations = [
        make_result("accuracy", 1.0, 0.9),
        make_result("completeness", 0.8, 0.8),
        make_result("logical_consistency", 0.6, 0.7),
        make_result("evidence_grounding", 0.4, 0.6),
    ]

    result = aggregate_evaluations(evaluations)

    assert result.score == pytest.approx(0.7)
    assert result.confidence == pytest.approx(0.75)
    assert len(result.evaluations) == 4


def test_aggregate_combines_errors_with_provenance():
    evaluations = [
        make_result(
            "accuracy",
            0.8,
            0.9,
            errors=["claim c1 is contradicted"],
        ),
        make_result(
            "completeness",
            0.6,
            0.8,
            errors=["claim c3 is missing"],
        ),
        make_result("logical_consistency", 1.0, 0.9),
        make_result(
            "evidence_grounding",
            0.7,
            0.8,
            errors=["claim c4 is ungrounded"],
        ),
    ]

    result = aggregate_evaluations(evaluations)

    assert result.errors == [
        "accuracy: claim c1 is contradicted",
        "completeness: claim c3 is missing",
        "evidence_grounding: claim c4 is ungrounded",
    ]


def test_aggregate_deduplicates_evidence_refs():
    evaluations = [
        make_result("accuracy", 1.0, 0.9, evidence_refs=["e1", "e2"]),
        make_result("completeness", 1.0, 0.9, evidence_refs=["e2", "e3"]),
        make_result("logical_consistency", 1.0, 0.9, evidence_refs=["e1"]),
        make_result("evidence_grounding", 1.0, 0.9, evidence_refs=["e3", "e4"]),
    ]

    result = aggregate_evaluations(evaluations)

    assert result.evidence_refs == ["e1", "e2", "e3", "e4"]


def test_aggregate_rejects_missing_evaluator():
    evaluations = [
        make_result("accuracy", 1.0, 0.9),
        make_result("completeness", 1.0, 0.9),
        make_result("logical_consistency", 1.0, 0.9),
    ]

    with pytest.raises(ValueError, match="evidence_grounding"):
        aggregate_evaluations(evaluations)


def test_aggregate_rejects_duplicate_evaluator():
    evaluations = [
        make_result("accuracy", 1.0, 0.9),
        make_result("accuracy", 0.8, 0.8),
        make_result("completeness", 1.0, 0.9),
        make_result("logical_consistency", 1.0, 0.9),
        make_result("evidence_grounding", 1.0, 0.9),
    ]

    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_evaluations(evaluations)