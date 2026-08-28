from src.aggregation.disagreement import DisagreementDetector
from src.models.evaluation import EvaluationResult


def make_result(criterion, score):
    return EvaluationResult(
        criterion=criterion,
        score=score,
        reasoning=f"Test reasoning for {criterion}.",
        issues=[],
        confidence=0.9,
    )


def test_no_disagreement():
    evaluations = [
        make_result("accuracy", 9),
        make_result("logic", 8),
        make_result("completeness", 9),
    ]

    result = DisagreementDetector().detect(evaluations)

    assert result.has_disagreement is False
    assert result.severity == "none"
    assert result.score_spread == 1
    assert result.reasons == []


def test_medium_disagreement():
    evaluations = [
        make_result("accuracy", 9),
        make_result("logic", 6),
        make_result("completeness", 8),
    ]

    result = DisagreementDetector().detect(evaluations)

    assert result.has_disagreement is True
    assert result.severity == "medium"
    assert result.score_spread == 3
    assert len(result.reasons) == 1


def test_high_disagreement():
    evaluations = [
        make_result("accuracy", 9),
        make_result("logic", 3),
        make_result("completeness", 9),
    ]

    result = DisagreementDetector().detect(evaluations)

    assert result.has_disagreement is True
    assert result.severity == "high"
    assert result.score_spread == 6
    assert len(result.reasons) == 1


def test_agreement_on_low_scores():
    evaluations = [
        make_result("accuracy", 3),
        make_result("logic", 3),
        make_result("completeness", 3),
    ]

    result = DisagreementDetector().detect(evaluations)

    assert result.has_disagreement is False
    assert result.severity == "none"
    assert result.score_spread == 0


def test_boundary_for_medium_disagreement():
    evaluations = [
        make_result("accuracy", 9),
        make_result("logic", 6),
        make_result("completeness", 9),
    ]

    result = DisagreementDetector().detect(evaluations)

    assert result.has_disagreement is True
    assert result.severity == "medium"
    assert result.score_spread == 3