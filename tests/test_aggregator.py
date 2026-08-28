import pytest

from src.aggregation.aggregator import EvaluatorAggregator
from src.models.evaluation import EvaluationResult


def make_result(criterion, score, confidence=0.9, issues=None):
    return EvaluationResult(
        criterion=criterion,
        score=score,
        reasoning=f"Test reasoning for {criterion}.",
        issues=issues or [],
        confidence=confidence,
    )


def test_strong_evaluation():
    accuracy = make_result("accuracy", 9)
    logic = make_result("logic", 9)
    completeness = make_result("completeness", 9)

    result = EvaluatorAggregator().aggregate(
        [accuracy, logic, completeness]
    )

    print(result)

    assert result.overall_score == 9
    assert result.overall_confidence == pytest.approx(0.9)
    assert result.needs_adjudication is False
    assert "factually accurate" in result.summary


def test_factually_strong_logically_weak():
    accuracy = make_result("accuracy", 9)
    logic = make_result(
        "logic",
        3,
        issues=["The conclusion is not supported by the premises."]
    )
    completeness = make_result("completeness", 9)

    result = EvaluatorAggregator().aggregate(
        [accuracy, logic, completeness]
    )

    print(result)

    assert result.overall_score == 5
    assert "logical weaknesses" in result.summary
    assert len(result.key_issues) == 1


def test_logically_strong_factually_weak():
    accuracy = make_result(
        "accuracy",
        3,
        issues=["The response contains a factual error."]
    )
    logic = make_result("logic", 9)
    completeness = make_result("completeness", 9)

    result = EvaluatorAggregator().aggregate(
        [accuracy, logic, completeness]
    )

    print(result)

    assert result.overall_score == 5
    assert "factual" in result.summary


def test_both_weak():
    accuracy = make_result(
        "accuracy",
        3,
        issues=["Major factual error."]
    )
    logic = make_result(
        "logic",
        2,
        issues=["Major logical error."]
    )
    completeness = make_result("completeness", 9)

    result = EvaluatorAggregator().aggregate(
        [accuracy, logic, completeness]
    )

    print(result)

    assert result.overall_score == 4.5
    assert len(result.key_issues) == 2
    
def test_completeness_weak():
    accuracy = make_result("accuracy", 9)
    logic = make_result("logic", 9)
    completeness = make_result(
        "completeness",
        3,
        issues=["A major requested requirement is missing."]
    )

    result = EvaluatorAggregator().aggregate(
        [accuracy, logic, completeness]
    )

    print(result)

    assert result.overall_score == 5
    assert len(result.key_issues) == 1
    assert "incomplete" in result.summary