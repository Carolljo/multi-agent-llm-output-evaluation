from src.graph.nodes import finalize_result
from src.models.disagreement import DisagreementResult
from src.models.evaluation import EvaluationResult
from src.models.inputs import EvaluationInput
from src.models.summary import EvaluationSummary
from src.models.adjudication import AdjudicationResult

def create_evaluation(
    criterion: str,
    score: int,
) -> EvaluationResult:
    return EvaluationResult(
        criterion=criterion,
        score=score,
        reasoning=f"{criterion} evaluation.",
        issues=[],
        confidence=0.9,
    )


def create_state(scores: dict) -> dict:
    evaluations = [
        create_evaluation("accuracy", scores["accuracy"]),
        create_evaluation("logic", scores["logic"]),
        create_evaluation("completeness", scores["completeness"]),
    ]

    disagreement = DisagreementResult(
        has_disagreement=False,
        severity="none",
        score_spread=1,
        reasons=[],
    )

    summary = EvaluationSummary(
        evaluations=evaluations,
        overall_score=sum(scores.values()) / 3,
        overall_confidence=0.9,
        summary="Evaluation summary.",
        key_issues=[],
        needs_adjudication=False,
        disagreement=disagreement,
    )

    return {
        "evaluation_input": EvaluationInput(
            question="Test question",
            reference_answer="Test reference",
            response="Test response",
        ),
        "accuracy": evaluations[0],
        "logic": evaluations[1],
        "completeness": evaluations[2],
        "evaluations": evaluations,
        "evaluation_summary": summary,
        "disagreement": disagreement,
        "adjudication": None,
        "final_result": None,
    }


def test_finalization_returns_correct():
    state = create_state(
        {
            "accuracy": 10,
            "logic": 10,
            "completeness": 10,
        }
    )

    result = finalize_result(state)

    assert result["final_result"].verdict == "Correct"


def test_finalization_returns_mostly_correct():
    state = create_state(
        {
            "accuracy": 8,
            "logic": 8,
            "completeness": 8,
        }
    )

    result = finalize_result(state)

    assert result["final_result"].verdict == "Mostly Correct"


def test_finalization_returns_partially_correct():
    state = create_state(
        {
            "accuracy": 7,
            "logic": 7,
            "completeness": 8,
        }
    )

    result = finalize_result(state)

    assert result["final_result"].verdict == "Partially Correct"


def test_finalization_returns_incorrect_for_central_failure():
    state = create_state(
        {
            "accuracy": 9,
            "logic": 4,
            "completeness": 9,
        }
    )

    result = finalize_result(state)

    assert result["final_result"].verdict == "Incorrect"
    
def test_finalization_uses_adjudication_result():
    state = create_state(
        {
            "accuracy": 9,
            "logic": 3,
            "completeness": 9,
        }
    )

    state["adjudication"] = AdjudicationResult(
        final_verdict="Partially Correct",
        final_score=6,
        confidence=0.92,
        reasoning="The response contains a significant logical problem.",
        issues=["Invalid inference."],
    )

    result = finalize_result(state)

    final_result = result["final_result"]

    assert final_result.verdict == "Partially Correct"
    assert final_result.score == 6
    assert final_result.confidence == 0.92
    assert final_result.reasoning == (
        "The response contains a significant logical problem."
    )
    assert final_result.issues == ["Invalid inference."]