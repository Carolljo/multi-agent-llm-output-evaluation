from src.agents.adjudicator import Adjudicator
from src.llm.client import LLMClient
from src.models.disagreement import DisagreementResult
from src.models.evaluation import EvaluationResult
from src.models.inputs import EvaluationInput


llm = LLMClient(model="qwen3:1.7b")
adjudicator = Adjudicator(llm)


def test_adjudicator_resolves_logical_disagreement():
    evaluation_input = EvaluationInput(
        question=(
            "If all employees who complete security training receive "
            "a certificate, and Rahul received a certificate, "
            "what can we conclude about Rahul?"
        ),
        reference_answer=(
            "We cannot conclude that Rahul completed the training "
            "solely because he received a certificate."
        ),
        response=(
            "All employees who complete the training receive a certificate. "
            "Rahul received a certificate. Therefore, Rahul completed "
            "the training."
        ),
    )

    evaluations = [
        EvaluationResult(
            criterion="accuracy",
            score=8,
            reasoning="The response addresses the question but makes an unsupported conclusion.",
            issues=["The conclusion is not supported by the given information."],
            confidence=0.9,
        ),
        EvaluationResult(
            criterion="logic",
            score=3,
            reasoning="The response commits the fallacy of affirming the consequent.",
            issues=["The conclusion does not logically follow from the premise."],
            confidence=0.95,
        ),
        EvaluationResult(
            criterion="completeness",
            score=9,
            reasoning="The response directly addresses the requested question.",
            issues=[],
            confidence=0.9,
        ),
    ]

    disagreement = DisagreementResult(
        has_disagreement=True,
        severity="high",
        score_spread=6,
        reasons=["Evaluator scores differ by 6 points."],
    )

    result = adjudicator.adjudicate(
        evaluation_input,
        evaluations,
        disagreement,
    )

    print(result)

    assert result.final_verdict in {
        "Partially Correct",
        "Incorrect",
    }

    assert result.final_score <= 7
    assert result.confidence >= 0.0
    assert len(result.reasoning) > 0
    assert len(result.issues) > 0


def test_adjudicator_accepts_strong_agreement():
    evaluation_input = EvaluationInput(
        question="What is the capital of France?",
        reference_answer="The capital of France is Paris.",
        response="The capital of France is Paris.",
    )

    evaluations = [
        EvaluationResult(
            criterion="accuracy",
            score=10,
            reasoning="The answer is factually correct.",
            issues=[],
            confidence=0.98,
        ),
        EvaluationResult(
            criterion="logic",
            score=10,
            reasoning="The response contains no logical problems.",
            issues=[],
            confidence=0.98,
        ),
        EvaluationResult(
            criterion="completeness",
            score=10,
            reasoning="The response fully answers the question.",
            issues=[],
            confidence=0.98,
        ),
    ]

    disagreement = DisagreementResult(
        has_disagreement=False,
        severity="none",
        score_spread=0,
        reasons=[],
    )

    result = adjudicator.adjudicate(
        evaluation_input,
        evaluations,
        disagreement,
    )

    print(result)

    assert result.final_verdict == "Correct"
    assert result.final_score >= 8
    assert result.confidence >= 0.8
    assert len(result.issues) == 0