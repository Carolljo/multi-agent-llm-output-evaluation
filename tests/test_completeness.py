from src.agents.completeness import CompletenessEvaluator
from src.llm.client import LLMClient
from src.models.inputs import EvaluationInput


llm = LLMClient(model="qwen3:1.7b")
evaluator = CompletenessEvaluator(llm)


def test_complete_answer():
    evaluation_input = EvaluationInput(
        question=(
            "What is the capital of France and which river runs through it?"
        ),
        reference_answer=(
            "Paris is the capital of France, and the Seine runs through the city."
        ),
        response=(
            "Paris is the capital of France, and the Seine runs through it."
        ),
    )

    result = evaluator.evaluate(evaluation_input)

    print(result)

    assert result.criterion == "completeness"
    assert result.score >= 8
    assert len(result.issues) == 0


def test_incomplete_answer():
    evaluation_input = EvaluationInput(
        question=(
            "What is the capital of France and which river runs through it?"
        ),
        reference_answer=(
            "Paris is the capital of France, and the Seine runs through the city."
        ),
        response="Paris is the capital of France.",
    )

    result = evaluator.evaluate(evaluation_input)

    print(result)

    assert result.criterion == "completeness"
    assert result.score < 8
    assert len(result.issues) > 0


def test_unrequested_reference_information():
    evaluation_input = EvaluationInput(
        question="What is the capital of France?",
        reference_answer=(
            "Paris is the capital of France. "
            "It lies along the Seine and is a major cultural and political center."
        ),
        response="Paris is the capital of France.",
    )

    result = evaluator.evaluate(evaluation_input)

    print(result)

    assert result.criterion == "completeness"
    assert result.score >= 8
    assert len(result.issues) == 0
    
def test_wrong_but_complete_answer():
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

    result = evaluator.evaluate(evaluation_input)

    print(result)

    assert result.criterion == "completeness"
    assert result.score >= 8
    assert len(result.issues) == 0