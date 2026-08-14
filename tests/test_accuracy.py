from src.agents.accuracy import AccuracyEvaluator
from src.llm.client import LLMClient
from src.models.inputs import EvaluationInput


llm = LLMClient(model="qwen3:1.7b")
evaluator = AccuracyEvaluator(llm)


def test_incorrect_answer():
    evaluation_input = EvaluationInput(
        question="What is the capital of France?",
        response="The capital of France is Berlin.",
    )

    result = evaluator.evaluate(evaluation_input)

    assert result.criterion == "accuracy"
    assert result.score <= 3
    assert len(result.issues) > 0


def test_correct_answer():
    evaluation_input = EvaluationInput(
        question="What is the capital of France?",
        response="The capital of France is Paris.",
    )

    result = evaluator.evaluate(evaluation_input)

    assert result.criterion == "accuracy"
    assert result.score >= 8
    assert len(result.issues) == 0