from src.agents.logic import LogicEvaluator
from src.llm.client import LLMClient
from src.models.inputs import EvaluationInput


llm = LLMClient(model="qwen3:1.7b")
evaluator = LogicEvaluator(llm)


def test_valid_reasoning():
    evaluation_input = EvaluationInput(
        question=(
            "If all employees who complete security training receive a certificate, "
            "and Rahul completed the security training, what can we conclude?"
        ),
        reference_answer="Rahul receives a certificate.",
        response=(
            "Rahul completed the security training. "
            "All employees who complete the training receive a certificate. "
            "Therefore, Rahul receives a certificate."
        ),
    )

    result = evaluator.evaluate(evaluation_input)

    print(result)

    assert result.criterion == "logic"
    assert result.score >= 8
    assert len(result.issues) == 0


def test_invalid_inference():
    evaluation_input = EvaluationInput(
        question=(
            "If all employees who complete security training receive a certificate, "
            "and Rahul received a certificate, what can we conclude about Rahul?"
        ),
        reference_answer=(
            "We cannot conclude that Rahul completed the training solely because "
            "he received a certificate."
        ),
        response=(
            "All employees who complete the training receive a certificate. "
            "Rahul received a certificate. "
            "Therefore, Rahul completed the training."
        ),
    )

    result = evaluator.evaluate(evaluation_input)

    print(result)

    assert result.criterion == "logic"
    assert result.score <= 5
    assert len(result.issues) > 0


def test_contradictory_reasoning():
    evaluation_input = EvaluationInput(
        question="Was the server available during the maintenance window?",
        reference_answer=(
            "The server was unavailable during the maintenance window."
        ),
        response=(
            "The server was completely offline during the maintenance window. "
            "Therefore, it processed requests continuously during that same period."
        ),
    )

    result = evaluator.evaluate(evaluation_input)

    print(result)

    assert result.criterion == "logic"
    assert result.score <= 5
    assert len(result.issues) > 0