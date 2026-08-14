from src.llm.client import LLMClient
from src.models.evaluation import EvaluationResult
from src.models.inputs import EvaluationInput


class AccuracyEvaluator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        prompt = f"""
You are an accuracy evaluator.

Evaluate whether the following response correctly answers the question.

Question:
{evaluation_input.question}

Response:
{evaluation_input.response}

Evaluate ONLY the accuracy of the response.

The criterion field MUST be exactly "accuracy".

The confidence field represents how confident you are in your evaluation judgment,
not whether the answer itself is correct.

Return a score from 1 to 10.
Explain your reasoning.
List any factual issues.
Provide your confidence from 0.0 to 1.0.
"""

        raw_result = self.llm_client.generate(
            prompt=prompt,
            response_format=EvaluationResult.model_json_schema(),
        )

        evaluation = EvaluationResult.model_validate_json(raw_result)

        return evaluation