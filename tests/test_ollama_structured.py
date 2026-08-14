from src.llm.client import LLMClient
from src.models.evaluation import EvaluationResult


llm = LLMClient(model="qwen3:1.7b")


question = "What is the capital of France?"
response = "The capital of France is Berlin."


prompt = f"""
You are an accuracy evaluator.

Evaluate whether the following response correctly answers the question.

Question:
{question}

Response:
{response}

Evaluate ONLY the accuracy of the response.

The criterion field MUST be exactly "accuracy".

The confidence field represents how confident you are in your evaluation judgment,
not whether the answer itself is correct.

Return a score from 1 to 10.
Explain your reasoning.
List any factual issues.
Provide your confidence from 0.0 to 1.0.
"""


raw_result = llm.generate(
    prompt=prompt,
    response_format=EvaluationResult.model_json_schema(),
)


evaluation = EvaluationResult.model_validate_json(raw_result)

print(evaluation)