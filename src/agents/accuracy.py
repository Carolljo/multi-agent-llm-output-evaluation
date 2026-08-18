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

Candidate Response:
{evaluation_input.response}

Reference answer:
{evaluation_input.reference_answer}

Evaluate ONLY the accuracy of the response.

The criterion field MUST be exactly "accuracy".

Reference answer is the factual grounding for the evaluation.
Compare the meaning of the Candidate Response with the Reference Answer, not the exact wording.
The confidence field represents how confident you are in your evaluation judgment,
not whether the answer itself is correct.
Do not treat missing information from the Candidate Response as a factual error.
Do not penalize the Candidate Response for being less detailed than the Reference Answer unless it makes an incorrect claim.
Return a score from 1 to 10.
Explain your reasoning.
List any factual issues.
Provide your confidence from 0.0 to 1.0.
Scoring guidelines:

10 = Fully factually accurate with no substantive errors.

8-9 = Highly accurate with no meaningful factual errors.
     Only very minor imprecision that does not change the answer
     to a requested fact may receive this range.

6-7 = Mostly accurate but contains a meaningful factual error,
     while the main answer remains substantially correct.

4-5 = Partially accurate but contains significant factual errors
     that materially affect the answer.

2-3 = Mostly inaccurate; the central answer is incorrect or major
     factual errors substantially outweigh the correct information.

1 = Fundamentally or entirely incorrect.

IMPORTANT:
If the candidate gives a clearly incorrect answer to the central
question or to the only fact explicitly requested, the score MUST
be 3 or lower.

If a response answers multiple explicitly requested facts and one
major requested fact is incorrect while another is correct, the
score should normally be 6-7.

Confidence represents how confident you are that your evaluation
judgment, including the score and identified issues, is correct.
Do not use confidence to represent how correct the candidate response is.

Your reasoning and issues must be internally consistent.
If you identify a factual issue, do not state that there are no factual issues.

Do not treat missing information from the Candidate Response as a factual error.
Do not penalize the Candidate Response for being less detailed than the
Reference Answer unless it makes an incorrect claim.
"""

        raw_result = self.llm_client.generate(
            prompt=prompt,
            response_format=EvaluationResult.model_json_schema(),
        )

        evaluation = EvaluationResult.model_validate_json(raw_result)

        return evaluation