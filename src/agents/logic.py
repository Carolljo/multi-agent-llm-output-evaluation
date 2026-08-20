from src.llm.client import LLMClient
from src.models.evaluation import EvaluationResult
from src.models.inputs import EvaluationInput


class LogicEvaluator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        prompt = f"""
You are a strict logic evaluator.

Your task is to evaluate ONLY the logical quality of the candidate response.

Do NOT judge the response as correct merely because part of it agrees with
the reference answer.

Question:
{evaluation_input.question}

Candidate Response:
{evaluation_input.response}

Reference Answer:
{evaluation_input.reference_answer}

IMPORTANT:
The reference answer is provided as context only.
Do NOT use agreement with the reference answer as proof that the candidate
response is logically sound.

You must independently analyze the candidate response.

Follow this evaluation procedure:

STEP 1 — IDENTIFY CLAIMS

Break the candidate response into its individual claims or statements.

For example:

"The server was offline. Therefore, it processed requests."

contains at least two claims:

A. The server was offline.
B. The server processed requests.

STEP 2 — CHECK INTERNAL CONSISTENCY

Compare every important claim with the other claims in the response.

Look specifically for statements that cannot both be true under the same
conditions.

Examples of contradictions:

- "The server was offline" + "The server processed requests continuously."
- "Rahul never attended the meeting" + "Rahul presented during the meeting."
- "The database was deleted" + "The database remained available."
- "The device had no network connection" + "The device successfully uploaded data."

If the response contains a direct contradiction, you MUST identify it as a
logical issue.

A response must NOT receive a high score if it contains a clear contradiction,
even if another part of the response agrees with the reference answer.

STEP 3 — CHECK INFERENCE VALIDITY

For every conclusion:

- Identify the premises supporting it.
- Determine whether the conclusion actually follows from those premises.
- Detect unsupported assumptions.
- Detect invalid inference patterns.
- Do not assume missing premises are true.

STEP 4 — CHECK THE CONCLUSION

Determine whether the final conclusion is logically supported by the reasoning
given in the candidate response.

Do not simply compare the conclusion with the reference answer.

STEP 5 — DETERMINE THE SCORE

Use this scale:

10 = Completely logically sound. No contradictions, invalid inferences,
     unsupported jumps, or meaningful reasoning problems.

8-9 = Strong reasoning with only minor logical weaknesses.

6-7 = Mostly logical but contains a meaningful reasoning weakness.

4-5 = Significant logical problems, such as a contradiction or major
     unsupported inference.

2-3 = Major logical errors, multiple contradictions, or severely invalid
     reasoning.

1 = Fundamentally invalid, contradictory, or incoherent reasoning.

IMPORTANT SCORING RULE:

If the candidate response contains a clear direct contradiction between
statements, the score MUST be 5 or lower.

If the candidate response contains multiple severe contradictions or
fundamentally incoherent reasoning, the score should normally be 1-3.

If the candidate reaches a conclusion by reversing the direction
of a one-way implication (affirming the consequent), this is a
significant logical error and the score MUST be 5 or lower.
If the candidate's main or final conclusion depends on an invalid
inference, the score MUST be 5 or lower, even if the rest of the
reasoning is internally coherent.

Do not classify a central invalid inference as merely a "meaningful
reasoning weakness" in the 6-7 range.
CONFIDENCE FORMAT:

The score and confidence are different fields.

score:
- Integer from 1 to 10.
- Example: 9

confidence:
- Decimal number between 0.0 and 1.0.
- NEVER use the 1-10 scale for confidence.
- NEVER return confidence as 5, 7, 8, 9, or 10.
- Valid examples: 0.95, 0.9, 0.75, 0.5
- Invalid examples: 5, 7, 9, 10

Example of correct output:
score = 9
confidence = 0.95

OUTPUT REQUIREMENTS:

Return ONLY the structured evaluation.

The criterion field MUST be exactly:
"logic"

The score MUST be an integer from 1 to 10.

The reasoning must explain the actual logical analysis.

The issues list MUST contain every significant logical issue identified.

If a contradiction is identified, explicitly describe which statements
contradict each other.

The confidence field represents how confident you are that your evaluation
judgment is correct.

It does NOT represent how logically correct the candidate response is.

IMPORTANT CONSISTENCY RULE:

If issues contains a logical problem, the reasoning MUST acknowledge that
problem.

Never say that the reasoning is completely sound when issues contains a
contradiction or invalid inference.

Never say "there are no logical issues" when issues is non-empty.
"""

        raw_result = self.llm_client.generate(
            prompt=prompt,
            response_format=EvaluationResult.model_json_schema(),
        )

        evaluation = EvaluationResult.model_validate_json(raw_result)

        return evaluation