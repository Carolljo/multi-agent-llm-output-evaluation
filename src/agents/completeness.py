from src.llm.client import LLMClient
from src.models.evaluation import EvaluationResult
from src.models.inputs import EvaluationInput


class CompletenessEvaluator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        prompt = f"""
You are a strict completeness evaluator.

Your task is to evaluate ONLY the completeness of the candidate response.

Determine whether the candidate response adequately addresses all
important requirements of the question.

Question:
{evaluation_input.question}

Candidate Response:
{evaluation_input.response}

Reference Answer:
{evaluation_input.reference_answer}

IMPORTANT PRINCIPLE:

Completeness measures whether the candidate answered what the question
actually asks.

Do NOT require the candidate to include every piece of information
contained in the reference answer.

Do NOT penalize the candidate for omitting information that was not
requested by the question.

Do NOT treat verbosity or response length as evidence of completeness.

Follow this evaluation procedure:

STEP 1 — IDENTIFY REQUIREMENTS

Identify the explicit information, questions, tasks, or requirements
contained in the question.

STEP 2 — DETERMINE EXPECTED COVERAGE

Use the reference answer as supporting context to understand what would
constitute a valid answer to those requirements.

Do not treat the reference answer as a checklist that must be reproduced
word-for-word or in its entirety.

STEP 3 — CHECK CANDIDATE COVERAGE

For each important requirement, determine whether the candidate response:

- Fully addresses it
- Partially addresses it
- Fails to address it

STEP 4 — IGNORE UNREQUESTED INFORMATION

Do not penalize the candidate for omitting additional information that
appears in the reference answer but was not required by the question.

STEP 5 — DETERMINE THE SCORE

Use this scale:

10 = Fully complete. All important requirements are adequately addressed.

8-9 = Nearly complete. All major requirements are addressed with only
      minor omissions.
IMPORTANT SCORING RULE:

If the question contains multiple explicit requirements and the candidate
completely misses one or more major requirements, the score MUST be 7 or
lower.

Do not assign 8 or higher when a major explicitly requested component is
completely missing.

A response that correctly answers only part of a multi-part question is
not "nearly complete" merely because the answered part is correct.

6-7 = Mostly complete. Most important requirements are addressed, but
      one meaningful requirement is incomplete or partially addressed.

4-5 = Partially complete. Several important requirements are missing
      or inadequately addressed.

2-3 = Mostly incomplete. Only a small portion of the requested
      information is addressed.

1 = Essentially unanswered or fails to address the requested task.

IMPORTANT:

A concise response can receive a high completeness score if it fully
answers the question.

A long response can receive a low completeness score if it fails to
address important requirements.

Do not confuse factual correctness with completeness.

The candidate may be factually accurate while still being incomplete.

The criterion field MUST be exactly "completeness".

The score MUST be an integer from 1 to 10.

The reasoning must explain which important requirements were covered,
partially covered, or missing.

The issues list must contain significant missing or incomplete
requirements.

If all important requirements are adequately addressed, issues should
be empty.

CONFIDENCE FORMAT:

The confidence field represents how confident you are that your
completeness evaluation is correct.

It does NOT represent how complete the candidate response is.

Confidence MUST be a decimal number between 0.0 and 1.0.

Valid examples:
0.95
0.9
0.75
0.5

Invalid examples:
5
7
8
9
10

Return ONLY the structured evaluation.
"""

        raw_result = self.llm_client.generate(
            prompt=prompt,
            response_format=EvaluationResult.model_json_schema(),
        )

        evaluation = EvaluationResult.model_validate_json(raw_result)

        return evaluation