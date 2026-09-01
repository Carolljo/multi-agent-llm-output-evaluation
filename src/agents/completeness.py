from src.llm.client import LLMClient
from src.models.evaluation import EvaluationResult
from src.models.inputs import EvaluationInput


class CompletenessEvaluator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        prompt = f"""
You are evaluating ONLY the COMPLETENESS of an answer.

Your job is to determine whether the CANDIDATE RESPONSE answers every
explicit part of the QUESTION.

================ QUESTION ================
{evaluation_input.question}

============= CANDIDATE RESPONSE ==========
{evaluation_input.response}

============== REFERENCE ANSWER ===========
{evaluation_input.reference_answer}

============================================

IMPORTANT:

The three sections above are different.

- QUESTION = what the user asks for.
- CANDIDATE RESPONSE = what the candidate actually answered.
- REFERENCE ANSWER = supporting information that helps you understand
  the expected answer.

NEVER assume that information in the REFERENCE ANSWER appears in the
CANDIDATE RESPONSE.

ONLY information actually present in the CANDIDATE RESPONSE counts as
answered.

EVALUATION PROCEDURE:

1. Read the QUESTION.
2. Identify every explicit requirement in the QUESTION.
3. Check each requirement against the CANDIDATE RESPONSE.
4. Mark each requirement as:
   - fully addressed
   - partially addressed
   - missing
5. Use the REFERENCE ANSWER only to clarify what a correct answer to
   the requirement means.
6. Do NOT copy information from the REFERENCE ANSWER into the
   CANDIDATE RESPONSE.
7. Do NOT assume a requirement was answered just because it appears
   in the REFERENCE ANSWER.

IMPORTANT MULTI-PART RULE:

If the QUESTION asks for two or more distinct pieces of information
and the CANDIDATE RESPONSE answers only some of them, the response is
INCOMPLETE.

For example:

QUESTION:
"What is the capital of France and which river runs through it?"

CANDIDATE RESPONSE:
"Paris is the capital of France."

REFERENCE ANSWER:
"Paris is the capital of France, and the Seine runs through the city."

Correct completeness judgment:
- Capital → fully addressed
- River → missing
- Overall response → incomplete
- Score → 7 or lower

Do NOT give a score of 8 or higher when a major explicit requirement
is completely missing.

SCORING:

10 = All important requirements are fully addressed.
8-9 = All major requirements are addressed with only minor omissions.
6-7 = Most requirements are addressed, but one meaningful requirement
      is incomplete or partially addressed.
4-5 = Several important requirements are missing.
2-3 = Only a small portion of the requested information is addressed.
1   = Essentially unanswered.

Do NOT use response length as evidence of completeness.

Do NOT confuse factual correctness with completeness.

A response can be factually correct but incomplete.

OUTPUT REQUIREMENTS:

criterion MUST be exactly "completeness".

score MUST be an integer from 1 to 10.

reasoning MUST explain which explicit requirements were addressed,
partially addressed, or missing.

issues MUST contain ONLY problems with the CANDIDATE RESPONSE.

An issue is valid ONLY if the candidate response:
- misses an explicit requirement from the QUESTION,
- partially addresses an explicit requirement,
- or fails to provide information explicitly requested by the QUESTION.

NEVER create an issue because:
- the REFERENCE ANSWER contains additional information,
- the REFERENCE ANSWER is more detailed than the CANDIDATE RESPONSE,
- the CANDIDATE RESPONSE does not reproduce every sentence or detail
  from the REFERENCE ANSWER,
- the REFERENCE ANSWER contains information that was not requested
  by the QUESTION.

REFERENCE-ONLY INFORMATION IS NOT A COMPLETENESS ISSUE.

For example:

QUESTION:
"What is the capital of France?"

CANDIDATE RESPONSE:
"Paris is the capital of France."

REFERENCE ANSWER:
"Paris is the capital of France. It lies along the Seine and is a
major cultural and political center."

Correct output:
- score = 10
- issues = []

The Seine information is NOT required because the QUESTION does not
ask about the Seine.

If the QUESTION explicitly asked:
"What is the capital of France and which river runs through it?"

then omitting the river would be a valid completeness issue.

If all explicit requirements in the QUESTION are satisfied,
issues MUST be an empty list.

confidence MUST be a decimal between 0.0 and 1.0 and represents
confidence in the evaluation, NOT the completeness score.

Return ONLY the structured evaluation.
"""

        raw_result = self.llm_client.generate(
            prompt=prompt,
            response_format=EvaluationResult.model_json_schema(),
        )

        evaluation = EvaluationResult.model_validate_json(raw_result)

        return evaluation