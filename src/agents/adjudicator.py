from src.llm.client import LLMClient
from src.models.adjudication import AdjudicationResult
from src.models.disagreement import DisagreementResult
from src.models.evaluation import EvaluationResult
from src.models.inputs import EvaluationInput


class Adjudicator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def adjudicate(
        self,
        evaluation_input: EvaluationInput,
        evaluations: list[EvaluationResult],
        disagreement: DisagreementResult,
    ) -> AdjudicationResult:

        evaluation_details = "\n\n".join(
            [
                f"""
Criterion: {evaluation.criterion}
Score: {evaluation.score}
Confidence: {evaluation.confidence}
Reasoning: {evaluation.reasoning}
Issues: {evaluation.issues}
"""
                for evaluation in evaluations
            ]
        )

        prompt = f"""
You are the final adjudicator in a multi-agent LLM evaluation system.

Your task is to make the FINAL judgment about the candidate response
after reviewing the independent evaluations from the accuracy, logic,
and completeness evaluators.

================ QUESTION ================
{evaluation_input.question}

============= CANDIDATE RESPONSE =========
{evaluation_input.response}

============= REFERENCE ANSWER ===========
{evaluation_input.reference_answer}

=========== EVALUATOR RESULTS ============
{evaluation_details}

=========== DISAGREEMENT ANALYSIS =========
Has disagreement: {disagreement.has_disagreement}
Severity: {disagreement.severity}
Score spread: {disagreement.score_spread}
Reasons: {disagreement.reasons}

============================================

ADJUDICATION RULES:

1. Independently review the candidate response using the question,
   reference answer, and candidate response.

2. The evaluator results are REPORTS ABOUT the candidate response.
   They are NOT part of the candidate response itself.

3. Never attribute an evaluator's reasoning, finding, or issue to the
   candidate response.

4. When describing a problem, clearly distinguish between:
   - what the candidate said,
   - what an evaluator identified,
   - and your own final judgment.

5. Do NOT blindly average evaluator scores.

6. Resolve disagreements by examining the actual candidate response
   and the evidence provided by the evaluators.

7. A serious factual error affecting the central answer must strongly
   reduce the final score.

8. A serious logical error affecting the main or final conclusion must
   strongly reduce the final score.

9. If the candidate's main or final conclusion depends on an invalid
   inference, this is a significant error and the final score MUST be
   5 or lower.

10. If the candidate contains a direct contradiction between important
    statements, the final score MUST be 5 or lower.

11. A major missing requirement must reduce the final score.

12. Do not classify a central factual or logical error as a minor
    weakness merely because other parts of the response are correct.

13. Do not treat evaluator disagreement itself as an error in the
    candidate response. Evaluate the candidate based on the evidence.

14. Confidence represents how confident you are in the FINAL JUDGMENT,
    not how correct the candidate response is.

15. The final reasoning MUST be consistent with the final score,
    verdict, and issues.

16. If issues are present, the reasoning must acknowledge those issues.

17. If there are no issues, the reasoning must explain why the response
    is acceptable across the relevant dimensions.
FINAL VERDICT SCALE:

Correct:
The response is accurate, logically sound, and sufficiently complete.

Mostly Correct:
The response is substantially correct but contains a minor or
non-central weakness.

Partially Correct:
The response contains meaningful problems but still provides
substantial correct or useful information.

Incorrect:
The central answer is wrong, fundamentally unsupported, or has
severe problems that make the response unacceptable.

FINAL SCORE:

10 = Fully acceptable with no substantive problems.
8-9 = Strong response with only minor, non-central weaknesses.
6-7 = Substantially useful response with meaningful but non-central
      problems.
4-5 = Significant problem affecting an important part of the answer.
2-3 = Major problem affecting the central answer or reasoning.
1 = Fundamentally unacceptable.

MANDATORY OVERRIDES:

- If the main/final conclusion depends on an invalid inference,
  final_score MUST be 5 or lower.

- If there is a clear direct contradiction in the candidate response,
  final_score MUST be 5 or lower.

- If the central factual answer is clearly incorrect,
  final_score MUST be 3 or lower.
Return ONLY the structured evaluation.
"""
        
        raw_result = self.llm_client.generate(
            prompt=prompt,
            response_format=AdjudicationResult.model_json_schema(),
        )

        adjudication = AdjudicationResult.model_validate_json(raw_result)

        return adjudication