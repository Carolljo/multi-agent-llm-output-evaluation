import json

from src.v2.llm.client import LLMClient
from src.v2.models.schemas import (
    Adjudication,
    Claim,
    Disagreement,
    Evidence,
    EvaluatorResult,
    ReferenceAnswer,
)


class Adjudicator:
    """Resolve evaluator disagreement using the configured LLM."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def adjudicate(
        self,
        question: str,
        response: str,
        reference_answer: ReferenceAnswer,
        claims: list[Claim],
        evidence: list[Evidence],
        evaluations: list[EvaluatorResult],
        disagreement: Disagreement,
    ) -> Adjudication:
        """Resolve evaluator disagreement."""

        if not disagreement.requires_adjudication:
            raise ValueError("Adjudication is not required")

        if not evaluations:
            raise ValueError(
                "At least one evaluator result is required."
            )

        if not evidence:
            raise ValueError(
                "At least one evidence item is required."
            )

        prompt = self._build_prompt(
            question=question,
            response=response,
            reference_answer=reference_answer,
            claims=claims,
            evidence=evidence,
            evaluations=evaluations,
            disagreement=disagreement,
        )

        raw_response = self.llm_client.generate(
            prompt=prompt,
            response_format=Adjudication.model_json_schema(),
        )

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Adjudicator returned invalid JSON."
            ) from exc

        try:
            adjudication = Adjudication.model_validate(data)
        except ValueError as exc:
            raise ValueError(
                "Adjudicator returned invalid structured output."
            ) from exc

        self._validate_evidence_refs(
            adjudication.evidence_refs,
            evidence,
        )

        return adjudication

    @staticmethod
    def _validate_evidence_refs(
        evidence_refs: list[str],
        evidence: list[Evidence],
    ) -> None:
        """Validate that all referenced evidence IDs exist."""

        valid_ids = {
            item.evidence_id
            for item in evidence
        }

        invalid_refs = [
            ref
            for ref in evidence_refs
            if ref not in valid_ids
        ]

        if invalid_refs:
            raise ValueError(
                "Adjudicator returned invalid evidence "
                f"reference(s): {', '.join(invalid_refs)}"
            )

    @staticmethod
    def _build_prompt(
        *,
        question: str,
        response: str,
        reference_answer: ReferenceAnswer,
        claims: list[Claim],
        evidence: list[Evidence],
        evaluations: list[EvaluatorResult],
        disagreement: Disagreement,
    ) -> str:
        """Build the evidence-grounded adjudication prompt."""

        return f"""
You are the adjudicator in a multi-agent LLM evaluation system.

Your task is to resolve disagreement between independent evaluators.

Do NOT perform a completely new evaluation.

Instead, analyze the evaluator results, claims, evidence,
reference answer, and disagreement information and determine
the most defensible final verdict.

Allowed verdicts:

- accept:
  The response is substantially correct and acceptable.

- partial:
  The response contains meaningful correct content but also
  has material issues.

- reject:
  The response is substantially incorrect, unsupported,
  or unacceptable.

You must:

1. Consider all evaluator results.
2. Pay particular attention to the evaluators involved
   in the disagreement.
3. Examine the relevant claims and supplied evidence.
4. Use ONLY the supplied evidence.
5. Do NOT invent evidence references.
6. Explain why the disagreement should be resolved
   the way you decided.
7. Return confidence between 0 and 1.

Do NOT return a numerical evaluation score.
The application calculates the final score deterministically.

QUESTION:
{question}

CANDIDATE RESPONSE:
{response}

REFERENCE ANSWER:
{reference_answer.model_dump_json(indent=2)}

CLAIMS:
{json.dumps(
    [claim.model_dump() for claim in claims],
    indent=2,
)}

EVIDENCE:
{json.dumps(
    [item.model_dump() for item in evidence],
    indent=2,
)}

EVALUATOR RESULTS:
{json.dumps(
    [evaluation.model_dump() for evaluation in evaluations],
    indent=2,
)}

DISAGREEMENT:
{disagreement.model_dump_json(indent=2)}

Return ONLY valid JSON matching the requested schema.
""".strip()
