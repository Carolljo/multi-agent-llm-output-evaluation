import json

from src.v2.llm.client import LLMClient
from src.v2.models.schemas import (
    Claim,
    Evidence,
    EvaluatorResult,
    ReferenceAnswer,
)


COMPLETENESS_EVALUATION_FORMAT = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "reference_claim_id": {
                        "type": "string",
                    },
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "covered",
                            "partially_covered",
                            "missing",
                        ],
                    },
                    "reasoning": {
                        "type": "string",
                    },
                    "candidate_claim_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                },
                "required": [
                    "reference_claim_id",
                    "verdict",
                    "reasoning",
                    "candidate_claim_ids",
                ],
            },
        },
        "confidence": {
            "type": "number",
        },
        "reasoning": {
            "type": "string",
        },
    },
    "required": [
        "claims",
        "confidence",
        "reasoning",
    ],
}


class CompletenessEvaluator:
    """Evaluate how completely the candidate covers the reference."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def evaluate(
        self,
        question: str,
        claims: list[Claim],
        reference_answer: ReferenceAnswer,
        evidence: list[Evidence],
    ) -> EvaluatorResult:
        """Evaluate coverage of important reference claims."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not claims:
            raise ValueError(
                "At least one candidate claim is required."
            )

        if not reference_answer.claims:
            raise ValueError(
                "Reference answer must contain at least one claim."
            )

        if not evidence:
            raise ValueError(
                "At least one evidence item is required."
            )

        candidate_claims_text = "\n".join(
            f"[{claim.claim_id}] {claim.text}"
            for claim in claims
        )

        reference_claims_text = "\n".join(
            (
                f"[{claim.claim_id}] {claim.text} "
                f"(importance: {claim.importance})"
            )
            for claim in reference_answer.claims
        )

        evidence_text = "\n\n".join(
            (
                f"[{item.evidence_id}] "
                f"Source: {item.source or 'unknown'}\n"
                f"{item.content}"
            )
            for item in evidence
        )

        prompt = f"""
You are a completeness evaluation component in an
LLM output evaluation system.

Your task is to determine whether the candidate response
contains the important information represented by the
validated reference answer.

Use ONLY the supplied reference claims, candidate claims,
and evidence.

Do NOT use outside knowledge.
Do NOT judge whether candidate claims are factually correct.
Do NOT penalize the candidate for stylistic differences.
Focus only on whether important reference information
is covered by the candidate response.

QUESTION:
{question}

REFERENCE CLAIMS:
{reference_claims_text}

CANDIDATE CLAIMS:
{candidate_claims_text}

RETRIEVED EVIDENCE:
{evidence_text}

For every reference claim, assign exactly one verdict:

- covered:
  The candidate response clearly communicates the
  information contained in the reference claim.

- partially_covered:
  The candidate communicates some but not all of the
  important information in the reference claim.

- missing:
  The candidate does not communicate the information
  represented by the reference claim.

RULES:
1. Evaluate every reference claim.
2. Preserve every reference claim_id exactly.
3. Identify which candidate claims provide the coverage.
4. Do not invent candidate claim IDs.
5. Consider the importance of each reference claim.
6. Do not judge factual accuracy.
7. Assign confidence between 0 and 1.
8. Provide an overall reasoning summary.
"""

        raw_response = self.llm_client.generate(
            prompt=prompt,
            response_format=COMPLETENESS_EVALUATION_FORMAT,
        )

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON."
            ) from exc

        evaluated_claims = data.get("claims", [])

        if len(evaluated_claims) != len(
            reference_answer.claims
        ):
            raise ValueError(
                "LLM did not evaluate every reference claim."
            )

        score = self._calculate_score(evaluated_claims)

        errors = [
            (
                f"{item['reference_claim_id']}: "
                f"{item['verdict']} - "
                f"{item['reasoning']}"
            )
            for item in evaluated_claims
            if item["verdict"] != "covered"
        ]

        candidate_claim_ids = {
            claim.claim_id
            for claim in claims
        }

        invalid_candidate_refs = sorted(
            {
                candidate_id
                for item in evaluated_claims
                for candidate_id in item.get(
                    "candidate_claim_ids",
                    [],
                )
                if candidate_id not in candidate_claim_ids
            }
        )

        if invalid_candidate_refs:
            raise ValueError(
                "LLM returned unknown candidate claim IDs: "
                + ", ".join(invalid_candidate_refs)
            )

        return EvaluatorResult(
            evaluator="completeness",
            score=score,
            confidence=data["confidence"],
            claims_evaluated=len(evaluated_claims),
            errors=errors,
            reasoning=data["reasoning"],
            evidence_refs=[],
        )

    @staticmethod
    def _calculate_score(
        evaluated_claims: list[dict],
    ) -> float:
        """Calculate deterministic completeness score."""

        verdict_scores = {
            "covered": 1.0,
            "partially_covered": 0.5,
            "missing": 0.0,
        }

        total = sum(
            verdict_scores[item["verdict"]]
            for item in evaluated_claims
        )

        return total / len(evaluated_claims)