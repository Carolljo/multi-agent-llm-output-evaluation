import json

from src.v2.llm.client import LLMClient
from src.v2.models.schemas import (
    Claim,
    Evidence,
    EvaluatorResult,
    ReferenceAnswer,
)


LOGICAL_CONSISTENCY_EVALUATION_FORMAT = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "string",
                    },
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "consistent",
                            "partially_inconsistent",
                            "contradicted",
                        ],
                    },
                    "conflicting_claim_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                    "reasoning": {
                        "type": "string",
                    },
                },
                "required": [
                    "claim_id",
                    "verdict",
                    "conflicting_claim_ids",
                    "reasoning",
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


class LogicalConsistencyEvaluator:
    """Evaluate internal logical consistency of candidate claims."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def evaluate(
        self,
        question: str,
        claims: list[Claim],
        reference_answer: ReferenceAnswer,
        evidence: list[Evidence],
    ) -> EvaluatorResult:
        """Evaluate whether candidate claims are mutually consistent."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not claims:
            raise ValueError(
                "At least one candidate claim is required."
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
            f"[{claim.claim_id}] {claim.text}"
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
You are a logical consistency evaluation component in an
LLM output evaluation system.

Your task is to determine whether the claims made by the
candidate response are logically and semantically consistent
with one another.

The primary focus is INTERNAL consistency.

Use the supplied reference answer and evidence only as
context for interpreting the claims.

Do NOT use outside knowledge.
Do NOT treat disagreement with the reference answer as an
internal contradiction by itself.
Do NOT evaluate completeness.
Do NOT evaluate writing quality.
Do NOT evaluate whether every claim is factually correct.

QUESTION:
{question}

CANDIDATE CLAIMS:
{candidate_claims_text}

REFERENCE CLAIMS:
{reference_claims_text}

RETRIEVED EVIDENCE:
{evidence_text}

For every candidate claim, assign exactly one verdict:

- consistent:
  The claim does not conflict with the other candidate
  claims.

- partially_inconsistent:
  The claim has a possible or limited conflict with another
  candidate claim, but the contradiction is not complete.

- contradicted:
  The claim directly conflicts with one or more other
  candidate claims.

RULES:
1. Evaluate every candidate claim.
2. Preserve every candidate claim_id exactly.
3. Identify conflicting candidate claim IDs when a conflict
   exists.
4. Do not invent candidate claim IDs.
5. Focus on contradictions between candidate claims.
6. A difference from the reference answer is not automatically
   an internal contradiction.
7. Do not use outside knowledge.
8. Assign confidence between 0 and 1.
9. Provide an overall reasoning summary.
"""

        raw_response = self.llm_client.generate(
            prompt=prompt,
            response_format=LOGICAL_CONSISTENCY_EVALUATION_FORMAT,
        )

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON."
            ) from exc

        evaluated_claims = data.get("claims", [])

        if len(evaluated_claims) != len(claims):
            raise ValueError(
                "LLM did not evaluate every candidate claim."
            )

        candidate_claim_ids = {
            claim.claim_id
            for claim in claims
        }

        evaluated_claim_ids = {
            item["claim_id"]
            for item in evaluated_claims
        }

        missing_claim_ids = (
            candidate_claim_ids - evaluated_claim_ids
        )

        if missing_claim_ids:
            raise ValueError(
                "LLM did not return all candidate claim IDs: "
                + ", ".join(sorted(missing_claim_ids))
            )

        unknown_conflicting_ids = sorted(
            {
                conflicting_id
                for item in evaluated_claims
                for conflicting_id in item.get(
                    "conflicting_claim_ids",
                    [],
                )
                if conflicting_id not in candidate_claim_ids
            }
        )

        if unknown_conflicting_ids:
            raise ValueError(
                "LLM returned unknown conflicting claim IDs: "
                + ", ".join(unknown_conflicting_ids)
            )

        score = self._calculate_score(evaluated_claims)

        errors = [
            (
                f"{item['claim_id']}: "
                f"{item['verdict']} - "
                f"{item['reasoning']}"
            )
            for item in evaluated_claims
            if item["verdict"] != "consistent"
        ]

        return EvaluatorResult(
            evaluator="logical_consistency",
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
        """Calculate deterministic logical consistency score."""

        verdict_scores = {
            "consistent": 1.0,
            "partially_inconsistent": 0.5,
            "contradicted": 0.0,
        }

        total = sum(
            verdict_scores[item["verdict"]]
            for item in evaluated_claims
        )

        return total / len(evaluated_claims)