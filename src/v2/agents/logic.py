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
            "minimum": 0,
            "maximum": 1,
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

For every candidate claim, assign exactly one verdict.

IMPORTANT:
The "verdict" field MUST contain exactly one of these three
strings and nothing else:

"consistent"
"partially_inconsistent"
"contradicted"

Do NOT use "partial_inconsistent", "partially inconsistent",
"partial", "inconsistent", or any other variation.

- consistent:
  The claim does not conflict with the other candidate claims.

- partially_inconsistent:
  The claim has a possible or limited conflict with another
  candidate claim, but the contradiction is not complete.

- contradicted:
  The claim directly conflicts with one or more other
  candidate claims.

RULES:
1. Evaluate every candidate claim.
2. Preserve every candidate claim_id exactly.
3. Identify conflicting candidate claim IDs when a conflict exists.
4. Do not invent candidate claim IDs.
5. Focus on contradictions between candidate claims.
6. A difference from the reference answer is not automatically
   an internal contradiction.
7. Do not use outside knowledge.
8. Assign confidence as a JSON NUMBER between 0 and 1.
   Valid examples: 0.9, 0.95, 1.0, 0.75.
   Never write numbers as words.
   Never use formats such as "0. nine" or "zero point nine".
9. Provide an overall reasoning summary.

OUTPUT REQUIREMENTS:
- Return the exact JSON structure required by the response schema.
- The confidence field must contain a numeric value, not a string.
- Evaluate every candidate claim exactly once.
- Use only the supplied candidate claim IDs.
- Every conflicting_claim_id must identify an actual candidate claim.
- A claim must not identify itself as a conflicting claim.
- Do not include markdown fences or explanatory text outside the JSON.
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

        if not isinstance(data, dict):
            raise ValueError(
                "LLM response must be a JSON object."
            )

        # Validate confidence before constructing the result.
        confidence = data.get("confidence")

        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise ValueError(
                "LLM returned invalid confidence. "
                "Expected a number between 0 and 1."
            )

        # Validate the evaluated claims.
        evaluated_claims = data.get("claims")

        if not isinstance(evaluated_claims, list):
            raise ValueError(
                "LLM response must contain a claims list."
            )

        if len(evaluated_claims) != len(claims):
            raise ValueError(
                "LLM did not evaluate every candidate claim."
            )

        candidate_claim_ids = [
            claim.claim_id
            for claim in claims
        ]
        candidate_claim_id_set = set(candidate_claim_ids)

        if len(candidate_claim_id_set) != len(candidate_claim_ids):
            raise ValueError(
                "Candidate claims contain duplicate claim IDs."
            )

        # Validate claim objects before accessing their fields.
        for item in evaluated_claims:
            if not isinstance(item, dict):
                raise ValueError(
                    "Each evaluated claim must be a JSON object."
                )

            required_fields = {
                "claim_id",
                "verdict",
                "conflicting_claim_ids",
                "reasoning",
            }

            missing_fields = required_fields - item.keys()

            if missing_fields:
                raise ValueError(
                    "Evaluated claim is missing required fields: "
                    + ", ".join(sorted(missing_fields))
                )

            if not isinstance(item["claim_id"], str):
                raise ValueError(
                    "Every evaluated claim must have a string claim_id."
                )

            if not isinstance(item["reasoning"], str):
                raise ValueError(
                    "Every evaluated claim must have string reasoning."
                )

            if not isinstance(item["conflicting_claim_ids"], list):
                raise ValueError(
                    "conflicting_claim_ids must be a list."
                )

            if not all(
                isinstance(claim_id, str)
                for claim_id in item["conflicting_claim_ids"]
            ):
                raise ValueError(
                    "Every conflicting claim ID must be a string."
                )

        evaluated_claim_ids = [
            item["claim_id"]
            for item in evaluated_claims
        ]

        # Every candidate claim ID must appear exactly once.
        if len(set(evaluated_claim_ids)) != len(evaluated_claim_ids):
            raise ValueError(
                "LLM returned duplicate candidate claim IDs."
            )

        if set(evaluated_claim_ids) != candidate_claim_id_set:
            missing_claim_ids = (
                candidate_claim_id_set - set(evaluated_claim_ids)
            )
            unknown_claim_ids = (
                set(evaluated_claim_ids) - candidate_claim_id_set
            )

            raise ValueError(
                "LLM did not return all candidate claim IDs. "
                f"Missing: {sorted(missing_claim_ids)}; "
                f"Unknown: {sorted(unknown_claim_ids)}"
            )

        # Validate verdicts even when the LLM response is malformed.
        valid_verdicts = {
            "consistent",
            "partially_inconsistent",
            "contradicted",
        }

        invalid_verdicts = sorted(
            {
                item["verdict"]
                for item in evaluated_claims
                if item["verdict"] not in valid_verdicts
            }
        )

        if invalid_verdicts:
            raise ValueError(
                "LLM returned invalid verdicts: "
                + ", ".join(invalid_verdicts)
            )

        # All conflict references must point to existing candidate claims.
        unknown_conflicting_ids = sorted(
            {
                conflicting_id
                for item in evaluated_claims
                for conflicting_id in item["conflicting_claim_ids"]
                if conflicting_id not in candidate_claim_id_set
            }
        )

        if unknown_conflicting_ids:
            raise ValueError(
                "LLM returned unknown conflicting claim IDs: "
                + ", ".join(unknown_conflicting_ids)
            )

        # A claim cannot conflict with itself.
        invalid_self_conflicts = sorted(
            {
                item["claim_id"]
                for item in evaluated_claims
                if item["claim_id"] in item["conflicting_claim_ids"]
            }
        )

        if invalid_self_conflicts:
            raise ValueError(
                "Claims cannot conflict with themselves: "
                + ", ".join(invalid_self_conflicts)
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
            confidence=float(confidence),
            claims_evaluated=len(evaluated_claims),
            errors=errors,
            reasoning=data.get("reasoning", ""),
            evidence_refs=[],
        )

    @staticmethod
    def _calculate_score(
        evaluated_claims: list[dict],
    ) -> float:
        """Calculate deterministic logical consistency score."""

        if not evaluated_claims:
            raise ValueError(
                "Cannot calculate a score without evaluated claims."
            )

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