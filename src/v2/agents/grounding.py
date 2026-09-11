import json

from src.v2.llm.client import LLMClient
from src.v2.models.schemas import (
    Claim,
    Evidence,
    EvaluatorResult,
    ReferenceAnswer,
)


EVIDENCE_GROUNDING_EVALUATION_FORMAT = {
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
                            "grounded",
                            "partially_grounded",
                            "ungrounded",
                        ],
                    },
                    "evidence_refs": {
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
                    "evidence_refs",
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


class EvidenceGroundingEvaluator:
    """Evaluate whether candidate claims are supported by evidence."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def evaluate(
        self,
        question: str,
        claims: list[Claim],
        reference_answer: ReferenceAnswer,
        evidence: list[Evidence],
    ) -> EvaluatorResult:
        """Evaluate grounding of candidate claims in retrieved evidence."""

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
            (
                f"[{claim.claim_id}] {claim.text} "
                f"(claimed evidence: "
                f"{', '.join(claim.evidence_refs) or 'none'})"
            )
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
You are an evidence-grounding evaluation component in an
LLM output evaluation system.

Your task is to determine whether each candidate claim is
supported by the supplied retrieved evidence.

Use ONLY the supplied evidence.

Do NOT use outside knowledge.
Do NOT assume a claim is grounded merely because it sounds
plausible.
Do NOT evaluate writing quality.
Do NOT evaluate completeness.
Do NOT determine general factual truth independently of
the supplied evidence.

QUESTION:
{question}

CANDIDATE CLAIMS:
{candidate_claims_text}

REFERENCE CLAIMS:
{reference_claims_text}

RETRIEVED EVIDENCE:
{evidence_text}

For every candidate claim, assign exactly one verdict:

- grounded:
  The supplied evidence directly supports the claim.

- partially_grounded:
  The evidence supports some but not all of the information
  contained in the claim.

- ungrounded:
  The supplied evidence does not support the claim.

RULES:
1. Evaluate every candidate claim.
2. Preserve every candidate claim_id exactly.
3. Identify the evidence IDs that actually support the claim.
4. Do not invent evidence IDs.
5. Do not treat the reference answer itself as evidence.
6. A claim is ungrounded if the supplied evidence does not
   provide sufficient support.
7. Assign confidence between 0 and 1.
8. Provide an overall reasoning summary.
"""

        raw_response = self.llm_client.generate(
            prompt=prompt,
            response_format=EVIDENCE_GROUNDING_EVALUATION_FORMAT,
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

        evidence_ids = {
            item.evidence_id
            for item in evidence
        }

        unknown_evidence_refs = sorted(
            {
                evidence_ref
                for item in evaluated_claims
                for evidence_ref in item.get(
                    "evidence_refs",
                    [],
                )
                if evidence_ref not in evidence_ids
            }
        )

        if unknown_evidence_refs:
            raise ValueError(
                "LLM returned unknown evidence IDs: "
                + ", ".join(unknown_evidence_refs)
            )

        score = self._calculate_score(evaluated_claims)

        errors = [
            (
                f"{item['claim_id']}: "
                f"{item['verdict']} - "
                f"{item['reasoning']}"
            )
            for item in evaluated_claims
            if item["verdict"] != "grounded"
        ]

        evidence_refs = sorted(
            {
                evidence_ref
                for item in evaluated_claims
                for evidence_ref in item.get(
                    "evidence_refs",
                    [],
                )
            }
        )

        return EvaluatorResult(
            evaluator="evidence_grounding",
            score=score,
            confidence=data["confidence"],
            claims_evaluated=len(evaluated_claims),
            errors=errors,
            reasoning=data["reasoning"],
            evidence_refs=evidence_refs,
        )

    @staticmethod
    def _calculate_score(
        evaluated_claims: list[dict],
    ) -> float:
        """Calculate deterministic grounding score."""

        verdict_scores = {
            "grounded": 1.0,
            "partially_grounded": 0.5,
            "ungrounded": 0.0,
        }

        total = sum(
            verdict_scores[item["verdict"]]
            for item in evaluated_claims
        )

        return total / len(evaluated_claims)