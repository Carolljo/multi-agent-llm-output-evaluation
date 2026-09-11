import json

from src.v2.llm.client import LLMClient
from src.v2.models.schemas import (
    Claim,
    Evidence,
    EvaluatorResult,
    ReferenceAnswer,
)


ACCURACY_EVALUATION_FORMAT = {
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
                            "supported",
                            "partially_supported",
                            "contradicted",
                            "not_verifiable",
                        ],
                    },
                    "reasoning": {
                        "type": "string",
                    },
                    "evidence_refs": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                },
                "required": [
                    "claim_id",
                    "verdict",
                    "reasoning",
                    "evidence_refs",
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


class AccuracyEvaluator:
    """Evaluate factual accuracy of candidate claims."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def evaluate(
        self,
        question: str,
        claims: list[Claim],
        reference_answer: ReferenceAnswer,
        evidence: list[Evidence],
    ) -> EvaluatorResult:
        """Evaluate candidate claims against the reference and evidence."""

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
You are an accuracy evaluation component in an
LLM output evaluation system.

Your task is to evaluate the factual accuracy of the
candidate response at the claim level.

Use ONLY the supplied reference answer and retrieved
evidence.

Do NOT use outside knowledge.
Do NOT assume the candidate response is correct.
Do NOT evaluate completeness, style, or relevance.
Focus only on whether each candidate claim is factually
supported.

QUESTION:
{question}

CANDIDATE CLAIMS:
{candidate_claims_text}

REFERENCE ANSWER:
{reference_answer.answer}

REFERENCE CLAIMS:
{reference_claims_text}

RETRIEVED EVIDENCE:
{evidence_text}

For every candidate claim, assign exactly one verdict:

- supported:
  The claim is directly supported by the reference answer
  and/or retrieved evidence.

- partially_supported:
  The claim contains both supported and unsupported parts,
  or is only partially supported by the available evidence.

- contradicted:
  The claim conflicts with the reference answer or evidence.

- not_verifiable:
  The available reference and evidence are insufficient
  to determine whether the claim is correct.

RULES:
1. Evaluate every candidate claim.
2. Preserve the original claim_id.
3. Do not invent evidence references.
4. Evidence references must contain only supplied evidence IDs.
5. Explain the reason for each verdict.
6. Assign confidence between 0 and 1.
7. Provide an overall reasoning summary.
"""

        raw_response = self.llm_client.generate(
            prompt=prompt,
            response_format=ACCURACY_EVALUATION_FORMAT,
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

        score = self._calculate_score(evaluated_claims)

        errors = [
            (
                f"{item['claim_id']}: "
                f"{item['verdict']} - "
                f"{item['reasoning']}"
            )
            for item in evaluated_claims
            if item["verdict"] != "supported"
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
            evaluator="accuracy",
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
        """Calculate deterministic accuracy from claim verdicts."""

        verdict_scores = {
            "supported": 1.0,
            "partially_supported": 0.5,
            "contradicted": 0.0,
            "not_verifiable": 0.0,
        }

        total = sum(
            verdict_scores[item["verdict"]]
            for item in evaluated_claims
        )

        return total / len(evaluated_claims)