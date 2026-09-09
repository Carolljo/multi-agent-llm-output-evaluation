import json

from src.v2.llm.client import LLMClient
from src.v2.models.schemas import (
    Evidence,
    ReferenceAnswer,
    ReferenceValidation,
)


REFERENCE_VALIDATION_FORMAT = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["valid", "invalid", "needs_review"],
        },
        "score": {
            "type": "number",
        },
        "confidence": {
            "type": "number",
        },
        "supported_claims": {
            "type": "array",
            "items": {"type": "string"},
        },
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
        },
        "missing_information": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {
            "type": "string",
        },
    },
    "required": [
        "status",
        "score",
        "confidence",
        "supported_claims",
        "unsupported_claims",
        "missing_information",
        "reasoning",
    ],
}


class ReferenceAnswerValidator:
    """Independently validate an evidence-grounded reference answer."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def validate(
        self,
        question: str,
        reference_answer: ReferenceAnswer,
        evidence: list[Evidence],
    ) -> ReferenceValidation:
        """Validate the reference answer against retrieved evidence."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not evidence:
            raise ValueError(
                "At least one evidence item is required."
            )

        evidence_text = "\n\n".join(
            (
                f"[{item.evidence_id}] "
                f"Source: {item.source or 'unknown'}\n"
                f"{item.content}"
            )
            for item in evidence
        )

        claims_text = "\n".join(
            (
                f"[{claim.claim_id}] {claim.text} "
                f"(evidence: {', '.join(claim.evidence_refs)})"
            )
            for claim in reference_answer.claims
        )

        prompt = f"""
You are an independent validation component in an
LLM output evaluation system.

Your task is to validate a generated reference answer
against the supplied evidence.

Do NOT assume the reference answer is correct.
Do NOT use outside knowledge.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer.answer}

REFERENCE CLAIMS:
{claims_text}

RETRIEVED EVIDENCE:
{evidence_text}

RULES:
1. Check every reference claim against the evidence.
2. Identify claims that are directly supported.
3. Identify claims that are unsupported or contradicted.
4. Identify important information from the evidence that
   the reference answer failed to include.
5. Assign a validation score between 0 and 1.
6. Assign confidence between 0 and 1.
7. Use "valid" when the reference is adequately supported.
8. Use "invalid" when it contains significant unsupported
   or contradicted information.
9. Use "needs_review" when the evidence is insufficient
   to confidently determine validity.
10. Explain your reasoning.
"""

        raw_response = self.llm_client.generate(
            prompt=prompt,
            response_format=REFERENCE_VALIDATION_FORMAT,
        )

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON."
            ) from exc

        return ReferenceValidation(**data)