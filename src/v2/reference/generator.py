import json

from src.v2.llm.client import LLMClient
from src.v2.models.schemas import Claim, Evidence, ReferenceAnswer


REFERENCE_ANSWER_FORMAT = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "string",
                    },
                    "text": {
                        "type": "string",
                    },
                    "evidence_refs": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                    "importance": {
                        "type": "number",
                    },
                },
                "required": [
                    "claim_id",
                    "text",
                    "evidence_refs",
                    "importance",
                ],
            },
        },
        "evidence_refs": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "confidence": {
            "type": "number",
        },
    },
    "required": [
        "answer",
        "claims",
        "evidence_refs",
        "confidence",
    ],
}


class ReferenceAnswerGenerator:
    """Generate an evidence-grounded reference answer."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate(
        self,
        question: str,
        evidence: list[Evidence],
    ) -> ReferenceAnswer:
        """Generate a reference answer from retrieved evidence."""

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

        prompt = f"""
You are a reference-answer generation component in an
LLM evaluation system.

Your task is to generate the authoritative reference answer
for the question using ONLY the supplied evidence.

QUESTION:
{question}

RETRIEVED EVIDENCE:
{evidence_text}

RULES:
1. Use only information supported by the supplied evidence.
2. Do not invent facts.
3. Do not use outside knowledge.
4. Make the answer concise but complete.
5. Break the reference answer into factual claims.
6. Every claim must reference one or more evidence IDs.
7. Assign each claim an importance between 0 and 1.
8. Set confidence between 0 and 1.
9. The evidence_refs field must contain the evidence IDs
   actually used by the reference answer.
"""

        raw_response = self.llm_client.generate(
            prompt=prompt,
            response_format=REFERENCE_ANSWER_FORMAT,
        )

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON."
            ) from exc

        claims = [
            Claim(**claim)
            for claim in data.get("claims", [])
        ]

        return ReferenceAnswer(
            answer=data["answer"],
            claims=claims,
            evidence_refs=data["evidence_refs"],
            confidence=data["confidence"],
        )