import json

from src.v2.llm.client import LLMClient
from src.v2.models.schemas import Claim, Evidence, ReferenceAnswer


CLAIM_EXTRACTION_FORMAT = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "text": {"type": "string"},
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "importance": {"type": "number"},
                },
                "required": [
                    "claim_id",
                    "text",
                    "evidence_refs",
                    "importance",
                ],
            },
        }
    },
    "required": ["claims"],
}


class ClaimExtractor:
    """Extract evaluable claims from a candidate response."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def extract(
        self,
        question: str,
        response: str,
        reference_answer: ReferenceAnswer,
        evidence: list[Evidence],
    ) -> list[Claim]:
        """Extract claims from the candidate response."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not response.strip():
            raise ValueError("Response cannot be empty.")

        if not evidence:
            raise ValueError(
                "At least one evidence item is required."
            )

        evidence_text = "\n\n".join(
            (
                f"[{item.evidence_id}]\n"
                f"{item.content}"
            )
            for item in evidence
        )

        reference_claims = "\n".join(
            f"[{claim.claim_id}] {claim.text}"
            for claim in reference_answer.claims
        )

        prompt = f"""
You are a claim extraction component in an LLM evaluation
system.

Extract the distinct factual, semantic, and logical claims
made by the candidate response.

Do not evaluate whether the claims are correct.
Do not rewrite the candidate's meaning.
Do not invent claims.

QUESTION:
{question}

CANDIDATE RESPONSE:
{response}

REFERENCE CLAIMS:
{reference_claims}

AVAILABLE EVIDENCE:
{evidence_text}

RULES:
1. Extract meaningful claims from the candidate response.
2. Preserve the original meaning.
3. Give every claim a unique claim_id.
4. Link a claim to evidence IDs when the evidence is
   relevant to that claim.
5. Use an importance value between 0 and 1.
6. Do not assign correctness scores.
7. Return an empty list if there are no meaningful claims.
"""

        raw_response = self.llm_client.generate(
            prompt=prompt,
            response_format=CLAIM_EXTRACTION_FORMAT,
        )

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON."
            ) from exc

        return [
            Claim(**claim)
            for claim in data.get("claims", [])
        ]