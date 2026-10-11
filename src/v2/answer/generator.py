import json

from src.v2.llm.client import LLMClient


CANDIDATE_ANSWER_FORMAT = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
        },
    },
    "required": ["answer"],
}


class CandidateAnswerGenerator:
    """Generate an initial answer independently of retrieved evidence."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate(self, question: str) -> str:
        """Generate a candidate answer from the user's question."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        prompt = f"""
You are an AI assistant answering a user's question.

QUESTION:
{question}

INSTRUCTIONS:
1. Answer the question directly and clearly.
2. Use your general knowledge.
3. Do not claim that you searched external sources.
4. Do not invent citations or sources.
5. If you are uncertain, acknowledge the uncertainty.
6. Return the answer in the required JSON format.
"""

        raw_response = self.llm_client.generate(
            prompt=prompt,
            response_format=CANDIDATE_ANSWER_FORMAT,
        )

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON for the candidate answer."
            ) from exc

        answer = data.get("answer")

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                "LLM returned an empty or invalid candidate answer."
            )

        return answer.strip()