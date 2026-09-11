import pytest

from src.v2.agents.accuracy import AccuracyEvaluator
from src.v2.models.schemas import (
    Claim,
    Evidence,
    ReferenceAnswer,
)


class FakeLLMClient:
    """Fake LLM client for deterministic unit tests."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None
        self.last_response_format = None

    def generate(self, prompt: str, response_format: dict) -> str:
        self.last_prompt = prompt
        self.last_response_format = response_format
        return self.response


def build_claims() -> list[Claim]:
    return [
        Claim(
            claim_id="c1",
            text="The Eiffel Tower was completed in 1889.",
            evidence_refs=["e1"],
            importance=1.0,
        ),
        Claim(
            claim_id="c2",
            text="The Eiffel Tower is located in London.",
            evidence_refs=["e1"],
            importance=1.0,
        ),
    ]


def build_reference_answer() -> ReferenceAnswer:
    return ReferenceAnswer(
        answer="The Eiffel Tower was completed in 1889 and is located in Paris.",
        claims=[
            Claim(
                claim_id="r1",
                text="The Eiffel Tower was completed in 1889.",
                evidence_refs=["e1"],
                importance=1.0,
            ),
            Claim(
                claim_id="r2",
                text="The Eiffel Tower is located in Paris.",
                evidence_refs=["e1"],
                importance=1.0,
            ),
        ],
        evidence_refs=["e1"],
        confidence=0.95,
    )


def build_evidence() -> list[Evidence]:
    return [
        Evidence(
            evidence_id="e1",
            document_id="doc1",
            chunk_id="chunk1",
            content=(
                "The Eiffel Tower was completed in 1889 "
                "and is located in Paris."
            ),
            source="test-source",
            relevance_score=0.95,
        )
    ]


def test_accuracy_evaluator_returns_result():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "supported",
                    "reasoning": "The evidence directly supports the claim.",
                    "evidence_refs": ["e1"]
                },
                {
                    "claim_id": "c2",
                    "verdict": "contradicted",
                    "reasoning": "The evidence states that the Eiffel Tower is in Paris.",
                    "evidence_refs": ["e1"]
                }
            ],
            "confidence": 0.95,
            "reasoning": "One claim is supported and one is contradicted."
        }
        """
    )

    evaluator = AccuracyEvaluator(llm)

    result = evaluator.evaluate(
        question="Where is the Eiffel Tower and when was it completed?",
        claims=build_claims(),
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert result.evaluator == "accuracy"
    assert result.score == 0.5
    assert result.confidence == 0.95
    assert result.claims_evaluated == 2
    assert len(result.errors) == 1
    assert result.evidence_refs == ["e1"]


def test_accuracy_score_is_deterministic():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "supported",
                    "reasoning": "Supported.",
                    "evidence_refs": ["e1"]
                },
                {
                    "claim_id": "c2",
                    "verdict": "partially_supported",
                    "reasoning": "Partially supported.",
                    "evidence_refs": ["e1"]
                },
                {
                    "claim_id": "c3",
                    "verdict": "contradicted",
                    "reasoning": "Contradicted.",
                    "evidence_refs": ["e1"]
                },
                {
                    "claim_id": "c4",
                    "verdict": "not_verifiable",
                    "reasoning": "Insufficient evidence.",
                    "evidence_refs": []
                }
            ],
            "confidence": 0.8,
            "reasoning": "Mixed accuracy."
        }
        """
    )

    claims = [
        Claim(
            claim_id=f"c{i}",
            text=f"Claim {i}",
            evidence_refs=["e1"],
            importance=1.0,
        )
        for i in range(1, 5)
    ]

    evaluator = AccuracyEvaluator(llm)

    result = evaluator.evaluate(
        question="Test question",
        claims=claims,
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert result.score == 0.375


def test_accuracy_rejects_empty_question():
    llm = FakeLLMClient("{}")
    evaluator = AccuracyEvaluator(llm)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        evaluator.evaluate(
            question="",
            claims=build_claims(),
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_accuracy_rejects_empty_claims():
    llm = FakeLLMClient("{}")
    evaluator = AccuracyEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="At least one candidate claim is required",
    ):
        evaluator.evaluate(
            question="Test question",
            claims=[],
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_accuracy_rejects_empty_evidence():
    llm = FakeLLMClient("{}")
    evaluator = AccuracyEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="At least one evidence item is required",
    ):
        evaluator.evaluate(
            question="Test question",
            claims=build_claims(),
            reference_answer=build_reference_answer(),
            evidence=[],
        )


def test_accuracy_rejects_invalid_json():
    llm = FakeLLMClient("not valid json")
    evaluator = AccuracyEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="LLM returned invalid JSON",
    ):
        evaluator.evaluate(
            question="Test question",
            claims=build_claims(),
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_accuracy_rejects_incomplete_claim_evaluation():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "supported",
                    "reasoning": "Supported.",
                    "evidence_refs": ["e1"]
                }
            ],
            "confidence": 0.9,
            "reasoning": "Only one claim evaluated."
        }
        """
    )

    evaluator = AccuracyEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="LLM did not evaluate every candidate claim",
    ):
        evaluator.evaluate(
            question="Test question",
            claims=build_claims(),
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_accuracy_prompt_contains_required_context():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "supported",
                    "reasoning": "Supported.",
                    "evidence_refs": ["e1"]
                },
                {
                    "claim_id": "c2",
                    "verdict": "contradicted",
                    "reasoning": "Contradicted.",
                    "evidence_refs": ["e1"]
                }
            ],
            "confidence": 0.9,
            "reasoning": "Evaluation complete."
        }
        """
    )

    evaluator = AccuracyEvaluator(llm)

    evaluator.evaluate(
        question="Where is the Eiffel Tower?",
        claims=build_claims(),
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert "Where is the Eiffel Tower?" in llm.last_prompt
    assert "The Eiffel Tower was completed in 1889." in llm.last_prompt
    assert "The Eiffel Tower is located in London." in llm.last_prompt
    assert "The Eiffel Tower is located in Paris." in llm.last_prompt
    assert "[e1]" in llm.last_prompt