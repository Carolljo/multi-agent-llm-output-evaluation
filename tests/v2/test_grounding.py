import pytest

from src.v2.agents.grounding import EvidenceGroundingEvaluator
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
            text="Paris is the capital of France.",
            evidence_refs=["e1"],
            importance=1.0,
        ),
        Claim(
            claim_id="c2",
            text="Paris is located on the Seine River.",
            evidence_refs=["e1"],
            importance=0.8,
        ),
        Claim(
            claim_id="c3",
            text="Paris has the largest population in Europe.",
            evidence_refs=[],
            importance=0.5,
        ),
    ]


def build_reference_answer() -> ReferenceAnswer:
    return ReferenceAnswer(
        answer=(
            "Paris is the capital of France and is located "
            "on the Seine River."
        ),
        claims=[
            Claim(
                claim_id="r1",
                text="Paris is the capital of France.",
                evidence_refs=["e1"],
                importance=1.0,
            ),
            Claim(
                claim_id="r2",
                text="Paris is located on the Seine River.",
                evidence_refs=["e1"],
                importance=0.8,
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
                "Paris is the capital of France and is located "
                "on the Seine River."
            ),
            source="test-source",
            relevance_score=0.95,
        ),
        Evidence(
            evidence_id="e2",
            document_id="doc1",
            chunk_id="chunk2",
            content=(
                "Paris is one of the most visited cities in Europe."
            ),
            source="test-source",
            relevance_score=0.80,
        ),
    ]


def test_grounding_evaluator_returns_result():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "grounded",
                    "evidence_refs": ["e1"],
                    "reasoning": "The evidence directly supports the claim."
                },
                {
                    "claim_id": "c2",
                    "verdict": "grounded",
                    "evidence_refs": ["e1"],
                    "reasoning": "The evidence directly supports the claim."
                },
                {
                    "claim_id": "c3",
                    "verdict": "ungrounded",
                    "evidence_refs": [],
                    "reasoning": "The supplied evidence does not support the claim."
                }
            ],
            "confidence": 0.94,
            "reasoning": "Two claims are grounded and one is unsupported."
        }
        """
    )

    evaluator = EvidenceGroundingEvaluator(llm)

    result = evaluator.evaluate(
        question="What can you tell me about Paris?",
        claims=build_claims(),
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert result.evaluator == "evidence_grounding"
    assert result.score == pytest.approx(2 / 3)
    assert result.confidence == 0.94
    assert result.claims_evaluated == 3
    assert len(result.errors) == 1
    assert result.evidence_refs == ["e1"]


def test_grounding_score_is_deterministic():
    claims = [
        Claim(
            claim_id="c1",
            text="Claim one.",
            evidence_refs=["e1"],
            importance=1.0,
        ),
        Claim(
            claim_id="c2",
            text="Claim two.",
            evidence_refs=["e1"],
            importance=1.0,
        ),
        Claim(
            claim_id="c3",
            text="Claim three.",
            evidence_refs=["e1"],
            importance=1.0,
        ),
        Claim(
            claim_id="c4",
            text="Claim four.",
            evidence_refs=[],
            importance=1.0,
        ),
    ]

    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "grounded",
                    "evidence_refs": ["e1"],
                    "reasoning": "Grounded."
                },
                {
                    "claim_id": "c2",
                    "verdict": "partially_grounded",
                    "evidence_refs": ["e1"],
                    "reasoning": "Partially grounded."
                },
                {
                    "claim_id": "c3",
                    "verdict": "ungrounded",
                    "evidence_refs": [],
                    "reasoning": "Ungrounded."
                },
                {
                    "claim_id": "c4",
                    "verdict": "grounded",
                    "evidence_refs": ["e2"],
                    "reasoning": "Grounded."
                }
            ],
            "confidence": 0.8,
            "reasoning": "Mixed grounding."
        }
        """
    )

    evaluator = EvidenceGroundingEvaluator(llm)

    result = evaluator.evaluate(
        question="Test question.",
        claims=claims,
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert result.score == pytest.approx(0.625)


def test_grounding_rejects_empty_question():
    llm = FakeLLMClient("{}")
    evaluator = EvidenceGroundingEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        evaluator.evaluate(
            question="",
            claims=build_claims(),
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_grounding_rejects_empty_claims():
    llm = FakeLLMClient("{}")
    evaluator = EvidenceGroundingEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="At least one candidate claim is required",
    ):
        evaluator.evaluate(
            question="Test question.",
            claims=[],
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_grounding_rejects_empty_evidence():
    llm = FakeLLMClient("{}")
    evaluator = EvidenceGroundingEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="At least one evidence item is required",
    ):
        evaluator.evaluate(
            question="Test question.",
            claims=build_claims(),
            reference_answer=build_reference_answer(),
            evidence=[],
        )


def test_grounding_rejects_invalid_json():
    llm = FakeLLMClient("not valid json")
    evaluator = EvidenceGroundingEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="LLM returned invalid JSON",
    ):
        evaluator.evaluate(
            question="Test question.",
            claims=build_claims(),
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_grounding_rejects_incomplete_claim_evaluation():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "grounded",
                    "evidence_refs": ["e1"],
                    "reasoning": "Grounded."
                }
            ],
            "confidence": 0.9,
            "reasoning": "Only one claim evaluated."
        }
        """
    )

    evaluator = EvidenceGroundingEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="LLM did not evaluate every candidate claim",
    ):
        evaluator.evaluate(
            question="Test question.",
            claims=build_claims(),
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_grounding_rejects_missing_candidate_claim_id():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "grounded",
                    "evidence_refs": ["e1"],
                    "reasoning": "Grounded."
                },
                {
                    "claim_id": "c2",
                    "verdict": "grounded",
                    "evidence_refs": ["e1"],
                    "reasoning": "Grounded."
                },
                {
                    "claim_id": "wrong_id",
                    "verdict": "grounded",
                    "evidence_refs": ["e1"],
                    "reasoning": "Grounded."
                }
            ],
            "confidence": 0.9,
            "reasoning": "Evaluation complete."
        }
        """
    )

    evaluator = EvidenceGroundingEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="LLM did not return all candidate claim IDs",
    ):
        evaluator.evaluate(
            question="Test question.",
            claims=build_claims(),
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_grounding_rejects_unknown_evidence_ids():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "grounded",
                    "evidence_refs": ["unknown"],
                    "reasoning": "Grounded."
                },
                {
                    "claim_id": "c2",
                    "verdict": "grounded",
                    "evidence_refs": ["e1"],
                    "reasoning": "Grounded."
                },
                {
                    "claim_id": "c3",
                    "verdict": "ungrounded",
                    "evidence_refs": [],
                    "reasoning": "Ungrounded."
                }
            ],
            "confidence": 0.9,
            "reasoning": "Evaluation complete."
        }
        """
    )

    evaluator = EvidenceGroundingEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="LLM returned unknown evidence IDs",
    ):
        evaluator.evaluate(
            question="Test question.",
            claims=build_claims(),
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_grounding_prompt_contains_required_context():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "grounded",
                    "evidence_refs": ["e1"],
                    "reasoning": "Grounded."
                },
                {
                    "claim_id": "c2",
                    "verdict": "grounded",
                    "evidence_refs": ["e1"],
                    "reasoning": "Grounded."
                },
                {
                    "claim_id": "c3",
                    "verdict": "ungrounded",
                    "evidence_refs": [],
                    "reasoning": "Ungrounded."
                }
            ],
            "confidence": 0.9,
            "reasoning": "Evaluation complete."
        }
        """
    )

    evaluator = EvidenceGroundingEvaluator(llm)

    evaluator.evaluate(
        question="What can you tell me about Paris?",
        claims=build_claims(),
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert "What can you tell me about Paris?" in llm.last_prompt
    assert "Paris is the capital of France." in llm.last_prompt
    assert "Paris is located on the Seine River." in llm.last_prompt
    assert "Paris has the largest population in Europe." in llm.last_prompt
    assert "[c1]" in llm.last_prompt
    assert "[r1]" in llm.last_prompt
    assert "[e1]" in llm.last_prompt


def test_grounding_requires_actual_evidence_support():
    claims = [
        Claim(
            claim_id="c1",
            text="Paris is located in France.",
            evidence_refs=["e1"],
            importance=1.0,
        ),
    ]

    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "ungrounded",
                    "evidence_refs": [],
                    "reasoning": "The supplied evidence does not explicitly support this claim."
                }
            ],
            "confidence": 0.9,
            "reasoning": "The claim cannot be grounded in the supplied evidence."
        }
        """
    )

    evaluator = EvidenceGroundingEvaluator(llm)

    result = evaluator.evaluate(
        question="Where is Paris?",
        claims=claims,
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert result.score == 0.0
    assert result.errors
    assert result.evidence_refs == []