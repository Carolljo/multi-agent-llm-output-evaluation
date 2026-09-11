import pytest

from src.v2.agents.completeness import CompletenessEvaluator
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


def build_candidate_claims() -> list[Claim]:
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
    ]


def build_reference_answer() -> ReferenceAnswer:
    return ReferenceAnswer(
        answer=(
            "Paris is the capital of France. "
            "It is located on the Seine River. "
            "The city has a population of over two million."
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
            Claim(
                claim_id="r3",
                text="Paris has a population of over two million.",
                evidence_refs=["e1"],
                importance=0.7,
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
                "on the Seine River. The city has a population "
                "of over two million."
            ),
            source="test-source",
            relevance_score=0.95,
        )
    ]


def test_completeness_evaluator_returns_result():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "reference_claim_id": "r1",
                    "verdict": "covered",
                    "reasoning": "The candidate states that Paris is the capital of France.",
                    "candidate_claim_ids": ["c1"]
                },
                {
                    "reference_claim_id": "r2",
                    "verdict": "covered",
                    "reasoning": "The candidate states that Paris is located on the Seine River.",
                    "candidate_claim_ids": ["c2"]
                },
                {
                    "reference_claim_id": "r3",
                    "verdict": "missing",
                    "reasoning": "The candidate does not mention the population.",
                    "candidate_claim_ids": []
                }
            ],
            "confidence": 0.94,
            "reasoning": "The candidate covers two of three reference claims."
        }
        """
    )

    evaluator = CompletenessEvaluator(llm)

    result = evaluator.evaluate(
        question="What can you tell me about Paris?",
        claims=build_candidate_claims(),
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert result.evaluator == "completeness"
    assert result.score == pytest.approx(2 / 3)
    assert result.confidence == 0.94
    assert result.claims_evaluated == 3
    assert len(result.errors) == 1
    assert result.evidence_refs == []


def test_completeness_score_is_deterministic():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "reference_claim_id": "r1",
                    "verdict": "covered",
                    "reasoning": "Covered.",
                    "candidate_claim_ids": ["c1"]
                },
                {
                    "reference_claim_id": "r2",
                    "verdict": "partially_covered",
                    "reasoning": "Partially covered.",
                    "candidate_claim_ids": ["c2"]
                },
                {
                    "reference_claim_id": "r3",
                    "verdict": "missing",
                    "reasoning": "Missing.",
                    "candidate_claim_ids": []
                }
            ],
            "confidence": 0.8,
            "reasoning": "Mixed completeness."
        }
        """
    )

    evaluator = CompletenessEvaluator(llm)

    result = evaluator.evaluate(
        question="Test question",
        claims=build_candidate_claims(),
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert result.score == pytest.approx(0.5)


def test_completeness_rejects_empty_question():
    llm = FakeLLMClient("{}")
    evaluator = CompletenessEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        evaluator.evaluate(
            question="",
            claims=build_candidate_claims(),
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_completeness_rejects_empty_candidate_claims():
    llm = FakeLLMClient("{}")
    evaluator = CompletenessEvaluator(llm)

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


def test_completeness_rejects_reference_without_claims():
    llm = FakeLLMClient("{}")
    evaluator = CompletenessEvaluator(llm)

    reference = ReferenceAnswer(
        answer="Some answer.",
        claims=[],
        evidence_refs=["e1"],
        confidence=0.9,
    )

    with pytest.raises(
        ValueError,
        match="Reference answer must contain at least one claim",
    ):
        evaluator.evaluate(
            question="Test question",
            claims=build_candidate_claims(),
            reference_answer=reference,
            evidence=build_evidence(),
        )


def test_completeness_rejects_empty_evidence():
    llm = FakeLLMClient("{}")
    evaluator = CompletenessEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="At least one evidence item is required",
    ):
        evaluator.evaluate(
            question="Test question",
            claims=build_candidate_claims(),
            reference_answer=build_reference_answer(),
            evidence=[],
        )


def test_completeness_rejects_invalid_json():
    llm = FakeLLMClient("not valid json")
    evaluator = CompletenessEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="LLM returned invalid JSON",
    ):
        evaluator.evaluate(
            question="Test question",
            claims=build_candidate_claims(),
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_completeness_rejects_incomplete_reference_evaluation():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "reference_claim_id": "r1",
                    "verdict": "covered",
                    "reasoning": "Covered.",
                    "candidate_claim_ids": ["c1"]
                }
            ],
            "confidence": 0.9,
            "reasoning": "Only one reference claim evaluated."
        }
        """
    )

    evaluator = CompletenessEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="LLM did not evaluate every reference claim",
    ):
        evaluator.evaluate(
            question="Test question",
            claims=build_candidate_claims(),
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_completeness_rejects_unknown_candidate_claim_ids():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "reference_claim_id": "r1",
                    "verdict": "covered",
                    "reasoning": "Covered.",
                    "candidate_claim_ids": ["unknown"]
                },
                {
                    "reference_claim_id": "r2",
                    "verdict": "covered",
                    "reasoning": "Covered.",
                    "candidate_claim_ids": ["c2"]
                },
                {
                    "reference_claim_id": "r3",
                    "verdict": "missing",
                    "reasoning": "Missing.",
                    "candidate_claim_ids": []
                }
            ],
            "confidence": 0.9,
            "reasoning": "Evaluation complete."
        }
        """
    )

    evaluator = CompletenessEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="LLM returned unknown candidate claim IDs",
    ):
        evaluator.evaluate(
            question="Test question",
            claims=build_candidate_claims(),
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_completeness_prompt_contains_required_context():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "reference_claim_id": "r1",
                    "verdict": "covered",
                    "reasoning": "Covered.",
                    "candidate_claim_ids": ["c1"]
                },
                {
                    "reference_claim_id": "r2",
                    "verdict": "covered",
                    "reasoning": "Covered.",
                    "candidate_claim_ids": ["c2"]
                },
                {
                    "reference_claim_id": "r3",
                    "verdict": "missing",
                    "reasoning": "Missing.",
                    "candidate_claim_ids": []
                }
            ],
            "confidence": 0.9,
            "reasoning": "Evaluation complete."
        }
        """
    )

    evaluator = CompletenessEvaluator(llm)

    evaluator.evaluate(
        question="What can you tell me about Paris?",
        claims=build_candidate_claims(),
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert "What can you tell me about Paris?" in llm.last_prompt
    assert "Paris is the capital of France." in llm.last_prompt
    assert "Paris is located on the Seine River." in llm.last_prompt
    assert "Paris has a population of over two million." in llm.last_prompt
    assert "[c1]" in llm.last_prompt
    assert "[r1]" in llm.last_prompt
    assert "[e1]" in llm.last_prompt