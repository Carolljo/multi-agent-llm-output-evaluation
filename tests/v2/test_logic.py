import pytest

from src.v2.agents.logic import LogicalConsistencyEvaluator
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
            text="The company was founded in 1995.",
            evidence_refs=["e1"],
            importance=1.0,
        ),
        Claim(
            claim_id="c2",
            text="The company is headquartered in London.",
            evidence_refs=["e1"],
            importance=0.8,
        ),
        Claim(
            claim_id="c3",
            text="The company has more than 500 employees.",
            evidence_refs=["e1"],
            importance=0.7,
        ),
    ]


def build_reference_answer() -> ReferenceAnswer:
    return ReferenceAnswer(
        answer=(
            "The company was founded in 1995, is headquartered "
            "in London, and has more than 500 employees."
        ),
        claims=[
            Claim(
                claim_id="r1",
                text="The company was founded in 1995.",
                evidence_refs=["e1"],
                importance=1.0,
            ),
            Claim(
                claim_id="r2",
                text="The company is headquartered in London.",
                evidence_refs=["e1"],
                importance=0.8,
            ),
            Claim(
                claim_id="r3",
                text="The company has more than 500 employees.",
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
                "The company was founded in 1995, is headquartered "
                "in London, and has more than 500 employees."
            ),
            source="test-source",
            relevance_score=0.95,
        )
    ]


def test_logic_evaluator_returns_result():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "The founding date does not conflict with other claims."
                },
                {
                    "claim_id": "c2",
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "The headquarters claim does not conflict with other claims."
                },
                {
                    "claim_id": "c3",
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "The employee count does not conflict with other claims."
                }
            ],
            "confidence": 0.95,
            "reasoning": "All candidate claims are mutually consistent."
        }
        """
    )

    evaluator = LogicalConsistencyEvaluator(llm)

    result = evaluator.evaluate(
        question="Tell me about the company.",
        claims=build_claims(),
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert result.evaluator == "logical_consistency"
    assert result.score == 1.0
    assert result.confidence == 0.95
    assert result.claims_evaluated == 3
    assert result.errors == []
    assert result.evidence_refs == []


def test_logic_detects_direct_contradiction():
    claims = [
        Claim(
            claim_id="c1",
            text="The company was founded in 1995.",
            evidence_refs=["e1"],
            importance=1.0,
        ),
        Claim(
            claim_id="c2",
            text="The company was founded in 2003.",
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
                    "verdict": "contradicted",
                    "conflicting_claim_ids": ["c2"],
                    "reasoning": "Claim c2 gives a conflicting founding year."
                },
                {
                    "claim_id": "c2",
                    "verdict": "contradicted",
                    "conflicting_claim_ids": ["c1"],
                    "reasoning": "Claim c1 gives a conflicting founding year."
                }
            ],
            "confidence": 0.98,
            "reasoning": "The two founding dates directly contradict each other."
        }
        """
    )

    evaluator = LogicalConsistencyEvaluator(llm)

    result = evaluator.evaluate(
        question="When was the company founded?",
        claims=claims,
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert result.score == 0.0
    assert result.claims_evaluated == 2
    assert len(result.errors) == 2


def test_logic_score_is_deterministic():
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
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "Consistent."
                },
                {
                    "claim_id": "c2",
                    "verdict": "partially_inconsistent",
                    "conflicting_claim_ids": ["c3"],
                    "reasoning": "Partially inconsistent."
                },
                {
                    "claim_id": "c3",
                    "verdict": "contradicted",
                    "conflicting_claim_ids": ["c2"],
                    "reasoning": "Contradicted."
                },
                {
                    "claim_id": "c4",
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "Consistent."
                }
            ],
            "confidence": 0.8,
            "reasoning": "Mixed consistency."
        }
        """
    )

    evaluator = LogicalConsistencyEvaluator(llm)

    result = evaluator.evaluate(
        question="Test question.",
        claims=claims,
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert result.score == pytest.approx(0.625)


def test_logic_rejects_empty_question():
    llm = FakeLLMClient("{}")
    evaluator = LogicalConsistencyEvaluator(llm)

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


def test_logic_rejects_empty_claims():
    llm = FakeLLMClient("{}")
    evaluator = LogicalConsistencyEvaluator(llm)

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


def test_logic_rejects_empty_evidence():
    llm = FakeLLMClient("{}")
    evaluator = LogicalConsistencyEvaluator(llm)

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


def test_logic_rejects_invalid_json():
    llm = FakeLLMClient("not valid json")
    evaluator = LogicalConsistencyEvaluator(llm)

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


def test_logic_rejects_incomplete_claim_evaluation():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "Consistent."
                }
            ],
            "confidence": 0.9,
            "reasoning": "Only one claim evaluated."
        }
        """
    )

    evaluator = LogicalConsistencyEvaluator(llm)

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


def test_logic_rejects_unknown_conflicting_claim_ids():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "contradicted",
                    "conflicting_claim_ids": ["unknown"],
                    "reasoning": "Conflict detected."
                },
                {
                    "claim_id": "c2",
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "Consistent."
                },
                {
                    "claim_id": "c3",
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "Consistent."
                }
            ],
            "confidence": 0.9,
            "reasoning": "Evaluation complete."
        }
        """
    )

    evaluator = LogicalConsistencyEvaluator(llm)

    with pytest.raises(
        ValueError,
        match="LLM returned unknown conflicting claim IDs",
    ):
        evaluator.evaluate(
            question="Test question.",
            claims=build_claims(),
            reference_answer=build_reference_answer(),
            evidence=build_evidence(),
        )


def test_logic_rejects_missing_candidate_claim_id():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "Consistent."
                },
                {
                    "claim_id": "c2",
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "Consistent."
                },
                {
                    "claim_id": "wrong_id",
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "Consistent."
                }
            ],
            "confidence": 0.9,
            "reasoning": "Evaluation complete."
        }
        """
    )

    evaluator = LogicalConsistencyEvaluator(llm)

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


def test_logic_prompt_contains_required_context():
    llm = FakeLLMClient(
        """
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "Consistent."
                },
                {
                    "claim_id": "c2",
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "Consistent."
                },
                {
                    "claim_id": "c3",
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "Consistent."
                }
            ],
            "confidence": 0.9,
            "reasoning": "Evaluation complete."
        }
        """
    )

    evaluator = LogicalConsistencyEvaluator(llm)

    evaluator.evaluate(
        question="Tell me about the company.",
        claims=build_claims(),
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert "Tell me about the company." in llm.last_prompt
    assert "The company was founded in 1995." in llm.last_prompt
    assert "The company is headquartered in London." in llm.last_prompt
    assert "The company has more than 500 employees." in llm.last_prompt
    assert "[c1]" in llm.last_prompt
    assert "[r1]" in llm.last_prompt
    assert "[e1]" in llm.last_prompt


def test_logic_does_not_treat_reference_disagreement_as_internal_contradiction():
    claims = [
        Claim(
            claim_id="c1",
            text="The company was founded in 2003.",
            evidence_refs=["e1"],
            importance=1.0,
        ),
        Claim(
            claim_id="c2",
            text="The company is headquartered in London.",
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
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "The claim does not contradict another candidate claim."
                },
                {
                    "claim_id": "c2",
                    "verdict": "consistent",
                    "conflicting_claim_ids": [],
                    "reasoning": "The claim does not contradict another candidate claim."
                }
            ],
            "confidence": 0.85,
            "reasoning": "The candidate claims are internally consistent even though one differs from the reference."
        }
        """
    )

    evaluator = LogicalConsistencyEvaluator(llm)

    result = evaluator.evaluate(
        question="Tell me about the company.",
        claims=claims,
        reference_answer=build_reference_answer(),
        evidence=build_evidence(),
    )

    assert result.score == 1.0