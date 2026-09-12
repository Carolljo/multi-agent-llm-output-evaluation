import json

import pytest

from src.v2.arbitration.adjudicator import Adjudicator
from src.v2.models.schemas import (
    Adjudication,
    Claim,
    Disagreement,
    Evidence,
    EvaluatorResult,
    ReferenceAnswer,
)


class FakeLLMClient:
    """Fake LLM client used to test adjudicator behavior."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.last_prompt = None
        self.last_response_format = None

    def generate(self, *, prompt: str, response_format: dict) -> str:
        self.last_prompt = prompt
        self.last_response_format = response_format

        return json.dumps(self.payload)


def build_test_inputs():
    reference_answer = ReferenceAnswer(
        answer="The system supports three authentication methods.",
        claims=[
            Claim(
                claim_id="C1",
                text="The system supports three authentication methods.",
                evidence_refs=["E1"],
            )
        ],
        evidence_refs=["E1"],
        confidence=0.9,
    )

    claims = [
        Claim(
            claim_id="C1",
            text="The system supports three authentication methods.",
            evidence_refs=["E1"],
        )
    ]

    evidence = [
        Evidence(
            evidence_id="E1",
            document_id="DOC1",
            chunk_id="CH1",
            content="The system supports three authentication methods.",
            source="test-document",
        )
    ]

    evaluations = [
        EvaluatorResult(
            evaluator="accuracy",
            score=0.9,
            confidence=0.9,
            claims_evaluated=1,
            reasoning="The claim is supported.",
            evidence_refs=["E1"],
        ),
        EvaluatorResult(
            evaluator="completeness",
            score=0.8,
            confidence=0.85,
            claims_evaluated=1,
            reasoning="The response covers the expected information.",
            evidence_refs=["E1"],
        ),
        EvaluatorResult(
            evaluator="logical_consistency",
            score=0.9,
            confidence=0.9,
            claims_evaluated=1,
            reasoning="No contradiction found.",
            evidence_refs=["E1"],
        ),
        EvaluatorResult(
            evaluator="evidence_grounding",
            score=0.4,
            confidence=0.8,
            claims_evaluated=1,
            reasoning="Grounding is weaker than the other evaluators.",
            evidence_refs=["E1"],
        ),
    ]

    disagreement = Disagreement(
        detected=True,
        evaluators=["accuracy", "evidence_grounding"],
        score_range=0.5,
        reason="Evaluator scores differ significantly.",
        requires_adjudication=True,
    )

    return reference_answer, claims, evidence, evaluations, disagreement


def test_adjudicator_returns_valid_result():
    fake_llm = FakeLLMClient(
        {
            "verdict": "accept",
            "confidence": 0.91,
            "reasoning": "The majority of evaluators support the response.",
            "evidence_refs": ["E1"],
        }
    )

    adjudicator = Adjudicator(fake_llm)

    (
        reference_answer,
        claims,
        evidence,
        evaluations,
        disagreement,
    ) = build_test_inputs()

    result = adjudicator.adjudicate(
        question="How many authentication methods does the system support?",
        response="The system supports three authentication methods.",
        reference_answer=reference_answer,
        claims=claims,
        evidence=evidence,
        evaluations=evaluations,
        disagreement=disagreement,
    )

    assert isinstance(result, Adjudication)
    assert result.verdict == "accept"
    assert result.confidence == 0.91
    assert result.evidence_refs == ["E1"]


def test_adjudicator_rejects_invalid_verdict():
    fake_llm = FakeLLMClient(
        {
            "verdict": "mostly_correct",
            "confidence": 0.8,
            "reasoning": "Invalid verdict.",
            "evidence_refs": ["E1"],
        }
    )

    adjudicator = Adjudicator(fake_llm)

    (
        reference_answer,
        claims,
        evidence,
        evaluations,
        disagreement,
    ) = build_test_inputs()

    with pytest.raises(ValueError):
        adjudicator.adjudicate(
            question="Test question",
            response="Test response",
            reference_answer=reference_answer,
            claims=claims,
            evidence=evidence,
            evaluations=evaluations,
            disagreement=disagreement,
        )


def test_adjudication_requires_disagreement():
    fake_llm = FakeLLMClient(
        {
            "verdict": "accept",
            "confidence": 0.9,
            "reasoning": "No disagreement.",
            "evidence_refs": ["E1"],
        }
    )

    adjudicator = Adjudicator(fake_llm)

    (
        reference_answer,
        claims,
        evidence,
        evaluations,
        _,
    ) = build_test_inputs()

    no_disagreement = Disagreement(
        detected=False,
        evaluators=[],
        score_range=0.1,
        reason="Scores are close.",
        requires_adjudication=False,
    )

    with pytest.raises(
        ValueError,
        match="Adjudication is not required",
    ):
        adjudicator.adjudicate(
            question="Test question",
            response="Test response",
            reference_answer=reference_answer,
            claims=claims,
            evidence=evidence,
            evaluations=evaluations,
            disagreement=no_disagreement,
        )


def test_adjudicator_rejects_invalid_evidence_reference():
    fake_llm = FakeLLMClient(
        {
            "verdict": "accept",
            "confidence": 0.9,
            "reasoning": "The response is supported.",
            "evidence_refs": ["E999"],
        }
    )

    adjudicator = Adjudicator(fake_llm)

    (
        reference_answer,
        claims,
        evidence,
        evaluations,
        disagreement,
    ) = build_test_inputs()

    with pytest.raises(
        ValueError,
        match="invalid evidence reference",
    ):
        adjudicator.adjudicate(
            question="Test question",
            response="Test response",
            reference_answer=reference_answer,
            claims=claims,
            evidence=evidence,
            evaluations=evaluations,
            disagreement=disagreement,
        )


def test_adjudicator_rejects_unexpected_score():
    fake_llm = FakeLLMClient(
        {
            "verdict": "accept",
            "score": 0.99,
            "confidence": 0.9,
            "reasoning": "The response is supported.",
            "evidence_refs": ["E1"],
        }
    )

    adjudicator = Adjudicator(fake_llm)

    (
        reference_answer,
        claims,
        evidence,
        evaluations,
        disagreement,
    ) = build_test_inputs()

    with pytest.raises(ValueError):
        adjudicator.adjudicate(
            question="Test question",
            response="Test response",
            reference_answer=reference_answer,
            claims=claims,
            evidence=evidence,
            evaluations=evaluations,
            disagreement=disagreement,
        )
