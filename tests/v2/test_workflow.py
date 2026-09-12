import pytest

from src.v2.graph.workflow import build_arbitration_workflow
from src.v2.models.schemas import (
    Adjudication,
    Claim,
    Disagreement,
    EvaluationInput,
    Evidence,
    EvaluatorResult,
    ReferenceAnswer,
)


class FakeAdjudicator:
    """Fake adjudicator for workflow testing."""

    def __init__(self):
        self.called = False

    def adjudicate(
        self,
        *,
        question,
        response,
        reference_answer,
        claims,
        evidence,
        evaluations,
        disagreement,
    ):
        self.called = True

        return Adjudication(
            verdict="accept",
            confidence=0.92,
            reasoning="The response is supported.",
            evidence_refs=["E1"],
        )


def build_base_state():
    reference_answer = ReferenceAnswer(
        answer="Three authentication methods are supported.",
        claims=[
            Claim(
                claim_id="C1",
                text="Three authentication methods are supported.",
                evidence_refs=["E1"],
            )
        ],
        evidence_refs=["E1"],
        confidence=0.9,
    )

    return {
        "evaluation_input": EvaluationInput(
            question="How many authentication methods are supported?",
            response="Three authentication methods are supported.",
        ),
        "evidence": [
            Evidence(
                evidence_id="E1",
                document_id="DOC1",
                chunk_id="CH1",
                content="Three authentication methods are supported.",
                source="test-document",
            )
        ],
        "reference_answer": reference_answer,
        "reference_validation": None,
        "claims": reference_answer.claims,
        "accuracy": None,
        "completeness": None,
        "logic": None,
        "grounding": None,
        "evaluations": [
            EvaluatorResult(
                evaluator="accuracy",
                score=0.9,
                confidence=0.9,
                claims_evaluated=1,
                errors=[],
                reasoning="Supported.",
                evidence_refs=["E1"],
            ),
            EvaluatorResult(
                evaluator="completeness",
                score=0.8,
                confidence=0.8,
                claims_evaluated=1,
                errors=[],
                reasoning="Complete.",
                evidence_refs=["E1"],
            ),
            EvaluatorResult(
                evaluator="logical_consistency",
                score=0.9,
                confidence=0.9,
                claims_evaluated=1,
                errors=[],
                reasoning="Consistent.",
                evidence_refs=["E1"],
            ),
            EvaluatorResult(
                evaluator="evidence_grounding",
                score=0.8,
                confidence=0.8,
                claims_evaluated=1,
                errors=[],
                reasoning="Grounded.",
                evidence_refs=["E1"],
            ),
        ],
        "aggregated": None,
        "disagreement": None,
        "adjudication": None,
        "final_verdict": None,
    }


def test_workflow_without_disagreement():
    fake_adjudicator = FakeAdjudicator()

    workflow = build_arbitration_workflow(fake_adjudicator)

    state = build_base_state()

    result = workflow.invoke(state)

    assert result["aggregated"].score == pytest.approx(0.85)
    assert result["aggregated"].confidence == pytest.approx(0.85)

    assert result["disagreement"].detected is False
    assert result["adjudication"] is None

    assert result["final_verdict"].verdict == "accept"
    assert result["final_verdict"].score == pytest.approx(0.85)

    assert fake_adjudicator.called is False


def test_workflow_with_disagreement():
    fake_adjudicator = FakeAdjudicator()

    workflow = build_arbitration_workflow(fake_adjudicator)

    state = build_base_state()

    state["evaluations"][3] = EvaluatorResult(
        evaluator="evidence_grounding",
        score=0.4,
        confidence=0.8,
        claims_evaluated=1,
        errors=[],
        reasoning="Weak grounding.",
        evidence_refs=["E1"],
    )

    result = workflow.invoke(state)

    assert result["aggregated"].score == pytest.approx(0.75)
    assert result["aggregated"].confidence == pytest.approx(0.85)

    assert result["disagreement"].detected is True
    assert result["disagreement"].requires_adjudication is True

    assert fake_adjudicator.called is True

    assert result["adjudication"].verdict == "accept"
    assert result["adjudication"].confidence == pytest.approx(0.92)

    # Numerical score remains deterministic.
    assert result["final_verdict"].score == pytest.approx(0.75)

    # Qualitative verdict comes from adjudication.
    assert result["final_verdict"].verdict == "accept"