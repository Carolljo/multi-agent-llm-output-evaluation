import pytest

from src.v2.graph.nodes import (
    adjudicate_node,
    aggregate_node,
    disagreement_node,
    route_after_disagreement,
)
from src.v2.models.schemas import (
    Adjudication,
    Claim,
    Disagreement,
    EvaluationInput,
    Evidence,
    EvaluatorResult,
    FinalVerdict,
    ReferenceAnswer,
)

def build_evaluations():
    return [
        EvaluatorResult(
            evaluator="accuracy",
            score=0.9,
            confidence=0.9,
            claims_evaluated=1,
            reasoning="Supported.",
        ),
        EvaluatorResult(
            evaluator="completeness",
            score=0.8,
            confidence=0.8,
            claims_evaluated=1,
            reasoning="Mostly complete.",
        ),
        EvaluatorResult(
            evaluator="logical_consistency",
            score=0.9,
            confidence=0.9,
            claims_evaluated=1,
            reasoning="Consistent.",
        ),
        EvaluatorResult(
            evaluator="evidence_grounding",
            score=0.4,
            confidence=0.8,
            claims_evaluated=1,
            reasoning="Weak grounding.",
        ),
    ]


def test_aggregate_node():
    state = {
        "evaluations": build_evaluations(),
    }

    result = aggregate_node(state)

    assert "aggregated" in result
    assert result["aggregated"].score == 0.75
    assert result["aggregated"].confidence == pytest.approx(0.85)


def test_disagreement_node():
    state = {
        "evaluations": build_evaluations(),
    }

    result = disagreement_node(state)

    assert "disagreement" in result

    disagreement = result["disagreement"]

    assert disagreement.detected is True
    assert disagreement.requires_adjudication is True
    assert disagreement.score_range == 0.5
    assert "accuracy" in disagreement.evaluators
    assert "evidence_grounding" in disagreement.evaluators


def test_route_after_disagreement_to_adjudication():
    state = {
        "disagreement": Disagreement(
            detected=True,
            evaluators=["accuracy", "evidence_grounding"],
            score_range=0.5,
            reason="Scores differ significantly.",
            requires_adjudication=True,
        )
    }

    assert route_after_disagreement(state) == "adjudicate"


def test_route_after_disagreement_to_finalize():
    state = {
        "disagreement": Disagreement(
            detected=False,
            evaluators=[],
            score_range=0.1,
            reason="Scores are close.",
            requires_adjudication=False,
        )
    }

    assert route_after_disagreement(state) == "finalize"


class FakeAdjudicator:
    """Fake adjudicator used to test graph-node behavior."""

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
            reasoning="The response is supported by the supplied evidence.",
            evidence_refs=["E1"],
        )


def test_adjudicate_node():
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

    state = {
        "evaluation_input": EvaluationInput(
            question="How many authentication methods does the system support?",
            response="The system supports three authentication methods.",
        ),
        "evidence": [
            Evidence(
                evidence_id="E1",
                document_id="DOC1",
                chunk_id="CH1",
                content="The system supports three authentication methods.",
            )
        ],
        "reference_answer": reference_answer,
        "claims": reference_answer.claims,
        "evaluations": build_evaluations(),
        "disagreement": Disagreement(
            detected=True,
            evaluators=["accuracy", "evidence_grounding"],
            score_range=0.5,
            reason="Scores differ significantly.",
            requires_adjudication=True,
        ),
    }

    fake_adjudicator = FakeAdjudicator()

    result = adjudicate_node(
        state,
        fake_adjudicator,
    )

    assert fake_adjudicator.called is True
    assert "adjudication" in result
    assert result["adjudication"].verdict == "accept"
    assert result["adjudication"].confidence == 0.92
    assert result["adjudication"].evidence_refs == ["E1"]
    
def test_finalize_node_without_disagreement():
    from src.v2.graph.nodes import finalize_node

    evaluations = build_evaluations()

    state = {
        "evaluations": evaluations,
        "aggregated": aggregate_node(
            {"evaluations": evaluations}
        )["aggregated"],
        "disagreement": Disagreement(
            detected=False,
            evaluators=[],
            score_range=0.1,
            reason="Scores are close.",
            requires_adjudication=False,
        ),
        "adjudication": None,
    }

    result = finalize_node(state)

    final_verdict = result["final_verdict"]

    assert isinstance(final_verdict, FinalVerdict)
    assert final_verdict.verdict == "partial"
    assert final_verdict.score == pytest.approx(0.75)
    assert final_verdict.confidence == pytest.approx(0.85)
    assert final_verdict.adjudication is None
    assert len(final_verdict.evaluator_results) == 4


def test_finalize_node_with_adjudication():
    from src.v2.graph.nodes import finalize_node

    evaluations = build_evaluations()

    disagreement = Disagreement(
        detected=True,
        evaluators=["accuracy", "evidence_grounding"],
        score_range=0.5,
        reason="Scores differ significantly.",
        requires_adjudication=True,
    )

    adjudication = Adjudication(
        verdict="accept",
        confidence=0.92,
        reasoning="The response is supported by the supplied evidence.",
        evidence_refs=["E1"],
    )

    state = {
        "evaluations": evaluations,
        "aggregated": aggregate_node(
            {"evaluations": evaluations}
        )["aggregated"],
        "disagreement": disagreement,
        "adjudication": adjudication,
    }

    result = finalize_node(state)

    final_verdict = result["final_verdict"]

    assert isinstance(final_verdict, FinalVerdict)

    # Verdict comes from adjudication.
    assert final_verdict.verdict == "accept"

    # Score remains deterministic from aggregation.
    assert final_verdict.score == pytest.approx(0.75)

    # Confidence comes from adjudication when disagreement exists.
    assert final_verdict.confidence == pytest.approx(0.92)

    assert final_verdict.adjudication == adjudication
    assert final_verdict.evidence_refs == ["E1"]
    assert len(final_verdict.evaluator_results) == 4