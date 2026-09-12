import pytest

from src.v2.graph.nodes import aggregate_node, disagreement_node
from src.v2.models.schemas import EvaluatorResult
from src.v2.models.schemas import Disagreement
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
    from src.v2.graph.nodes import route_after_disagreement

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
    from src.v2.graph.nodes import route_after_disagreement

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