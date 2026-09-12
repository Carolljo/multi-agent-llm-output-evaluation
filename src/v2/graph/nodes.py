from src.v2.aggregation.aggregator import aggregate_evaluations
from src.v2.aggregation.disagreement import detect_disagreement
from src.v2.graph.state import EvaluationState


def aggregate_node(state: EvaluationState) -> dict:
    """Aggregate independent evaluator results deterministically."""

    evaluations = state["evaluations"]

    aggregated = aggregate_evaluations(evaluations)

    return {
        "aggregated": aggregated,
    }


def disagreement_node(state: EvaluationState) -> dict:
    """Detect meaningful disagreement between evaluators."""

    evaluations = state["evaluations"]

    disagreement = detect_disagreement(evaluations)

    return {
        "disagreement": disagreement,
    }
    
def route_after_disagreement(state: EvaluationState) -> str:
    """Route to adjudication when evaluator disagreement requires it."""

    disagreement = state["disagreement"]

    if disagreement is None:
        raise ValueError("Disagreement result is missing.")

    if disagreement.requires_adjudication:
        return "adjudicate"

    return "finalize"