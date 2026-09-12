from langgraph.graph import END, START, StateGraph

from src.v2.arbitration.adjudicator import Adjudicator
from src.v2.graph.nodes import (
    adjudicate_node,
    aggregate_node,
    disagreement_node,
    finalize_node,
    route_after_disagreement,
)
from src.v2.graph.state import EvaluationState


def build_arbitration_workflow(
    adjudicator: Adjudicator,
):
    """Build and compile the V2 arbitration workflow."""

    graph = StateGraph(EvaluationState)

    graph.add_node(
        "aggregate",
        aggregate_node,
    )

    graph.add_node(
        "disagreement",
        disagreement_node,
    )

    graph.add_node(
        "adjudicate",
        lambda state: adjudicate_node(
            state,
            adjudicator,
        ),
    )

    graph.add_node(
        "finalize",
        finalize_node,
    )

    graph.add_edge(
        START,
        "aggregate",
    )

    graph.add_edge(
        "aggregate",
        "disagreement",
    )

    graph.add_conditional_edges(
        "disagreement",
        route_after_disagreement,
        {
            "adjudicate": "adjudicate",
            "finalize": "finalize",
        },
    )

    graph.add_edge(
        "adjudicate",
        "finalize",
    )

    graph.add_edge(
        "finalize",
        END,
    )

    return graph.compile()