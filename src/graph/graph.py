from langgraph.graph import END, START, StateGraph

from src.agents.accuracy import AccuracyEvaluator
from src.agents.adjudicator import Adjudicator
from src.agents.completeness import CompletenessEvaluator
from src.agents.logic import LogicEvaluator
from src.aggregation.aggregator import EvaluatorAggregator
from src.aggregation.disagreement import DisagreementDetector
from src.graph.nodes import (
    create_accuracy_node,
    create_adjudication_node,
    create_aggregation_node,
    create_completeness_node,
    create_logic_node,
    finalize_result,
    route_after_aggregation,
)
from src.graph.state import EvaluationState
from src.llm.client import LLMClient


def create_evaluation_graph(
    model: str = "qwen3:1.7b",
    accuracy_evaluator=None,
    logic_evaluator=None,
    completeness_evaluator=None,
    aggregator=None,
    adjudicator=None,
):
    if (
        accuracy_evaluator is None
        or logic_evaluator is None
        or completeness_evaluator is None
        or aggregator is None
        or adjudicator is None
    ):
        llm_client = LLMClient(model=model)

        accuracy_evaluator = (
            accuracy_evaluator
            or AccuracyEvaluator(llm_client)
        )

        logic_evaluator = (
            logic_evaluator
            or LogicEvaluator(llm_client)
        )

        completeness_evaluator = (
            completeness_evaluator
            or CompletenessEvaluator(llm_client)
        )

        if aggregator is None:
            disagreement_detector = DisagreementDetector()
            aggregator = EvaluatorAggregator(disagreement_detector)

        adjudicator = (
            adjudicator
            or Adjudicator(llm_client)
        )

    accuracy_node = create_accuracy_node(accuracy_evaluator)
    logic_node = create_logic_node(logic_evaluator)
    completeness_node = create_completeness_node(
        completeness_evaluator
    )

    aggregation_node = create_aggregation_node(aggregator)
    adjudication_node = create_adjudication_node(adjudicator)

    graph = StateGraph(EvaluationState)

    # Register nodes
    graph.add_node("accuracy", accuracy_node)
    graph.add_node("logic", logic_node)
    graph.add_node("completeness", completeness_node)
    graph.add_node("aggregate", aggregation_node)
    graph.add_node("adjudicate", adjudication_node)
    graph.add_node("finalize", finalize_result)

    # Start independent evaluators
    graph.add_edge(START, "accuracy")
    graph.add_edge(START, "logic")
    graph.add_edge(START, "completeness")

    # Evaluators feed into aggregation
    graph.add_edge("accuracy", "aggregate")
    graph.add_edge("logic", "aggregate")
    graph.add_edge("completeness", "aggregate")

    # Conditional adjudication routing
    graph.add_conditional_edges(
        "aggregate",
        route_after_aggregation,
        {
            "adjudicate": "adjudicate",
            "finalize": "finalize",
        },
    )

    # Adjudication feeds finalization
    graph.add_edge("adjudicate", "finalize")

    # Finalization terminates the workflow
    graph.add_edge("finalize", END)

    return graph.compile()