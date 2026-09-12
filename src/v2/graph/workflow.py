from langgraph.graph import END, START, StateGraph

from src.v2.arbitration.adjudicator import Adjudicator
from src.v2.claims.extractor import ClaimExtractor
from src.v2.graph.nodes import (
    accuracy_node,
    adjudicate_node,
    aggregate_node,
    collect_evaluations_node,
    completeness_node,
    disagreement_node,
    extract_claims_node,
    finalize_node,
    generate_reference_node,
    grounding_node,
    logic_node,
    retrieve_evidence_node,
    route_after_disagreement,
    validate_reference_node,
)
from src.v2.graph.state import EvaluationState
from src.v2.reference.generator import ReferenceAnswerGenerator
from src.v2.reference.validator import ReferenceAnswerValidator


def build_arbitration_workflow(
    adjudicator: Adjudicator,
):
    """Build and compile the existing arbitration-only workflow."""

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


def build_autonomous_workflow(
    retriever,
    reference_generator: ReferenceAnswerGenerator,
    reference_validator: ReferenceAnswerValidator,
    claim_extractor: ClaimExtractor,
    accuracy_evaluator,
    completeness_evaluator,
    logic_evaluator,
    grounding_evaluator,
    adjudicator: Adjudicator,
    top_k: int = 5,
):
    """Build and compile the complete autonomous V2 evaluation workflow."""

    graph = StateGraph(EvaluationState)

    # Evidence retrieval
    graph.add_node(
        "retrieve_evidence",
        lambda state: retrieve_evidence_node(
            state,
            retriever,
            top_k=top_k,
        ),
    )

    # Autonomous reference pipeline
    graph.add_node(
        "generate_reference",
        lambda state: generate_reference_node(
            state,
            reference_generator,
        ),
    )

    graph.add_node(
        "validate_reference",
        lambda state: validate_reference_node(
            state,
            reference_validator,
        ),
    )

    # Candidate claim extraction
    graph.add_node(
        "extract_claims",
        lambda state: extract_claims_node(
            state,
            claim_extractor,
        ),
    )

    # Independent evaluators
    graph.add_node(
        "accuracy",
        lambda state: accuracy_node(
            state,
            accuracy_evaluator,
        ),
    )

    graph.add_node(
        "completeness",
        lambda state: completeness_node(
            state,
            completeness_evaluator,
        ),
    )

    graph.add_node(
        "logic",
        lambda state: logic_node(
            state,
            logic_evaluator,
        ),
    )

    graph.add_node(
        "grounding",
        lambda state: grounding_node(
            state,
            grounding_evaluator,
        ),
    )

    # Synchronization point after parallel evaluation
    graph.add_node(
        "collect_evaluations",
        collect_evaluations_node,
    )

    # Arbitration
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

    # Sequential autonomous pipeline
    graph.add_edge(
        START,
        "retrieve_evidence",
    )

    graph.add_edge(
        "retrieve_evidence",
        "generate_reference",
    )

    graph.add_edge(
        "generate_reference",
        "validate_reference",
    )

    graph.add_edge(
        "validate_reference",
        "extract_claims",
    )

    # Fan out to four independent evaluators
    graph.add_edge(
        "extract_claims",
        "accuracy",
    )

    graph.add_edge(
        "extract_claims",
        "completeness",
    )

    graph.add_edge(
        "extract_claims",
        "logic",
    )

    graph.add_edge(
        "extract_claims",
        "grounding",
    )

    # Fan in
    graph.add_edge(
        "accuracy",
        "collect_evaluations",
    )

    graph.add_edge(
        "completeness",
        "collect_evaluations",
    )

    graph.add_edge(
        "logic",
        "collect_evaluations",
    )

    graph.add_edge(
        "grounding",
        "collect_evaluations",
    )

    graph.add_edge(
        "collect_evaluations",
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