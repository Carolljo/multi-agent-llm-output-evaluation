from langgraph.graph import END, START, StateGraph

from src.v2.evidence.quality import EvidenceQualityScorer
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
from src.v2.memory.decision import MemoryDecisionPolicy
from src.v2.memory.nodes import answer_memory_node
from src.v2.memory.retriever import AnswerMemoryRetriever
from src.v2.reference.generator import ReferenceAnswerGenerator
from src.v2.reference.validator import ReferenceAnswerValidator


def route_after_memory(state: EvaluationState) -> str:
    """
    Route the workflow based on the verified answer memory decision.

    A strong verified memory match uses stored memory evidence.
    Otherwise, the workflow performs fresh RAG retrieval.
    """

    if state.get("memory_hit", False):
        return "use_memory"

    return "fresh_retrieval"


def use_memory_evidence_node(
    state: EvaluationState,
):
    """
    Move verified memory evidence into the shared evidence pool.

    Memory answers are never treated as automatically correct.
    Only the stored evidence is passed into the normal V2
    verification pipeline.
    """

    memory_evidence = state.get("memory_evidence", [])

    if not memory_evidence:
        raise ValueError(
            "Memory hit occurred but no memory evidence was available."
        )

    return {
        "evidence": memory_evidence,
    }


def score_evidence_quality_node(
    state: EvaluationState,
    scorer: EvidenceQualityScorer,
):
    """
    Score the quality of every evidence item in the shared evidence pool.

    This does not remove evidence. It creates deterministic quality
    metadata that can be used by later verification and audit logic.
    """

    evidence = state.get("evidence", [])

    if not evidence:
        raise ValueError(
            "Cannot score evidence quality because the evidence pool is empty."
        )

    quality_scores = scorer.score_many(evidence)

    return {
        "evidence_quality": quality_scores,
    }


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
    memory_retriever: AnswerMemoryRetriever | None = None,
    memory_decision_policy: MemoryDecisionPolicy | None = None,
    evidence_quality_scorer: EvidenceQualityScorer | None = None,
):
    """
    Build and compile the complete autonomous V2 evaluation workflow.

    The workflow first searches verified answer memory.

    If a sufficiently strong verified memory match exists,
    its evidence is reused.

    Otherwise, fresh RAG retrieval is performed.

    Both paths converge into the same evidence-quality,
    reference generation, validation, claim evaluation,
    aggregation, disagreement, adjudication, and finalization
    pipeline.
    """

    if evidence_quality_scorer is None:
        evidence_quality_scorer = EvidenceQualityScorer()

    graph = StateGraph(EvaluationState)

    # ---------------------------------------------------------
    # Memory layer
    # ---------------------------------------------------------

    if memory_retriever is not None:

        graph.add_node(
            "answer_memory",
            lambda state: answer_memory_node(
                state,
                memory_retriever=memory_retriever,
                decision_policy=memory_decision_policy,
            ),
        )

        graph.add_node(
            "use_memory_evidence",
            use_memory_evidence_node,
        )

    # ---------------------------------------------------------
    # Fresh RAG evidence retrieval
    # ---------------------------------------------------------

    graph.add_node(
        "retrieve_evidence",
        lambda state: retrieve_evidence_node(
            state,
            retriever,
            top_k=top_k,
        ),
    )

    # ---------------------------------------------------------
    # Evidence quality
    # ---------------------------------------------------------

    graph.add_node(
        "score_evidence_quality",
        lambda state: score_evidence_quality_node(
            state,
            evidence_quality_scorer,
        ),
    )

    # ---------------------------------------------------------
    # Autonomous reference pipeline
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Candidate claim extraction
    # ---------------------------------------------------------

    graph.add_node(
        "extract_claims",
        lambda state: extract_claims_node(
            state,
            claim_extractor,
        ),
    )

    # ---------------------------------------------------------
    # Independent evaluators
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Synchronization point
    # ---------------------------------------------------------

    graph.add_node(
        "collect_evaluations",
        collect_evaluations_node,
    )

    # ---------------------------------------------------------
    # Arbitration
    # ---------------------------------------------------------

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

    # =========================================================
    # START → MEMORY → EVIDENCE → QUALITY
    # =========================================================

    if memory_retriever is not None:

        graph.add_edge(
            START,
            "answer_memory",
        )

        graph.add_conditional_edges(
            "answer_memory",
            route_after_memory,
            {
                "use_memory": "use_memory_evidence",
                "fresh_retrieval": "retrieve_evidence",
            },
        )

        graph.add_edge(
            "use_memory_evidence",
            "score_evidence_quality",
        )

    else:

        # Backward-compatible mode:
        # if no memory retriever is supplied, use the
        # original V2 workflow directly.
        graph.add_edge(
            START,
            "retrieve_evidence",
        )

    # Fresh retrieval path
    graph.add_edge(
        "retrieve_evidence",
        "score_evidence_quality",
    )

    # Both evidence sources converge at the same
    # deterministic evidence-quality assessment stage.
    graph.add_edge(
        "score_evidence_quality",
        "generate_reference",
    )

    # =========================================================
    # EXISTING V2 PIPELINE
    # =========================================================

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