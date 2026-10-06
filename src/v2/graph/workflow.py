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
from src.v2.memory.nodes import (
    answer_memory_node,
    use_memory_reference_node,
    write_verified_answer_node,
)
from src.v2.memory.retriever import AnswerMemoryRetriever
from src.v2.memory.store import ChromaAnswerMemory
from src.v2.memory.write_policy import MemoryWritePolicy
from src.v2.reference.generator import ReferenceAnswerGenerator
from src.v2.reference.validator import ReferenceAnswerValidator


def route_after_memory(state: EvaluationState) -> str:
    """
    Route the workflow based on the verified answer memory decision.

    A strong verified memory match reuses the stored reference answer.

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

    This allows the memory-hit path to use the same evidence-quality
    scoring stage as the fresh retrieval path.
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

    This stage is deterministic and does not call an LLM.
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
    memory_store: ChromaAnswerMemory | None = None,
    memory_write_policy: MemoryWritePolicy | None = None,
    evidence_quality_scorer: EvidenceQualityScorer | None = None,
):
    """
    Build and compile the complete autonomous V2 evaluation workflow.

    MEMORY HIT:
        Reuse a previously verified reference answer and its evidence.
        Skip fresh RAG retrieval, reference generation, and reference
        validation.

    MEMORY MISS:
        Perform fresh RAG retrieval, generate a reference answer,
        independently validate it, and continue through the normal
        evaluation pipeline.

    Both paths converge at candidate claim extraction and evaluation.

    Freshly verified answers may be persisted into answer memory
    according to the conservative memory write policy.
    """

    if evidence_quality_scorer is None:
        evidence_quality_scorer = EvidenceQualityScorer()

    graph = StateGraph(EvaluationState)

    # =========================================================
    # MEMORY LAYER
    # =========================================================

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

        graph.add_node(
            "use_memory_reference",
            use_memory_reference_node,
        )

    # =========================================================
    # FRESH RAG EVIDENCE RETRIEVAL
    # =========================================================

    graph.add_node(
        "retrieve_evidence",
        lambda state: retrieve_evidence_node(
            state,
            retriever,
            top_k=top_k,
        ),
    )

    # =========================================================
    # EVIDENCE QUALITY
    # =========================================================

    graph.add_node(
        "score_evidence_quality",
        lambda state: score_evidence_quality_node(
            state,
            evidence_quality_scorer,
        ),
    )

    # =========================================================
    # AUTONOMOUS REFERENCE PIPELINE
    # =========================================================

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

    # =========================================================
    # CANDIDATE CLAIM EXTRACTION
    # =========================================================

    graph.add_node(
        "extract_claims",
        lambda state: extract_claims_node(
            state,
            claim_extractor,
        ),
    )

    # =========================================================
    # INDEPENDENT EVALUATORS
    # =========================================================

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

    # =========================================================
    # SYNCHRONIZATION POINT
    # =========================================================

    graph.add_node(
        "collect_evaluations",
        collect_evaluations_node,
    )

    # =========================================================
    # ARBITRATION
    # =========================================================

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
    # MEMORY WRITE
    # =========================================================

    if memory_store is not None:

        graph.add_node(
            "write_verified_answer",
            lambda state: write_verified_answer_node(
                state,
                memory_store=memory_store,
                write_policy=memory_write_policy,
            ),
        )

    # =========================================================
    # START → MEMORY
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

        # -----------------------------------------------------
        # MEMORY HIT PATH
        # -----------------------------------------------------
        #
        # Verified memory evidence is first moved into the
        # shared evidence pool.
        #
        # Then evidence quality is scored.
        #
        # Then the verified memory reference answer is reused.
        #
        # This ordering keeps the memory path structurally
        # compatible with the fresh evidence path.
        # -----------------------------------------------------

        graph.add_edge(
            "use_memory_evidence",
            "score_evidence_quality",
        )

        graph.add_conditional_edges(
            "score_evidence_quality",
            route_after_memory,
            {
                "use_memory": "use_memory_reference",
                "fresh_retrieval": "generate_reference",
            },
        )

        graph.add_edge(
            "use_memory_reference",
            "extract_claims",
        )

    else:

        # -----------------------------------------------------
        # BACKWARD-COMPATIBLE MODE
        # -----------------------------------------------------
        #
        # If memory is not configured, preserve the original
        # V2 workflow.
        # -----------------------------------------------------

        graph.add_edge(
            START,
            "retrieve_evidence",
        )

    # =========================================================
    # FRESH RETRIEVAL PATH
    # =========================================================

    graph.add_edge(
        "retrieve_evidence",
        "score_evidence_quality",
    )

    graph.add_edge(
        "score_evidence_quality",
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

    # =========================================================
    # COMMON CANDIDATE EVALUATION PIPELINE
    # =========================================================

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

    # =========================================================
    # EVALUATOR FAN-IN
    # =========================================================

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

    # =========================================================
    # ARBITRATION
    # =========================================================

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

    # =========================================================
    # FINALIZATION + MEMORY WRITE
    # =========================================================

    if memory_store is not None:

        graph.add_conditional_edges(
            "finalize",
            lambda state: (
                "write_memory"
                if not state.get("memory_hit", False)
                else "end"
            ),
            {
                "write_memory": "write_verified_answer",
                "end": END,
            },
        )

        graph.add_edge(
            "write_verified_answer",
            END,
        )

    else:

        graph.add_edge(
            "finalize",
            END,
        )

    return graph.compile()