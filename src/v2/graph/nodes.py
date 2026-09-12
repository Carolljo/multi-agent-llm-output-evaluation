from src.v2.arbitration.adjudicator import Adjudicator
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


def adjudicate_node(
    state: EvaluationState,
    adjudicator: Adjudicator,
) -> dict:
    """Resolve evaluator disagreement using the adjudicator."""

    disagreement = state["disagreement"]

    if disagreement is None:
        raise ValueError("Disagreement result is missing.")

    reference_answer = state["reference_answer"]

    if reference_answer is None:
        raise ValueError("Reference answer is missing.")

    result = adjudicator.adjudicate(
        question=state["evaluation_input"].question,
        response=state["evaluation_input"].response,
        reference_answer=reference_answer,
        claims=state["claims"],
        evidence=state["evidence"],
        evaluations=state["evaluations"],
        disagreement=disagreement,
    )

    return {
        "adjudication": result,
    }
    
from src.v2.models.schemas import FinalVerdict


def finalize_node(state: EvaluationState) -> dict:
    """Build the final verdict from deterministic aggregation and adjudication."""

    aggregated = state["aggregated"]

    if aggregated is None:
        raise ValueError("Aggregated evaluation is missing.")

    disagreement = state["disagreement"]

    if disagreement is None:
        raise ValueError("Disagreement result is missing.")

    adjudication = state["adjudication"]

    if disagreement.requires_adjudication:
        if adjudication is None:
            raise ValueError(
                "Adjudication result is required when disagreement "
                "requires adjudication."
            )

        verdict = adjudication.verdict
        confidence = adjudication.confidence
        evidence_refs = adjudication.evidence_refs

    else:
        score = aggregated.score

        if score >= 0.80:
            verdict = "accept"
        elif score >= 0.50:
            verdict = "partial"
        else:
            verdict = "reject"

        confidence = aggregated.confidence
        evidence_refs = aggregated.evidence_refs

    return {
        "final_verdict": FinalVerdict(
            verdict=verdict,
            score=aggregated.score,
            confidence=confidence,
            evaluator_results=aggregated.evaluations,
            disagreement=disagreement,
            adjudication=adjudication,
            key_errors=aggregated.errors,
            evidence_refs=evidence_refs,
        )
    }