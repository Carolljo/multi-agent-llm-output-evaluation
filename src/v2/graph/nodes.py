from src.v2.arbitration.adjudicator import Adjudicator
from src.v2.aggregation.aggregator import aggregate_evaluations
from src.v2.aggregation.disagreement import detect_disagreement
from src.v2.aggregation.reliability_gate import apply_reliability_gate
from src.v2.claims.extractor import ClaimExtractor
from src.v2.graph.state import EvaluationState
from src.v2.models.schemas import FinalVerdict
from src.v2.reference.generator import ReferenceAnswerGenerator
from src.v2.reference.validator import ReferenceAnswerValidator
from src.v2.answer.generator import CandidateAnswerGenerator

def retrieve_evidence_node(
    state: EvaluationState,
    retriever,
    top_k: int = 5,
) -> dict:
    """Retrieve evidence relevant to the evaluation question."""

    question = state["evaluation_input"].question

    evidence = retriever.search(
        query=question,
        top_k=top_k,
    )

    if not evidence:
        raise ValueError(
            "No evidence was retrieved for the evaluation question."
        )

    return {
        "evidence": evidence,
    }
def generate_candidate_answer_node(
    state: EvaluationState,
    candidate_generator: CandidateAnswerGenerator,
) -> dict:
    """Generate a candidate answer independently of retrieved evidence."""

    question = state["evaluation_input"].question

    candidate_answer = candidate_generator.generate(
        question=question,
    )

    if not candidate_answer or not candidate_answer.strip():
        raise ValueError(
            "Candidate answer generation returned an empty answer."
        )

    return {
        "candidate_answer": candidate_answer.strip(),
    }

def generate_reference_node(
    state: EvaluationState,
    reference_generator: ReferenceAnswerGenerator,
) -> dict:
    """Generate an evidence-grounded reference answer."""

    question = state["evaluation_input"].question
    evidence = state["evidence"]

    reference_answer = reference_generator.generate(
        question=question,
        evidence=evidence,
    )

    return {
        "reference_answer": reference_answer,
    }


def validate_reference_node(
    state: EvaluationState,
    reference_validator: ReferenceAnswerValidator,
) -> dict:
    """Independently validate the generated reference answer."""

    reference_answer = state["reference_answer"]

    if reference_answer is None:
        raise ValueError("Reference answer is missing.")

    validation = reference_validator.validate(
        question=state["evaluation_input"].question,
        reference_answer=reference_answer,
        evidence=state["evidence"],
    )

    if validation.status != "valid":
        raise ValueError(
            "Generated reference answer failed validation: "
            f"{validation.status}. "
            f"{validation.reasoning}"
        )

    return {
        "reference_validation": validation,
    }


def extract_claims_node(
    state: EvaluationState,
    claim_extractor: ClaimExtractor,
) -> dict:
    """Extract claims from the generated or user-provided candidate answer."""

    reference_answer = state["reference_answer"]

    if reference_answer is None:
        raise ValueError("Reference answer is missing.")

    candidate_answer = (
        state.get("candidate_answer")
        or state["evaluation_input"].response
    )

    if not candidate_answer or not candidate_answer.strip():
        raise ValueError("Candidate answer is missing or empty.")

    claims = claim_extractor.extract(
        question=state["evaluation_input"].question,
        response=candidate_answer,
        reference_answer=reference_answer,
        evidence=state["evidence"],
    )

    if not claims:
        raise ValueError(
            "No evaluable claims were extracted from the candidate response."
        )

    return {
        "claims": claims,
    }

def accuracy_node(
    state: EvaluationState,
    evaluator,
) -> dict:
    """Run the accuracy evaluator."""

    reference_answer = state["reference_answer"]

    if reference_answer is None:
        raise ValueError("Reference answer is missing.")

    result = evaluator.evaluate(
        question=state["evaluation_input"].question,
        claims=state["claims"],
        reference_answer=reference_answer,
        evidence=state["evidence"],
    )

    return {
        "accuracy": result,
    }


def completeness_node(
    state: EvaluationState,
    evaluator,
) -> dict:
    """Run the completeness evaluator."""

    reference_answer = state["reference_answer"]

    if reference_answer is None:
        raise ValueError("Reference answer is missing.")

    result = evaluator.evaluate(
        question=state["evaluation_input"].question,
        claims=state["claims"],
        reference_answer=reference_answer,
        evidence=state["evidence"],
    )

    return {
        "completeness": result,
    }


def logic_node(
    state: EvaluationState,
    evaluator,
) -> dict:
    """Run the logical-consistency evaluator."""

    reference_answer = state["reference_answer"]

    if reference_answer is None:
        raise ValueError("Reference answer is missing.")

    result = evaluator.evaluate(
        question=state["evaluation_input"].question,
        claims=state["claims"],
        reference_answer=reference_answer,
        evidence=state["evidence"],
    )

    return {
        "logic": result,
    }


def grounding_node(
    state: EvaluationState,
    evaluator,
) -> dict:
    """Run the evidence-grounding evaluator."""

    reference_answer = state["reference_answer"]

    if reference_answer is None:
        raise ValueError("Reference answer is missing.")

    result = evaluator.evaluate(
        question=state["evaluation_input"].question,
        claims=state["claims"],
        reference_answer=reference_answer,
        evidence=state["evidence"],
    )

    return {
        "grounding": result,
    }


def collect_evaluations_node(
    state: EvaluationState,
) -> dict:
    """Collect the four independent evaluator outputs."""

    results = [
        state["accuracy"],
        state["completeness"],
        state["logic"],
        state["grounding"],
    ]

    if any(result is None for result in results):
        raise ValueError(
            "All four evaluator results are required before aggregation."
        )

    return {
        "evaluations": results,
    }


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
    """Resolve evaluator disagreement using the evaluated candidate answer."""

    disagreement = state["disagreement"]

    if disagreement is None:
        raise ValueError("Disagreement result is missing.")

    reference_answer = state["reference_answer"]

    if reference_answer is None:
        raise ValueError("Reference answer is missing.")

    candidate_answer = (
        state.get("candidate_answer")
        or state["evaluation_input"].response
    )

    if not candidate_answer or not candidate_answer.strip():
        raise ValueError("Candidate answer is missing or empty.")

    result = adjudicator.adjudicate(
        question=state["evaluation_input"].question,
        response=candidate_answer,
        reference_answer=reference_answer,
        claims=state["claims"],
        evidence=state["evidence"],
        evaluations=state["evaluations"],
        disagreement=disagreement,
    )

    return {
        "adjudication": result,
    }


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

    final_verdict = FinalVerdict(
        verdict=verdict,
        score=aggregated.score,
        confidence=confidence,
        evaluator_results=aggregated.evaluations,
        disagreement=disagreement,
        adjudication=adjudication,
        key_errors=aggregated.errors,
        evidence_refs=evidence_refs,
    )

    gated_verdict = apply_reliability_gate(
        verdict=final_verdict,
        reference_validation=state.get("reference_validation"),
        evaluations=aggregated.evaluations,
        evidence_count=len(state.get("evidence", [])),
        unresolved_disagreement=(
            disagreement.requires_adjudication and adjudication is None
        ),
    )

    return {"final_verdict": gated_verdict}
    
def route_after_answer_memory(state: EvaluationState) -> str:
    """
    Backward-compatible memory routing helper.

    Older V2 tests and callers expect the legacy route names:
        memory_hit
        memory_miss

    The redesigned autonomous workflow uses route_after_memory()
    in workflow.py with the new route names:
        use_memory
        fresh_retrieval

    This compatibility function is intentionally kept separate so
    the redesigned workflow does not need to change its routing API.
    """

    if state.get("memory_hit", False):
        return "memory_hit"

    return "memory_miss"