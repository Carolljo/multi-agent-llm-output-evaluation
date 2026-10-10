from typing import Any

from src.v2.memory.decision import MemoryDecision, MemoryDecisionPolicy
from src.v2.memory.retriever import AnswerMemoryRetriever
from src.v2.memory.store import ChromaAnswerMemory
from src.v2.memory.write_policy import MemoryWritePolicy
from src.v2.models.schemas import Claim, ReferenceAnswer


def answer_memory_node(
    state: dict[str, Any],
    *,
    memory_retriever: AnswerMemoryRetriever,
    decision_policy: MemoryDecisionPolicy | None = None,
) -> dict[str, Any]:
    """
    Search verified answer memory and decide whether the best
    candidate is strong enough to be reused.

    This node does not generate an answer and does not call an LLM.
    """

    if decision_policy is None:
        decision_policy = MemoryDecisionPolicy()

    evaluation_input = state["evaluation_input"]

    if isinstance(evaluation_input, dict):
        question = evaluation_input["question"]
    else:
        question = evaluation_input.question

    matches = memory_retriever.search(question)

    if not matches:
        return {
            "memory_matches": [],
            "memory_hit": False,
            "memory_decision": None,
            "memory_decision_reason": (
                "No verified answer memory matched the question."
            ),
            "memory_similarity": None,
            "memory_validation_score": None,
            "memory_validation_confidence": None,
            "memory_decision_score": None,
            "memory_answer": None,
            "memory_evidence": [],
        }

    best_match = matches[0]

    decision: MemoryDecision = decision_policy.evaluate(best_match)

    if decision.decision == "use_memory":
        memory = best_match.memory

        return {
            "memory_matches": matches,
            "memory_hit": True,
            "memory_decision": decision,
            "memory_decision_reason": decision.reason,
            "memory_similarity": decision.similarity,
            "memory_validation_score": decision.validation_score,
            "memory_validation_confidence": decision.validation_confidence,
            "memory_decision_score": decision.decision_score,
            "memory_answer": memory.answer,
            "memory_evidence": memory.evidence,
        }

    return {
        "memory_matches": matches,
        "memory_hit": False,
        "memory_decision": decision,
        "memory_decision_reason": decision.reason,
        "memory_similarity": decision.similarity,
        "memory_validation_score": decision.validation_score,
        "memory_validation_confidence": decision.validation_confidence,
        "memory_decision_score": decision.decision_score,
        "memory_answer": None,
        "memory_evidence": [],
    }


def use_memory_reference_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert a verified memory entry into the ReferenceAnswer model
    expected by the existing V2 evaluation pipeline.

    Reuses the stored reference answer and evidence without calling
    the reference generator or validator.
    The candidate response is still evaluated normally.
    """

    matches = state.get("memory_matches", [])

    if not matches:
        raise ValueError(
            "Memory hit occurred but no memory match is available."
        )

    memory = matches[0].memory

    # Convert stored claim dictionaries into Claim model instances.
    claims = [
        claim
        if isinstance(claim, Claim)
        else Claim.model_validate(claim)
        for claim in memory.claims
    ]

    # Reconstruct the expected ReferenceAnswer model.
    reference_answer = ReferenceAnswer(
        answer=memory.answer,
        claims=claims,
        evidence_refs=memory.evidence_refs,
        confidence=memory.validation_confidence,
    )

    return {
        "reference_answer": reference_answer,
        "evidence": memory.evidence,
    }


def write_verified_answer_node(
    state: dict[str, Any],
    *,
    memory_store: ChromaAnswerMemory,
    write_policy: MemoryWritePolicy | None = None,
) -> dict[str, Any]:
    """
    Persist a freshly generated reference answer only when the
    final evaluation satisfies the conservative memory write policy.
    """

    if write_policy is None:
        write_policy = MemoryWritePolicy()

    final_verdict = state.get("final_verdict")
    validation = state.get("reference_validation")
    reference_answer = state.get("reference_answer")
    evidence = state.get("evidence", [])

    if final_verdict is None:
        raise ValueError("Final verdict is required before memory write.")

    if validation is None:
        raise ValueError(
            "Reference validation is required before memory write."
        )

    if reference_answer is None:
        raise ValueError(
            "Reference answer is required before memory write."
        )

    disagreement = final_verdict.disagreement

    unresolved_disagreement = bool(
        disagreement is not None
        and disagreement.requires_adjudication
        and final_verdict.adjudication is None
    )

    decision = write_policy.evaluate(
        verdict=final_verdict.verdict,
        validation_score=validation.score,
        validation_confidence=validation.confidence,
        aggregated_confidence=final_verdict.confidence,
        evidence_count=len(evidence),
        unresolved_disagreement=unresolved_disagreement,
    )

    stored_memory = None

    if decision.decision == "store":
        stored_memory = memory_store.save(
            question=state["evaluation_input"].question,
            reference_answer=reference_answer,
            validation=validation,
            evidence=evidence,
        )

    return {
        "memory_write_decision": decision,
        "memory_stored": stored_memory is not None,
    }