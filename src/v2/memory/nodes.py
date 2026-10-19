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
    candidate is safe to reuse.

    Entries without complete reference validation are rejected.
    This node does not generate answers or call an LLM.
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
    memory = best_match.memory

    decision: MemoryDecision = decision_policy.evaluate(best_match)

    # Legacy memory records may contain validation scores but not the
    # complete ReferenceValidation object. They must not bypass the gate.
    reference_validation = getattr(memory, "reference_validation", None)

    if decision.decision == "use_memory" and reference_validation is None:
        return {
            "memory_matches": matches,
            "memory_hit": False,
            "memory_decision": decision,
            "memory_decision_reason": (
                "The memory candidate passed the similarity policy, "
                "but its complete reference validation is unavailable. "
                "Fresh evidence retrieval and reference validation are required."
            ),
            "memory_similarity": decision.similarity,
            "memory_validation_score": decision.validation_score,
            "memory_validation_confidence": decision.validation_confidence,
            "memory_decision_score": decision.decision_score,
            "memory_answer": None,
            "memory_evidence": [],
        }

    if decision.decision == "use_memory":
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

    Reuses the stored reference answer, complete reference validation,
    and evidence without calling the reference generator or validator.

    The candidate response is still evaluated normally.
    """

    if not state.get("memory_hit", False):
        raise ValueError(
            "Cannot reuse a memory reference when memory_hit is false."
        )

    matches = state.get("memory_matches", [])

    if not matches:
        raise ValueError(
            "Memory hit occurred but no memory match is available."
        )

    memory = matches[0].memory

    reference_validation = getattr(memory, "reference_validation", None)

    if reference_validation is None:
        raise ValueError(
            "Stored memory has no complete reference validation. "
            "Fresh reference generation and validation are required."
        )

    if reference_validation.status != "valid":
        raise ValueError(
            "Stored reference validation is not valid. "
            "The memory entry cannot be reused."
        )

    # Convert stored claims into Claim model instances when necessary.
    claims = [
        claim
        if isinstance(claim, Claim)
        else Claim.model_validate(claim)
        for claim in memory.claims
    ]

    # Reconstruct the reference answer expected by the V2 pipeline.
    reference_answer = ReferenceAnswer(
        answer=memory.answer,
        claims=claims,
        evidence_refs=memory.evidence_refs,
        confidence=memory.validation_confidence,
    )

    return {
        "reference_answer": reference_answer,
        "reference_validation": reference_validation,
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
        evaluation_input = state["evaluation_input"]

        question = (
            evaluation_input["question"]
            if isinstance(evaluation_input, dict)
            else evaluation_input.question
        )

        stored_memory = memory_store.save(
            question=question,
            reference_answer=reference_answer,
            validation=validation,
            evidence=evidence,
        )

    return {
        "memory_write_decision": decision,
        "memory_stored": stored_memory is not None,
    }