from typing import Any

from src.v2.memory.decision import MemoryDecision, MemoryDecisionPolicy
from src.v2.memory.retriever import AnswerMemoryRetriever


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

    question = state["evaluation_input"].question

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