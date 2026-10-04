from dataclasses import dataclass

from src.v2.memory.schemas import AnswerMemoryMatch


@dataclass(frozen=True)
class MemoryDecision:
    """Deterministic decision about whether verified answer memory is reusable."""

    decision: str
    similarity: float
    validation_score: float
    validation_confidence: float
    decision_score: float
    reason: str


class MemoryDecisionPolicy:
    """
    Decide whether a retrieved verified answer is strong enough
    to be reused as candidate knowledge.

    The policy is intentionally deterministic and does not call an LLM.
    """

    def __init__(
        self,
        min_similarity: float = 0.85,
        min_validation_score: float = 0.80,
        min_validation_confidence: float = 0.80,
        min_decision_score: float = 0.85,
    ):
        self.min_similarity = min_similarity
        self.min_validation_score = min_validation_score
        self.min_validation_confidence = min_validation_confidence
        self.min_decision_score = min_decision_score

    def evaluate(
        self,
        match: AnswerMemoryMatch,
    ) -> MemoryDecision:
        """Evaluate a single memory candidate."""

        similarity = match.similarity
        validation_score = match.memory.validation_score
        validation_confidence = match.memory.validation_confidence

        decision_score = (
            0.50 * similarity
            + 0.30 * validation_score
            + 0.20 * validation_confidence
        )

        if similarity < self.min_similarity:
            return MemoryDecision(
                decision="fresh_retrieval",
                similarity=similarity,
                validation_score=validation_score,
                validation_confidence=validation_confidence,
                decision_score=decision_score,
                reason="Memory similarity is below the required threshold.",
            )

        if validation_score < self.min_validation_score:
            return MemoryDecision(
                decision="fresh_retrieval",
                similarity=similarity,
                validation_score=validation_score,
                validation_confidence=validation_confidence,
                decision_score=decision_score,
                reason="Previous memory validation score is below the required threshold.",
            )

        if validation_confidence < self.min_validation_confidence:
            return MemoryDecision(
                decision="fresh_retrieval",
                similarity=similarity,
                validation_score=validation_score,
                validation_confidence=validation_confidence,
                decision_score=decision_score,
                reason="Previous memory validation confidence is below the required threshold.",
            )

        if decision_score < self.min_decision_score:
            return MemoryDecision(
                decision="fresh_retrieval",
                similarity=similarity,
                validation_score=validation_score,
                validation_confidence=validation_confidence,
                decision_score=decision_score,
                reason="Combined memory decision score is below the required threshold.",
            )

        return MemoryDecision(
            decision="use_memory",
            similarity=similarity,
            validation_score=validation_score,
            validation_confidence=validation_confidence,
            decision_score=decision_score,
            reason=(
                "Memory candidate passed similarity, validation, "
                "confidence, and combined decision thresholds."
            ),
        )