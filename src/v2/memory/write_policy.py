from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryWriteDecision:
    """Decision about whether a verified answer should be persisted."""

    decision: str
    reason: str
    validation_score: float
    validation_confidence: float
    aggregated_confidence: float
    evidence_count: int
    unresolved_disagreement: bool


class MemoryWritePolicy:
    """
    Conservative policy for deciding whether a final evaluation
    is strong enough to be persisted as verified answer memory.

    The policy is deterministic and does not call an LLM.
    """

    def __init__(
        self,
        min_validation_score: float = 0.80,
        min_validation_confidence: float = 0.80,
        min_aggregated_confidence: float = 0.85,
        require_accept_verdict: bool = True,
        require_evidence: bool = True,
        reject_unresolved_disagreement: bool = True,
    ):
        self.min_validation_score = min_validation_score
        self.min_validation_confidence = min_validation_confidence
        self.min_aggregated_confidence = min_aggregated_confidence
        self.require_accept_verdict = require_accept_verdict
        self.require_evidence = require_evidence
        self.reject_unresolved_disagreement = reject_unresolved_disagreement

    def evaluate(
        self,
        *,
        verdict: str,
        validation_score: float,
        validation_confidence: float,
        aggregated_confidence: float,
        evidence_count: int,
        unresolved_disagreement: bool,
    ) -> MemoryWriteDecision:
        """Evaluate whether the final answer should be stored."""

        verdict = verdict.strip().upper()

        if self.require_accept_verdict and verdict != "ACCEPT":
            return MemoryWriteDecision(
                decision="do_not_store",
                reason="Only ACCEPT verdicts are eligible for persistent memory.",
                validation_score=validation_score,
                validation_confidence=validation_confidence,
                aggregated_confidence=aggregated_confidence,
                evidence_count=evidence_count,
                unresolved_disagreement=unresolved_disagreement,
            )

        if validation_score < self.min_validation_score:
            return MemoryWriteDecision(
                decision="do_not_store",
                reason="Reference validation score is below the required threshold.",
                validation_score=validation_score,
                validation_confidence=validation_confidence,
                aggregated_confidence=aggregated_confidence,
                evidence_count=evidence_count,
                unresolved_disagreement=unresolved_disagreement,
            )

        if validation_confidence < self.min_validation_confidence:
            return MemoryWriteDecision(
                decision="do_not_store",
                reason="Reference validation confidence is below the required threshold.",
                validation_score=validation_score,
                validation_confidence=validation_confidence,
                aggregated_confidence=aggregated_confidence,
                evidence_count=evidence_count,
                unresolved_disagreement=unresolved_disagreement,
            )

        if aggregated_confidence < self.min_aggregated_confidence:
            return MemoryWriteDecision(
                decision="do_not_store",
                reason="Aggregated evaluation confidence is below the required threshold.",
                validation_score=validation_score,
                validation_confidence=validation_confidence,
                aggregated_confidence=aggregated_confidence,
                evidence_count=evidence_count,
                unresolved_disagreement=unresolved_disagreement,
            )

        if self.require_evidence and evidence_count <= 0:
            return MemoryWriteDecision(
                decision="do_not_store",
                reason="The final answer has no supporting evidence.",
                validation_score=validation_score,
                validation_confidence=validation_confidence,
                aggregated_confidence=aggregated_confidence,
                evidence_count=evidence_count,
                unresolved_disagreement=unresolved_disagreement,
            )

        if (
            self.reject_unresolved_disagreement
            and unresolved_disagreement
        ):
            return MemoryWriteDecision(
                decision="do_not_store",
                reason="The evaluation contains unresolved disagreement.",
                validation_score=validation_score,
                validation_confidence=validation_confidence,
                aggregated_confidence=aggregated_confidence,
                evidence_count=evidence_count,
                unresolved_disagreement=unresolved_disagreement,
            )

        return MemoryWriteDecision(
            decision="store",
            reason=(
                "Answer passed verdict, validation, confidence, "
                "evidence, and disagreement requirements."
            ),
            validation_score=validation_score,
            validation_confidence=validation_confidence,
            aggregated_confidence=aggregated_confidence,
            evidence_count=evidence_count,
            unresolved_disagreement=unresolved_disagreement,
        )