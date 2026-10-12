"""Deterministic reliability gate for final evaluation verdicts."""

from dataclasses import dataclass

from src.v2.models.schemas import (
    EvaluatorResult,
    FinalVerdict,
    ReferenceValidation,
)


MIN_REFERENCE_SCORE = 0.80
MIN_REFERENCE_CONFIDENCE = 0.80
MIN_AGGREGATED_CONFIDENCE = 0.85
MIN_GROUNDING_SCORE = 0.80


@dataclass(frozen=True)
class ReliabilityGateResult:
    """Result of applying deterministic reliability rules."""

    allowed: bool
    reasons: tuple[str, ...]


def check_reliability(
    *,
    reference_validation: ReferenceValidation | None,
    evaluations: list[EvaluatorResult],
    evidence_count: int,
    aggregated_confidence: float,
    unresolved_disagreement: bool,
) -> ReliabilityGateResult:
    """Check whether an evaluation satisfies reliability requirements."""

    reasons: list[str] = []

    # The reference answer must be independently validated.
    if reference_validation is None:
        reasons.append("Reference validation is missing.")
    else:
        if reference_validation.status != "valid":
            reasons.append(
                f"Reference validation status is "
                f"'{reference_validation.status}', not 'valid'."
            )

        if reference_validation.score < MIN_REFERENCE_SCORE:
            reasons.append(
                f"Reference validation score is below "
                f"{MIN_REFERENCE_SCORE:.2f}."
            )

        if (
            reference_validation.confidence
            < MIN_REFERENCE_CONFIDENCE
        ):
            reasons.append(
                f"Reference validation confidence is below "
                f"{MIN_REFERENCE_CONFIDENCE:.2f}."
            )

    # A verdict must have supporting evidence.
    if evidence_count <= 0:
        reasons.append("No supporting evidence is available.")

    # Overall evaluator confidence must meet the minimum.
    if aggregated_confidence < MIN_AGGREGATED_CONFIDENCE:
        reasons.append(
            f"Aggregated confidence is below "
            f"{MIN_AGGREGATED_CONFIDENCE:.2f}."
        )

    # The evidence-grounding evaluator must support the answer.
    grounding_results = [
        result
        for result in evaluations
        if result.evaluator == "evidence_grounding"
    ]

    if not grounding_results:
        reasons.append("Evidence-grounding evaluation is missing.")
    elif grounding_results[0].score < MIN_GROUNDING_SCORE:
        reasons.append(
            f"Evidence-grounding score is below "
            f"{MIN_GROUNDING_SCORE:.2f}."
        )

    # Unresolved disagreement must never receive unconditional acceptance.
    if unresolved_disagreement:
        reasons.append("Evaluator disagreement remains unresolved.")

    return ReliabilityGateResult(
        allowed=not reasons,
        reasons=tuple(reasons),
    )


def apply_reliability_gate(
    *,
    verdict: FinalVerdict,
    reference_validation: ReferenceValidation | None,
    evaluations: list[EvaluatorResult],
    evidence_count: int,
    unresolved_disagreement: bool,
) -> FinalVerdict:
    """Prevent an unreliable evaluation from receiving an accept verdict.

    The gate never upgrades a verdict. It only downgrades 'accept'
    when one or more reliability requirements are not satisfied.
    """

    # The gate only restricts unconditional acceptance.
    if verdict.verdict != "accept":
        return verdict

    result = check_reliability(
        reference_validation=reference_validation,
        evaluations=evaluations,
        evidence_count=evidence_count,
        aggregated_confidence=verdict.confidence,
        unresolved_disagreement=unresolved_disagreement,
    )

    if result.allowed:
        return verdict

    # Downgrade instead of silently accepting an unreliable answer.
    new_verdict = (
        "partial" if verdict.score >= 0.50 else "reject"
    )

    gate_errors = [
        f"Reliability Gate: {reason}"
        for reason in result.reasons
    ]

    return verdict.model_copy(
        update={
            "verdict": new_verdict,
            "key_errors": [
                *verdict.key_errors,
                *gate_errors,
            ],
        }
    )