from dataclasses import dataclass

from src.v2.models.schemas import EvaluatorResult


REQUIRED_EVALUATORS = {
    "accuracy",
    "completeness",
    "logical_consistency",
    "evidence_grounding",
}


@dataclass
class AggregatedEvaluation:
    score: float
    confidence: float
    evaluations: list[EvaluatorResult]
    errors: list[str]
    evidence_refs: list[str]


def aggregate_evaluations(
    evaluations: list[EvaluatorResult],
) -> AggregatedEvaluation:
    """Aggregate the four independent evaluator results deterministically."""

    evaluator_names = {result.evaluator for result in evaluations}

    missing = REQUIRED_EVALUATORS - evaluator_names

    if missing:
        raise ValueError(
            f"Missing required evaluator result(s): "
            f"{', '.join(sorted(missing))}"
        )

    if len(evaluator_names) != len(evaluations):
        raise ValueError("Duplicate evaluator results are not allowed")

    score = sum(result.score for result in evaluations) / len(evaluations)

    confidence = (
        sum(result.confidence for result in evaluations)
        / len(evaluations)
    )

    errors = [
        f"{result.evaluator}: {error}"
        for result in evaluations
        for error in result.errors
    ]

    evidence_refs = list(
        dict.fromkeys(
            ref
            for result in evaluations
            for ref in result.evidence_refs
        )
    )

    return AggregatedEvaluation(
        score=score,
        confidence=confidence,
        evaluations=evaluations,
        errors=errors,
        evidence_refs=evidence_refs,
    )