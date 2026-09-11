from src.v2.models.schemas import Disagreement, EvaluatorResult


DISAGREEMENT_THRESHOLD = 0.30


def detect_disagreement(
    evaluations: list[EvaluatorResult],
) -> Disagreement:
    """Detect meaningful disagreement between evaluator scores."""

    if not evaluations:
        raise ValueError("At least one evaluator result is required")

    scores = [result.score for result in evaluations]

    score_range = max(scores) - min(scores)

    detected = score_range >= DISAGREEMENT_THRESHOLD

    if detected:
        highest_score = max(scores)
        lowest_score = min(scores)

        evaluators = [
            result.evaluator
            for result in evaluations
            if result.score == highest_score
            or result.score == lowest_score
        ]

        reason = (
            f"Evaluator scores differ by {score_range:.2f}, "
            f"from {lowest_score:.2f} to {highest_score:.2f}."
        )
    else:
        evaluators = []

        reason = (
            f"Evaluator scores are within the disagreement threshold "
            f"(range: {score_range:.2f})."
        )

    return Disagreement(
        detected=detected,
        evaluators=evaluators,
        score_range=score_range,
        reason=reason,
        requires_adjudication=detected,
    )