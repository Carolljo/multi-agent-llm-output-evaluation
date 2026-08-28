from src.models.evaluation import EvaluationResult
from src.models.disagreement import DisagreementResult


class DisagreementDetector:
    """Detects significant disagreement between evaluator scores."""

    def detect(
        self,
        evaluations: list[EvaluationResult]
    ) -> DisagreementResult:
        """Analyze evaluator scores and determine disagreement severity."""

        scores = [evaluation.score for evaluation in evaluations]

        score_spread = max(scores) - min(scores)

        if score_spread >= 5:
            severity = "high"
            has_disagreement = True

        elif score_spread >= 3:
            severity = "medium"
            has_disagreement = True

        else:
            severity = "none"
            has_disagreement = False

        reasons = []

        if has_disagreement:
            reasons.append(
                f"Evaluator scores differ by {score_spread} points."
            )

        return DisagreementResult(
            has_disagreement=has_disagreement,
            severity=severity,
            score_spread=score_spread,
            reasons=reasons,
        )