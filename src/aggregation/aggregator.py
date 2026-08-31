from src.models.evaluation import EvaluationResult
from src.models.summary import EvaluationSummary
from src.aggregation.disagreement import DisagreementDetector

class EvaluatorAggregator:
    """Combines independent evaluator results into an overall evaluation."""

    def __init__(self, disagreement_detector: DisagreementDetector):
        self.disagreement_detector = disagreement_detector
    def aggregate(
        self,
        evaluations: list[EvaluationResult]
    ) -> EvaluationSummary:
        """Aggregate accuracy, logic, and completeness evaluations."""

        accuracy = next(
            evaluation
            for evaluation in evaluations
            if evaluation.criterion == "accuracy"
        )

        logic = next(
            evaluation
            for evaluation in evaluations
            if evaluation.criterion == "logic"
        )

        completeness = next(
            evaluation
            for evaluation in evaluations
            if evaluation.criterion == "completeness"
        )

        weighted_score = (
            accuracy.score * 0.40
            + logic.score * 0.30
            + completeness.score * 0.30
        )

        weakest_score = min(
            accuracy.score,
            logic.score,
            completeness.score
        )

        if weakest_score <= 3:
            overall_score = min(weighted_score, 5.0)

        elif weakest_score <= 5:
            overall_score = min(weighted_score, 7.0)

        else:
            overall_score = weighted_score

        overall_confidence = (
            accuracy.confidence * 0.40
            + logic.confidence * 0.30
            + completeness.confidence * 0.30
        )
        disagreement = self.disagreement_detector.detect(evaluations)

        needs_adjudication = disagreement.has_disagreement
        key_issues = (
            accuracy.issues
            + logic.issues
            + completeness.issues
        )

        summary = self._build_summary(
            accuracy,
            logic,
            completeness
        )

        return EvaluationSummary(
            evaluations=evaluations,
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            summary=summary,
            key_issues=key_issues,
            needs_adjudication=needs_adjudication,
        )
    def _build_summary(
        self,
        accuracy: EvaluationResult,
        logic: EvaluationResult,
        completeness: EvaluationResult,
    ) -> str:
        """Build a human-readable summary from the three evaluations."""

        if (
            accuracy.score >= 8
            and logic.score >= 8
            and completeness.score >= 8
        ):
            return (
                "The response is factually accurate, "
                "logically sound, and complete."
            )

        if (
            accuracy.score >= 8
            and logic.score >= 8
            and completeness.score < 8
        ):
            return (
                "The response is factually accurate and logically sound "
                "but incomplete."
            )

        if (
            accuracy.score >= 8
            and logic.score < 8
            and completeness.score >= 8
        ):
            return (
                "The response is factually accurate and complete "
                "but has logical weaknesses."
            )

        if (
            accuracy.score < 8
            and logic.score >= 8
            and completeness.score >= 8
        ):
            return (
                "The response is logically sound and complete "
                "but contains factual weaknesses."
            )

        return (
            "The response has weaknesses in accuracy, logic, "
            "completeness, or a combination of these dimensions."
        )