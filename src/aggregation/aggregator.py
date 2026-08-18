from src.models.evaluation import EvaluationResult
from src.models.summary import EvaluationSummary


class EvaluatorAggregator:

    def aggregate(
        self,
        evaluations: list[EvaluationResult]
    ) -> EvaluationSummary:

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

        weighted_score = (
            accuracy.score * 0.5
            + logic.score * 0.5
        )

        weakest_score = min(
            accuracy.score,
            logic.score
        )

        if weakest_score <= 3:
            overall_score = min(weighted_score, 5.0)

        elif weakest_score <= 5:
            overall_score = min(weighted_score, 7.0)

        else:
            overall_score = weighted_score

        overall_confidence = (
            accuracy.confidence * 0.5
            + logic.confidence * 0.5
        )

        key_issues = (
            accuracy.issues
            + logic.issues
        )

        summary = self._build_summary(
            accuracy,
            logic
        )

        return EvaluationSummary(
            evaluations=evaluations,
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            summary=summary,
            key_issues=key_issues,
            needs_adjudication=False,
        )

    def _build_summary(
        self,
        accuracy: EvaluationResult,
        logic: EvaluationResult,
    ) -> str:

        if accuracy.score >= 8 and logic.score >= 8:
            return "The response is both factually accurate and logically sound."

        if accuracy.score >= 8 and logic.score < 8:
            return "The response is factually strong but has logical weaknesses."

        if accuracy.score < 8 and logic.score >= 8:
            return "The response is logically sound but contains factual weaknesses."

        return "The response contains both factual and logical weaknesses."