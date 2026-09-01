from src.agents.accuracy import AccuracyEvaluator
from src.agents.adjudicator import Adjudicator
from src.agents.completeness import CompletenessEvaluator
from src.agents.logic import LogicEvaluator
from src.aggregation.aggregator import EvaluatorAggregator
from src.graph.state import EvaluationState
from src.models.final_result import FinalEvaluationResult

def create_accuracy_node(
    evaluator: AccuracyEvaluator,
):
    def accuracy_node(state: EvaluationState) -> dict:
        result = evaluator.evaluate(state["evaluation_input"])

        return {
            "accuracy": result,
        }

    return accuracy_node


def create_logic_node(
    evaluator: LogicEvaluator,
):
    def logic_node(state: EvaluationState) -> dict:
        result = evaluator.evaluate(state["evaluation_input"])

        return {
            "logic": result,
        }

    return logic_node


def create_completeness_node(
    evaluator: CompletenessEvaluator,
):
    def completeness_node(state: EvaluationState) -> dict:
        result = evaluator.evaluate(state["evaluation_input"])

        return {
            "completeness": result,
        }

    return completeness_node


def create_aggregation_node(
    aggregator: EvaluatorAggregator,
):
    def aggregation_node(state: EvaluationState) -> dict:
        evaluations = [
            state["accuracy"],
            state["logic"],
            state["completeness"],
        ]

        if any(evaluation is None for evaluation in evaluations):
            raise ValueError(
                "All evaluator results are required before aggregation."
            )

        evaluation_results = [
            evaluation
            for evaluation in evaluations
            if evaluation is not None
        ]

        summary = aggregator.aggregate(evaluation_results)

        return {
            "evaluations": evaluation_results,
            "evaluation_summary": summary,
            "disagreement": summary.disagreement,
        }

    return aggregation_node


def route_after_aggregation(state: EvaluationState) -> str:
    summary = state["evaluation_summary"]

    if summary is None:
        raise ValueError(
            "Evaluation summary is required before routing."
        )

    if summary.needs_adjudication:
        return "adjudicate"

    return "finalize"


def create_adjudication_node(
    adjudicator: Adjudicator,
):
    def adjudication_node(state: EvaluationState) -> dict:
        evaluation_input = state["evaluation_input"]
        evaluations = state["evaluations"]
        disagreement = state["disagreement"]

        if disagreement is None:
            raise ValueError(
                "Disagreement result is required before adjudication."
            )

        result = adjudicator.adjudicate(
            evaluation_input,
            evaluations,
            disagreement,
        )

        return {
            "adjudication": result,
        }

    return adjudication_node

def finalize_result(state: EvaluationState) -> dict:
    adjudication = state["adjudication"]
    summary = state["evaluation_summary"]

    if adjudication is not None:
        return {
            "final_result": FinalEvaluationResult(
                verdict=adjudication.final_verdict,
                score=adjudication.final_score,
                confidence=adjudication.confidence,
                reasoning=adjudication.reasoning,
                issues=adjudication.issues,
            )
        }

    if summary is None:
        raise ValueError(
            "Evaluation summary is required before finalization."
        )

    evaluations = summary.evaluations

    scores = {
        evaluation.criterion: evaluation.score
        for evaluation in evaluations
    }

    accuracy_score = scores.get("accuracy", 0)
    logic_score = scores.get("logic", 0)
    completeness_score = scores.get("completeness", 0)

    if (
        accuracy_score >= 9
        and logic_score >= 9
        and completeness_score >= 9
    ):
        verdict = "Correct"

    elif (
        accuracy_score >= 8
        and logic_score >= 8
        and completeness_score >= 8
    ):
        verdict = "Mostly Correct"

    elif (
        accuracy_score < 5
        or logic_score < 5
    ):
        verdict = "Incorrect"

    else:
        verdict = "Partially Correct"

    return {
        "final_result": FinalEvaluationResult(
            verdict=verdict,
            score=summary.overall_score,
            confidence=summary.overall_confidence,
            reasoning=summary.summary,
            issues=summary.key_issues,
        )
    }