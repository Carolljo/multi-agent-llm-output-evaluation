from src.graph.graph import create_evaluation_graph
from src.models.adjudication import AdjudicationResult
from src.models.disagreement import DisagreementResult
from src.models.evaluation import EvaluationResult
from src.models.inputs import EvaluationInput
from src.models.summary import EvaluationSummary


class MockEvaluator:
    def __init__(self, criterion, score):
        self.criterion = criterion
        self.score = score

    def evaluate(self, evaluation_input):
        return EvaluationResult(
            criterion=self.criterion,
            score=self.score,
            reasoning=f"Mock {self.criterion} evaluation.",
            issues=[] if self.score >= 8 else [f"{self.criterion} issue"],
            confidence=0.9,
        )


class MockAggregator:
    def aggregate(self, evaluations):
        disagreement = DisagreementResult(
            has_disagreement=True,
            severity="high",
            score_spread=6,
            reasons=["Large evaluator disagreement."],
        )

        return EvaluationSummary(
            evaluations=evaluations,
            overall_score=7.0,
            overall_confidence=0.85,
            summary="Mock aggregated result.",
            key_issues=["Evaluator disagreement."],
            needs_adjudication=True,
            disagreement=disagreement,
        )


class MockAdjudicator:
    def adjudicate(
        self,
        evaluation_input,
        evaluations,
        disagreement,
    ):
        return AdjudicationResult(
            final_verdict="Partially Correct",
            final_score=6,
            confidence=0.92,
            reasoning="The disagreement requires final adjudication.",
            issues=["Significant evaluator disagreement."],
        )


def test_graph_executes_adjudication_branch():
    graph = create_evaluation_graph(
        accuracy_evaluator=MockEvaluator("accuracy", 9),
        logic_evaluator=MockEvaluator("logic", 3),
        completeness_evaluator=MockEvaluator("completeness", 9),
        aggregator=MockAggregator(),
        adjudicator=MockAdjudicator(),
    )

    evaluation_input = EvaluationInput(
        question="What is the capital of France?",
        reference_answer="The capital of France is Paris.",
        response="The capital of France is Lyon.",
    )

    result = graph.invoke(
        {
            "evaluation_input": evaluation_input,
            "accuracy": None,
            "logic": None,
            "completeness": None,
            "evaluations": [],
            "evaluation_summary": None,
            "disagreement": None,
            "adjudication": None,
            "final_result": None,
        }
    )

    assert result["disagreement"].has_disagreement is True
    assert result["evaluation_summary"].needs_adjudication is True

    assert result["adjudication"] is not None
    assert result["adjudication"].final_verdict == "Partially Correct"

    assert result["final_result"] is not None
    assert result["final_result"].score == 6