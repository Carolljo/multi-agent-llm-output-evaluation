
from unittest.mock import Mock

from fastapi.testclient import TestClient

from src.main import app
from src.models.adjudication import AdjudicationResult
from src.models.disagreement import DisagreementResult
from src.models.evaluation import EvaluationResult
from src.models.final_result import FinalEvaluationResult


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_evaluate_returns_final_result(monkeypatch):
    mock_graph = Mock()

    evaluations = [
        EvaluationResult(
            criterion="accuracy",
            score=10,
            confidence=0.98,
            reasoning="The answer is factually correct.",
            issues=[],
        ),
        EvaluationResult(
            criterion="logic",
            score=10,
            confidence=0.98,
            reasoning="The reasoning is logically sound.",
            issues=[],
        ),
        EvaluationResult(
            criterion="completeness",
            score=10,
            confidence=0.98,
            reasoning="The response fully answers the question.",
            issues=[],
        ),
    ]

    disagreement = DisagreementResult(
        has_disagreement=False,
        severity="none",
        score_spread=0,
        reasons=[],
    )

    mock_graph.invoke.return_value = {
        "evaluations": evaluations,
        "disagreement": disagreement,
        "adjudication": None,
        "final_result": FinalEvaluationResult(
            verdict="Correct",
            score=10,
            confidence=0.98,
            reasoning="The response is accurate, logically sound, and complete.",
            issues=[],
        ),
    }

    monkeypatch.setattr(
        "src.main.evaluation_graph",
        mock_graph,
    )

    response = client.post(
        "/evaluate",
        json={
            "question": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "reference_answer": "The capital of France is Paris.",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["final_result"]["verdict"] == "Correct"
    assert data["final_result"]["score"] == 10
    assert data["final_result"]["confidence"] == 0.98
    assert data["final_result"]["issues"] == []

    assert len(data["evaluations"]) == 3

    assert data["evaluations"][0]["criterion"] == "accuracy"
    assert data["evaluations"][0]["score"] == 10

    assert data["evaluations"][1]["criterion"] == "logic"
    assert data["evaluations"][1]["score"] == 10

    assert data["evaluations"][2]["criterion"] == "completeness"
    assert data["evaluations"][2]["score"] == 10

    assert data["disagreement"]["has_disagreement"] is False
    assert data["disagreement"]["severity"] == "none"
    assert data["disagreement"]["score_spread"] == 0
    assert data["disagreement"]["reasons"] == []

    assert data["adjudication"] is None

    mock_graph.invoke.assert_called_once()


def test_evaluate_returns_adjudication_result(monkeypatch):
    mock_graph = Mock()

    evaluations = [
        EvaluationResult(
            criterion="accuracy",
            score=8,
            confidence=0.9,
            reasoning="The response is mostly accurate.",
            issues=["The conclusion is not fully supported."],
        ),
        EvaluationResult(
            criterion="logic",
            score=3,
            confidence=0.95,
            reasoning="The response contains an invalid inference.",
            issues=["The conclusion does not logically follow."],
        ),
        EvaluationResult(
            criterion="completeness",
            score=9,
            confidence=0.9,
            reasoning="The response addresses the question.",
            issues=[],
        ),
    ]

    disagreement = DisagreementResult(
        has_disagreement=True,
        severity="high",
        score_spread=6,
        reasons=["Evaluator scores differ significantly."],
    )

    adjudication = AdjudicationResult(
        final_verdict="Partially Correct",
        final_score=6,
        confidence=0.92,
        reasoning="The candidate response contains a significant logical error.",
        issues=["The final conclusion is not supported by the premise."],
    )

    mock_graph.invoke.return_value = {
        "evaluations": evaluations,
        "disagreement": disagreement,
        "adjudication": adjudication,
        "final_result": FinalEvaluationResult(
            verdict="Partially Correct",
            score=6,
            confidence=0.92,
            reasoning="The candidate response contains a significant logical error.",
            issues=["The final conclusion is not supported by the premise."],
        ),
    }

    monkeypatch.setattr(
        "src.main.evaluation_graph",
        mock_graph,
    )

    response = client.post(
        "/evaluate",
        json={
            "question": (
                "If all employees who complete security training "
                "receive a certificate, and Rahul received a certificate, "
                "what can we conclude about Rahul?"
            ),
            "response": (
                "Rahul received a certificate. Therefore, Rahul "
                "completed the training."
            ),
            "reference_answer": (
                "We cannot conclude that Rahul completed the training "
                "solely because he received a certificate."
            ),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["disagreement"]["has_disagreement"] is True
    assert data["disagreement"]["severity"] == "high"
    assert data["disagreement"]["score_spread"] == 6

    assert data["adjudication"] is not None
    assert data["adjudication"]["final_verdict"] == "Partially Correct"
    assert data["adjudication"]["final_score"] == 6
    assert data["adjudication"]["confidence"] == 0.92

    assert data["final_result"]["verdict"] == "Partially Correct"
    assert data["final_result"]["score"] == 6

    mock_graph.invoke.assert_called_once()


def test_evaluate_rejects_empty_question():
    response = client.post(
        "/evaluate",
        json={
            "question": "",
            "response": "Paris.",
            "reference_answer": "Paris.",
        },
    )

    assert response.status_code == 422


def test_evaluate_returns_500_when_graph_fails(monkeypatch):
    mock_graph = Mock()
    mock_graph.invoke.side_effect = RuntimeError("Graph failure")

    monkeypatch.setattr(
        "src.main.evaluation_graph",
        mock_graph,
    )

    response = client.post(
        "/evaluate",
        json={
            "question": "What is the capital of France?",
            "response": "Paris.",
            "reference_answer": "The capital of France is Paris.",
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Evaluation pipeline failed."
    }

