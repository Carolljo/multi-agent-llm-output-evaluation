from fastapi.testclient import TestClient

from src.api.app import app
from src.v2.models.schemas import (
    Adjudication,
    Disagreement,
    EvaluationInput,
    EvaluatorResult,
    FinalVerdict,
)


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "version": "2.0.0",
    }


def test_evaluate_rejects_empty_question():
    response = client.post(
        "/evaluate",
        json={
            "question": "",
            "response": "A valid response.",
        },
    )

    assert response.status_code == 422


def test_evaluate_rejects_empty_response():
    response = client.post(
        "/evaluate",
        json={
            "question": "What is RAG?",
            "response": "",
        },
    )

    assert response.status_code == 422


def test_evaluate_requires_question_and_response():
    response = client.post(
        "/evaluate",
        json={},
    )

    assert response.status_code == 422


def test_evaluate_returns_v2_result(monkeypatch):
    class FakeWorkflow:
        def invoke(self, state):
            assert isinstance(
                state["evaluation_input"],
                EvaluationInput,
            )

            assert (
                state["evaluation_input"].question
                == "What is RAG?"
            )

            assert (
                state["evaluation_input"].response
                == "RAG retrieves evidence."
            )

            evaluation = EvaluatorResult(
                evaluator="accuracy",
                score=1.0,
                confidence=1.0,
                claims_evaluated=1,
                errors=[],
                reasoning="Supported.",
                evidence_refs=[],
            )

            disagreement = Disagreement(
                detected=False,
                evaluators=[],
                score_range=0.0,
                reason="No disagreement.",
                requires_adjudication=False,
            )

            final_verdict = FinalVerdict(
                verdict="accept",
                score=1.0,
                confidence=1.0,
                evaluator_results=[evaluation],
                disagreement=disagreement,
                adjudication=None,
                key_errors=[],
                evidence_refs=[],
            )

            return {
                "final_verdict": final_verdict,
                "evidence": [],
                "reference_answer": None,
                "reference_validation": None,
                "claims": [],
                "evaluations": [evaluation],
                "disagreement": disagreement,
                "adjudication": None,
            }

    monkeypatch.setattr(
        "src.api.app.workflow",
        FakeWorkflow(),
    )

    response = client.post(
        "/evaluate",
        json={
            "question": "What is RAG?",
            "response": "RAG retrieves evidence.",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["final_verdict"]["verdict"] == "accept"
    assert data["final_verdict"]["score"] == 1.0
    assert data["final_verdict"]["confidence"] == 1.0
    assert len(data["evaluations"]) == 1
    assert data["evaluations"][0]["evaluator"] == "accuracy"


def test_evaluate_handles_workflow_failure(monkeypatch):
    class FailingWorkflow:
        def invoke(self, state):
            raise RuntimeError("Simulated workflow failure")

    monkeypatch.setattr(
        "src.api.app.workflow",
        FailingWorkflow(),
    )

    response = client.post(
        "/evaluate",
        json={
            "question": "What is RAG?",
            "response": "RAG retrieves evidence.",
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Evaluation failed."
    }
def test_evaluate_accepts_question_without_response(monkeypatch):
    class FakeWorkflow:
        def invoke(self, state):
            evaluation_input = state["evaluation_input"]

            assert evaluation_input.question == "What is RAG?"
            assert evaluation_input.response is None

            evaluation = EvaluatorResult(
                evaluator="accuracy",
                score=0.9,
                confidence=0.9,
                claims_evaluated=1,
                errors=[],
                reasoning="Evaluation completed.",
                evidence_refs=[],
            )

            disagreement = Disagreement(
                detected=False,
                evaluators=[],
                score_range=0.0,
                reason="No disagreement.",
                requires_adjudication=False,
            )

            final_verdict = FinalVerdict(
                verdict="accept",
                score=0.9,
                confidence=0.9,
                evaluator_results=[evaluation],
                disagreement=disagreement,
                adjudication=None,
                key_errors=[],
                evidence_refs=[],
            )

            return {
                "final_verdict": final_verdict,
                "evidence": [],
                "reference_answer": None,
                "reference_validation": None,
                "claims": [],
                "evaluations": [evaluation],
                "disagreement": disagreement,
                "adjudication": None,
            }

    monkeypatch.setattr(
        "src.api.app.workflow",
        FakeWorkflow(),
    )

    response = client.post(
        "/evaluate",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 200
    assert response.json()["final_verdict"]["verdict"] == "accept"