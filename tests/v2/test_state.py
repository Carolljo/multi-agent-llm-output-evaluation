from src.v2.graph.state import EvaluationState
from src.v2.models.schemas import EvaluationInput


def test_evaluation_state_structure():
    state: EvaluationState = {
        "evaluation_input": EvaluationInput(
            question="What is RAG?",
            response="RAG retrieves external evidence before generating an answer.",
        ),
        "evidence": [],
        "reference_answer": None,
        "reference_validation": None,
        "claims": [],
        "accuracy": None,
        "completeness": None,
        "logic": None,
        "grounding": None,
        "evaluations": [],
        "disagreement": None,
        "adjudication": None,
        "final_verdict": None,
    }

    assert state["evaluation_input"].question == "What is RAG?"
    assert state["evidence"] == []
    assert state["reference_answer"] is None
    assert state["evaluations"] == []
    assert state["final_verdict"] is None