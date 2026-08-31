from src.graph.graph import create_evaluation_graph
from src.models.inputs import EvaluationInput


def test_graph_executes_without_adjudication():
    graph = create_evaluation_graph()

    evaluation_input = EvaluationInput(
        question="What is the capital of France?",
        reference_answer="The capital of France is Paris.",
        response="The capital of France is Paris.",
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
        }
    )

    print(result)

    assert result["accuracy"] is not None
    assert result["logic"] is not None
    assert result["completeness"] is not None
    assert result["evaluation_summary"] is not None
    assert result["disagreement"] is not None