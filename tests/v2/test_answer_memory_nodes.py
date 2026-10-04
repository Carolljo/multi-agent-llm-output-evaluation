from src.v2.graph.nodes import route_after_answer_memory
from src.v2.models.schemas import EvaluationInput


def test_memory_route_hits_when_verified_answer_is_found():
    state = {
        "evaluation_input": EvaluationInput(
            question="What is machine learning?",
            response="It learns patterns from data.",
        ),
        "memory_hit": True,
    }

    assert route_after_answer_memory(state) == "memory_hit"


def test_memory_route_misses_when_no_verified_answer_is_found():
    state = {
        "evaluation_input": EvaluationInput(
            question="What is machine learning?",
            response="It learns patterns from data.",
        ),
        "memory_hit": False,
    }

    assert route_after_answer_memory(state) == "memory_miss"
