import json
from unittest.mock import Mock

import pytest

from src.v2.answer.generator import CandidateAnswerGenerator


def test_generates_candidate_answer():
    """A valid LLM response should produce a candidate answer."""
    llm_client = Mock()

    llm_client.generate.return_value = json.dumps(
        {
            "answer": (
                "Machine learning enables computers to learn "
                "patterns from data."
            )
        }
    )

    generator = CandidateAnswerGenerator(llm_client)

    result = generator.generate(
        question="What is machine learning?"
    )

    assert result == (
        "Machine learning enables computers to learn "
        "patterns from data."
    )

    llm_client.generate.assert_called_once()


def test_rejects_empty_question():
    """An empty question must be rejected without calling the LLM."""
    llm_client = Mock()
    generator = CandidateAnswerGenerator(llm_client)

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        generator.generate(question="   ")

    llm_client.generate.assert_not_called()


def test_rejects_invalid_json():
    """Malformed JSON must produce a clear validation error."""
    llm_client = Mock()
    llm_client.generate.return_value = "not valid json"

    generator = CandidateAnswerGenerator(llm_client)

    with pytest.raises(
        ValueError,
        match="LLM returned invalid JSON",
    ):
        generator.generate(
            question="What is machine learning?"
        )


def test_rejects_empty_answer():
    """An empty answer must not be accepted."""
    llm_client = Mock()

    llm_client.generate.return_value = json.dumps(
        {"answer": "   "}
    )

    generator = CandidateAnswerGenerator(llm_client)

    with pytest.raises(
        ValueError,
        match="empty or invalid candidate answer",
    ):
        generator.generate(
            question="What is machine learning?"
        )


def test_rejects_missing_answer_field():
    """The response must contain the answer field."""
    llm_client = Mock()

    llm_client.generate.return_value = json.dumps(
        {"result": "Some response"}
    )

    generator = CandidateAnswerGenerator(llm_client)

    with pytest.raises(
        ValueError,
        match="empty or invalid candidate answer",
    ):
        generator.generate(
            question="What is machine learning?"
        )


def test_question_is_included_in_prompt():
    """The LLM prompt must contain the user's question."""
    llm_client = Mock()

    llm_client.generate.return_value = json.dumps(
        {"answer": "Machine learning learns patterns from data."}
    )

    generator = CandidateAnswerGenerator(llm_client)

    generator.generate(
        question="What is machine learning?"
    )

    llm_client.generate.assert_called_once()

    call = llm_client.generate.call_args
    prompt = call.kwargs["prompt"]

    assert "What is machine learning?" in prompt
    assert "Use your general knowledge" in prompt


def test_candidate_generation_does_not_require_evidence():
    """Candidate generation must work without RAG evidence."""
    llm_client = Mock()

    llm_client.generate.return_value = json.dumps(
        {"answer": "Python is a programming language."}
    )

    generator = CandidateAnswerGenerator(llm_client)

    result = generator.generate(
        question="What is Python?"
    )

    assert result == "Python is a programming language."

    prompt = llm_client.generate.call_args.kwargs["prompt"]

    assert "RETRIEVED EVIDENCE:" not in prompt