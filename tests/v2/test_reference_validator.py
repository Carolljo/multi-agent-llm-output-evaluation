import json
from unittest.mock import Mock

import pytest

from src.v2.models.schemas import (
    Claim,
    Evidence,
    ReferenceAnswer,
)
from src.v2.reference.validator import ReferenceAnswerValidator


def make_evidence():
    return [
        Evidence(
            evidence_id="E1",
            document_id="doc1",
            chunk_id="chunk1",
            content=(
                "RAG retrieves external information before "
                "generating a response."
            ),
            source="rag.txt",
        ),
        Evidence(
            evidence_id="E2",
            document_id="doc1",
            chunk_id="chunk2",
            content=(
                "Retrieved information provides additional "
                "context for the language model."
            ),
            source="rag.txt",
        ),
    ]


def make_reference():
    return ReferenceAnswer(
        answer=(
            "RAG retrieves external information and uses it "
            "as additional context."
        ),
        claims=[
            Claim(
                claim_id="C1",
                text=(
                    "RAG retrieves external information before "
                    "generating a response."
                ),
                evidence_refs=["E1"],
                importance=0.9,
            )
        ],
        evidence_refs=["E1", "E2"],
        confidence=0.95,
    )


def test_validates_reference_answer():
    llm_client = Mock()

    llm_client.generate.return_value = json.dumps(
        {
            "status": "valid",
            "score": 0.95,
            "confidence": 0.92,
            "supported_claims": ["C1"],
            "unsupported_claims": [],
            "missing_information": [],
            "reasoning": (
                "The reference claim is directly supported "
                "by the retrieved evidence."
            ),
        }
    )

    validator = ReferenceAnswerValidator(llm_client)

    result = validator.validate(
        question="How does RAG work?",
        reference_answer=make_reference(),
        evidence=make_evidence(),
    )

    assert result.status == "valid"
    assert result.score == 0.95
    assert result.confidence == 0.92
    assert result.supported_claims == ["C1"]


def test_rejects_empty_question():
    validator = ReferenceAnswerValidator(Mock())

    with pytest.raises(ValueError, match="Question cannot be empty"):
        validator.validate(
            question=" ",
            reference_answer=make_reference(),
            evidence=make_evidence(),
        )


def test_rejects_empty_evidence():
    validator = ReferenceAnswerValidator(Mock())

    with pytest.raises(
        ValueError,
        match="At least one evidence item is required",
    ):
        validator.validate(
            question="How does RAG work?",
            reference_answer=make_reference(),
            evidence=[],
        )


def test_rejects_invalid_json():
    llm_client = Mock()
    llm_client.generate.return_value = "invalid json"

    validator = ReferenceAnswerValidator(llm_client)

    with pytest.raises(
        ValueError,
        match="LLM returned invalid JSON",
    ):
        validator.validate(
            question="How does RAG work?",
            reference_answer=make_reference(),
            evidence=make_evidence(),
        )


def test_prompt_contains_reference_and_evidence():
    llm_client = Mock()

    llm_client.generate.return_value = json.dumps(
        {
            "status": "valid",
            "score": 0.9,
            "confidence": 0.9,
            "supported_claims": ["C1"],
            "unsupported_claims": [],
            "missing_information": [],
            "reasoning": "Supported.",
        }
    )

    validator = ReferenceAnswerValidator(llm_client)

    validator.validate(
        question="How does RAG work?",
        reference_answer=make_reference(),
        evidence=make_evidence(),
    )

    prompt = llm_client.generate.call_args.kwargs["prompt"]

    assert "How does RAG work?" in prompt
    assert "RAG retrieves external information" in prompt
    assert "[E1]" in prompt
    assert "[E2]" in prompt
    assert "[C1]" in prompt