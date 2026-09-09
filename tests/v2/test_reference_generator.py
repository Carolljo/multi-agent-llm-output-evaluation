import json
from unittest.mock import Mock

import pytest

from src.v2.models.schemas import Evidence
from src.v2.reference.generator import ReferenceAnswerGenerator


def make_evidence():
    return [
        Evidence(
            evidence_id="E1",
            document_id="doc1",
            chunk_id="chunk1",
            content=(
                "Retrieval augmented generation retrieves "
                "external information before generating a response."
            ),
            source="rag.txt",
            relevance_score=0.95,
        ),
        Evidence(
            evidence_id="E2",
            document_id="doc1",
            chunk_id="chunk2",
            content=(
                "The retrieved information provides additional "
                "context for the language model."
            ),
            source="rag.txt",
            relevance_score=0.90,
        ),
    ]


def test_generates_reference_answer():
    llm_client = Mock()

    llm_client.generate.return_value = json.dumps(
        {
            "answer": (
                "RAG retrieves external information and provides "
                "it as additional context before generating a response."
            ),
            "claims": [
                {
                    "claim_id": "C1",
                    "text": (
                        "RAG retrieves external information before "
                        "generating a response."
                    ),
                    "evidence_refs": ["E1"],
                    "importance": 0.9,
                },
                {
                    "claim_id": "C2",
                    "text": (
                        "Retrieved information provides additional "
                        "context for the language model."
                    ),
                    "evidence_refs": ["E2"],
                    "importance": 0.8,
                },
            ],
            "evidence_refs": ["E1", "E2"],
            "confidence": 0.95,
        }
    )

    generator = ReferenceAnswerGenerator(llm_client)

    result = generator.generate(
        question="How does RAG work?",
        evidence=make_evidence(),
    )

    assert result.answer
    assert len(result.claims) == 2
    assert result.evidence_refs == ["E1", "E2"]
    assert result.confidence == 0.95

    llm_client.generate.assert_called_once()


def test_rejects_empty_question():
    llm_client = Mock()
    generator = ReferenceAnswerGenerator(llm_client)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        generator.generate(
            question="   ",
            evidence=make_evidence(),
        )


def test_rejects_empty_evidence():
    llm_client = Mock()
    generator = ReferenceAnswerGenerator(llm_client)

    with pytest.raises(
        ValueError,
        match="At least one evidence item is required",
    ):
        generator.generate(
            question="How does RAG work?",
            evidence=[],
        )


def test_rejects_invalid_json():
    llm_client = Mock()
    llm_client.generate.return_value = "not valid json"

    generator = ReferenceAnswerGenerator(llm_client)

    with pytest.raises(
        ValueError,
        match="LLM returned invalid JSON",
    ):
        generator.generate(
            question="How does RAG work?",
            evidence=make_evidence(),
        )


def test_evidence_is_included_in_prompt():
    llm_client = Mock()

    llm_client.generate.return_value = json.dumps(
        {
            "answer": "RAG uses external evidence.",
            "claims": [],
            "evidence_refs": ["E1"],
            "confidence": 0.9,
        }
    )

    generator = ReferenceAnswerGenerator(llm_client)

    generator.generate(
        question="How does RAG work?",
        evidence=make_evidence(),
    )

    call = llm_client.generate.call_args

    prompt = call.kwargs["prompt"]

    assert "How does RAG work?" in prompt
    assert "E1" in prompt
    assert "E2" in prompt
    assert "external information" in prompt