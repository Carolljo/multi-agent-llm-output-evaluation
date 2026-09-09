import json
from unittest.mock import Mock

import pytest

from src.v2.claims.extractor import ClaimExtractor
from src.v2.models.schemas import Claim, Evidence, ReferenceAnswer


def make_evidence():
    return [
        Evidence(
            evidence_id="E1",
            document_id="doc1",
            chunk_id="chunk1",
            content="RAG retrieves external information before generating a response.",
            source="rag.txt",
        ),
        Evidence(
            evidence_id="E2",
            document_id="doc1",
            chunk_id="chunk2",
            content="Retrieved information provides additional context for the language model.",
            source="rag.txt",
        ),
    ]


def make_reference():
    return ReferenceAnswer(
        answer="RAG retrieves external information and uses it as additional context.",
        claims=[
            Claim(
                claim_id="C1",
                text="RAG retrieves external information before generating a response.",
                evidence_refs=["E1"],
                importance=0.9,
            )
        ],
        evidence_refs=["E1", "E2"],
        confidence=0.95,
    )


def test_extracts_candidate_claims():
    llm_client = Mock()

    llm_client.generate.return_value = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "CAND-1",
                    "text": "RAG retrieves external information.",
                    "evidence_refs": ["E1"],
                    "importance": 0.9,
                },
                {
                    "claim_id": "CAND-2",
                    "text": "RAG improves answer quality.",
                    "evidence_refs": [],
                    "importance": 0.7,
                },
            ]
        }
    )

    extractor = ClaimExtractor(llm_client)

    result = extractor.extract(
        "How does RAG work?",
        "RAG retrieves external information. It improves answer quality.",
        make_reference(),
        make_evidence(),
    )

    assert len(result) == 2
    assert result[0].claim_id == "CAND-1"
    assert result[0].evidence_refs == ["E1"]
    assert result[1].evidence_refs == []

    llm_client.generate.assert_called_once()


def test_rejects_empty_question():
    extractor = ClaimExtractor(Mock())

    with pytest.raises(ValueError, match="Question cannot be empty"):
        extractor.extract(
            "",
            "Some response.",
            make_reference(),
            make_evidence(),
        )


def test_rejects_empty_response():
    extractor = ClaimExtractor(Mock())

    with pytest.raises(ValueError, match="Response cannot be empty"):
        extractor.extract(
            "What is RAG?",
            "",
            make_reference(),
            make_evidence(),
        )


def test_rejects_empty_evidence():
    extractor = ClaimExtractor(Mock())

    with pytest.raises(ValueError, match="At least one evidence item is required"):
        extractor.extract(
            "What is RAG?",
            "RAG retrieves information.",
            make_reference(),
            [],
        )


def test_rejects_invalid_json():
    llm_client = Mock()
    llm_client.generate.return_value = "not valid json"

    extractor = ClaimExtractor(llm_client)

    with pytest.raises(ValueError, match="LLM returned invalid JSON"):
        extractor.extract(
            "What is RAG?",
            "RAG retrieves information.",
            make_reference(),
            make_evidence(),
        )


def test_prompt_contains_candidate_context():
    llm_client = Mock()

    llm_client.generate.return_value = json.dumps(
        {
            "claims": []
        }
    )

    extractor = ClaimExtractor(llm_client)

    extractor.extract(
        "How does RAG work?",
        "RAG retrieves external information.",
        make_reference(),
        make_evidence(),
    )

    call_kwargs = llm_client.generate.call_args.kwargs
    prompt = call_kwargs["prompt"]

    assert "How does RAG work?" in prompt
    assert "RAG retrieves external information." in prompt
    assert "[C1]" in prompt
    assert "[E1]" in prompt
    assert "[E2]" in prompt