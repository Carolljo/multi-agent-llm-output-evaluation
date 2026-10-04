from src.v2.memory.retriever import AnswerMemoryRetriever
from src.v2.memory.store import ChromaAnswerMemory
from src.v2.models.schemas import Claim, Evidence, ReferenceAnswer, ReferenceValidation


class FakeEmbeddingModel:
    """Deterministic embeddings for memory tests; no Ollama dependency."""

    def embed(self, text: str) -> list[float]:
        text = text.lower()
        if "machine learning" in text:
            return [1.0, 0.0, 0.0]
        if "deep learning" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def build_reference():
    return ReferenceAnswer(
        answer="Machine learning learns patterns from data.",
        claims=[
            Claim(
                claim_id="C1",
                text="Machine learning learns patterns from data.",
                evidence_refs=["E1"],
            )
        ],
        evidence_refs=["E1"],
        confidence=0.95,
    )


def build_validation(status="valid"):
    return ReferenceValidation(
        status=status,
        score=0.96,
        confidence=0.94,
        supported_claims=["C1"],
        reasoning="The reference is supported by the evidence.",
    )


def test_verified_answer_is_stored_and_retrieved(tmp_path):
    store = ChromaAnswerMemory(
        persist_directory=str(tmp_path),
        collection_name="answer_memory_test",
        embedding_model=FakeEmbeddingModel(),
    )

    memory = store.save(
        question="What is machine learning?",
        reference_answer=build_reference(),
        validation=build_validation(),
        evidence=[
            Evidence(
                evidence_id="E1",
                document_id="doc1",
                chunk_id="chunk1",
                content="Machine learning learns patterns from data.",
            )
        ],
    )

    assert memory.validation_status == "valid"
    assert memory.version == 1

    retriever = AnswerMemoryRetriever(
        store=store,
        min_similarity=0.85,
    )
    matches = retriever.search("Explain machine learning")

    assert len(matches) == 1
    assert matches[0].memory.answer == "Machine learning learns patterns from data."
    assert matches[0].similarity >= 0.85


def test_invalid_reference_cannot_be_stored(tmp_path):
    store = ChromaAnswerMemory(
        persist_directory=str(tmp_path),
        collection_name="answer_memory_invalid_test",
        embedding_model=FakeEmbeddingModel(),
    )

    invalid = build_validation(status="invalid")

    try:
        store.save(
            question="What is machine learning?",
            reference_answer=build_reference(),
            validation=invalid,
            evidence=[],
        )
    except ValueError as exc:
        assert "Only valid reference answers" in str(exc)
    else:
        raise AssertionError("Invalid reference was stored")

    assert store.collection.count() == 0


def test_unrelated_question_does_not_hit_memory(tmp_path):
    store = ChromaAnswerMemory(
        persist_directory=str(tmp_path),
        collection_name="answer_memory_miss_test",
        embedding_model=FakeEmbeddingModel(),
    )

    store.save(
        question="What is machine learning?",
        reference_answer=build_reference(),
        validation=build_validation(),
        evidence=[],
    )

    retriever = AnswerMemoryRetriever(
        store=store,
        min_similarity=0.85,
    )

    assert retriever.search("What is deep learning?") == []


def test_same_question_increments_version(tmp_path):
    store = ChromaAnswerMemory(
        persist_directory=str(tmp_path),
        collection_name="answer_memory_version_test",
        embedding_model=FakeEmbeddingModel(),
    )

    kwargs = {
        "question": "What is machine learning?",
        "reference_answer": build_reference(),
        "validation": build_validation(),
        "evidence": [],
    }

    first = store.save(**kwargs)
    second = store.save(**kwargs)

    assert first.version == 1
    assert second.version == 2
    assert store.collection.count() == 1
