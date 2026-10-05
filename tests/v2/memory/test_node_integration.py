from src.v2.memory.decision import MemoryDecisionPolicy
from src.v2.memory.nodes import answer_memory_node
from src.v2.memory.retriever import AnswerMemoryRetriever
from src.v2.memory.store import ChromaAnswerMemory
from src.v2.models.schemas import (
    Claim,
    EvaluationInput,
    Evidence,
    ReferenceAnswer,
    ReferenceValidation,
)


class FakeEmbeddingModel:
    def embed(self, text: str) -> list[float]:
        text = text.lower()

        if "machine learning" in text:
            return [1.0, 0.0, 0.0]

        if "deep learning" in text:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]


def build_memory(tmp_path):
    store = ChromaAnswerMemory(
        persist_directory=str(tmp_path),
        collection_name="test_verified_answer_memory",
        embedding_model=FakeEmbeddingModel(),
    )

    retriever = AnswerMemoryRetriever(
        store=store,
        min_similarity=0.85,
        top_k=3,
    )

    return store, retriever


def build_reference_answer():
    return ReferenceAnswer(
        answer="Machine learning learns patterns from data.",
        claims=[
            Claim(
                claim_id="claim-1",
                text="Machine learning learns patterns from data.",
                evidence_refs=["evidence-1"],
                importance=1.0,
            )
        ],
        evidence_refs=["evidence-1"],
        confidence=0.95,
    )


def build_validation():
    return ReferenceValidation(
        status="valid",
        score=0.96,
        confidence=0.95,
        supported_claims=["claim-1"],
        unsupported_claims=[],
        missing_information=[],
        reasoning="The reference answer is fully supported by the evidence.",
    )


def build_evidence():
    return [
        Evidence(
            evidence_id="evidence-1",
            document_id="01_rag",
            chunk_id="chunk-1",
            content="Machine learning learns patterns from data.",
            source="01_rag.txt",
            relevance_score=0.95,
        )
    ]


def build_state(question: str):
    return {
        "evaluation_input": EvaluationInput(
            question=question,
            response="Test response.",
        )
    }


def seed_memory(store):
    store.save(
        question="What is machine learning?",
        reference_answer=build_reference_answer(),
        validation=build_validation(),
        evidence=build_evidence(),
    )


def test_real_chroma_memory_reaches_memory_node(tmp_path):
    store, retriever = build_memory(tmp_path)

    seed_memory(store)

    result = answer_memory_node(
        build_state("What is machine learning?"),
        memory_retriever=retriever,
        decision_policy=MemoryDecisionPolicy(),
    )

    assert result["memory_hit"] is True
    assert result["memory_decision"].decision == "use_memory"
    assert result["memory_answer"] == (
        "Machine learning learns patterns from data."
    )
    assert result["memory_similarity"] >= 0.85
    assert result["memory_evidence"]


def test_real_chroma_memory_miss_routes_to_fresh_retrieval(tmp_path):
    store, retriever = build_memory(tmp_path)

    seed_memory(store)

    result = answer_memory_node(
        build_state("What is deep learning?"),
        memory_retriever=retriever,
        decision_policy=MemoryDecisionPolicy(),
    )

    assert result["memory_hit"] is False
    assert result["memory_decision"] is None
    assert result["memory_answer"] is None
    assert result["memory_evidence"] == []