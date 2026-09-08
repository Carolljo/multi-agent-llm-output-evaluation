from src.v2.models.schemas import Evidence
from src.v2.rag.retriever import ChromaRetriever


def test_add_and_search(tmp_path):
    retriever = ChromaRetriever(
        persist_directory=str(tmp_path),
        collection_name="test_collection",
    )

    evidence = [
        Evidence(
            evidence_id="E1",
            document_id="doc1",
            chunk_id="chunk1",
            content=(
                "Retrieval augmented generation retrieves "
                "external information before generating a response."
            ),
            source="rag.txt",
        ),
        Evidence(
            evidence_id="E2",
            document_id="doc1",
            chunk_id="chunk2",
            content=(
                "Machine learning models can learn patterns "
                "from training data."
            ),
            source="ml.txt",
        ),
    ]

    retriever.add_documents(evidence)

    results = retriever.search(
        "How does retrieval augmented generation use external information?",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].evidence_id == "E1"
    assert results[0].content
    assert results[0].relevance_score is not None