from src.v2.rag.chunking import TextChunker
from src.v2.rag.ingestion import DocumentIngestor


class FakeRetriever:
    def __init__(self):
        self.added_evidence = []

    def add_documents(self, evidence):
        self.added_evidence.extend(evidence)


def test_ingest_text_creates_evidence_chunks():
    retriever = FakeRetriever()
    chunker = TextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    ingestor = DocumentIngestor(
        retriever=retriever,
        chunker=chunker,
    )

    evidence = ingestor.ingest_text(
        text=(
            "Artificial intelligence is a field of computer science. "
            "Machine learning is a subset of artificial intelligence."
        ),
        document_id="test_doc",
        source="test",
    )

    assert evidence
    assert len(evidence) == len(retriever.added_evidence)

    for index, item in enumerate(evidence):
        assert item.document_id == "test_doc"
        assert item.chunk_id == f"chunk_{index}"
        assert item.evidence_id == f"test_doc_chunk_{index}"
        assert item.content
        assert item.source == "test"


def test_ingest_text_rejects_empty_document():
    retriever = FakeRetriever()

    ingestor = DocumentIngestor(
        retriever=retriever,
    )

    try:
        ingestor.ingest_text(
            text="",
            document_id="test_doc",
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "empty" in str(exc).lower()