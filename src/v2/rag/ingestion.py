from pathlib import Path

from src.v2.models.schemas import Evidence
from src.v2.rag.chunking import TextChunker
from src.v2.rag.retriever import ChromaRetriever


class DocumentIngestor:
    """Convert text documents into evidence chunks and store them in ChromaDB."""

    def __init__(
        self,
        retriever: ChromaRetriever,
        chunker: TextChunker | None = None,
    ):
        self.retriever = retriever
        self.chunker = chunker or TextChunker()

    def ingest_text(
        self,
        text: str,
        document_id: str,
        source: str | None = None,
    ) -> list[Evidence]:
        """Chunk text, create Evidence objects, and store them."""

        if not text.strip():
            raise ValueError("Document text cannot be empty.")

        if not document_id.strip():
            raise ValueError("document_id cannot be empty.")

        chunks = self.chunker.split(text)

        evidence = [
            Evidence(
                evidence_id=f"{document_id}_chunk_{index}",
                document_id=document_id,
                chunk_id=f"chunk_{index}",
                content=chunk,
                source=source,
                metadata={
                    "document_id": document_id,
                    "chunk_index": index,
                },
            )
            for index, chunk in enumerate(chunks)
        ]

        self.retriever.add_documents(evidence)

        return evidence

    def ingest_file(
        self,
        file_path: str | Path,
        document_id: str | None = None,
    ) -> list[Evidence]:
        """Read a text file and ingest it into the vector store."""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        text = path.read_text(encoding="utf-8")

        resolved_document_id = (
            document_id
            or path.stem
        )

        return self.ingest_text(
            text=text,
            document_id=resolved_document_id,
            source=str(path),
        )