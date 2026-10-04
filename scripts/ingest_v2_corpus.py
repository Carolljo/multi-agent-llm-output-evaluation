from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v2.rag.ingestion import DocumentIngestor
from src.v2.rag.retriever import ChromaRetriever


DOCUMENTS_DIR = ROOT / "data" / "v2" / "documents"


def main() -> None:
    retriever = ChromaRetriever(
        persist_directory=str(ROOT / "data" / "v2" / "vector_store"),
        collection_name="project3_v2",
    )

    ingestor = DocumentIngestor(retriever=retriever)

    files = sorted(DOCUMENTS_DIR.glob("*.txt"))

    if not files:
        raise RuntimeError(f"No documents found in {DOCUMENTS_DIR}")

    print(f"Found {len(files)} documents.")

    total_chunks = 0

    for path in files:
        document_id = path.stem

        chunks = ingestor.ingest_file(
            file_path=path,
            document_id=document_id,
        )

        total_chunks += len(chunks)

        print(
            f"Ingested: {path.name} | "
            f"document_id={document_id} | "
            f"chunks={len(chunks)}"
        )

    print()
    print(f"Documents processed: {len(files)}")
    print(f"Total chunks generated: {total_chunks}")
    print(f"Chroma collection count: {retriever.collection.count()}")


if __name__ == "__main__":
    main()