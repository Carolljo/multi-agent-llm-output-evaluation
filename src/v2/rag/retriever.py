from pathlib import Path
from typing import Any

import chromadb

from src.v2.models.schemas import Evidence
from src.v2.rag.embeddings import OllamaEmbeddingModel


class ChromaRetriever:
    """Store and retrieve evidence chunks using ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "data/v2/vector_store",
        collection_name: str = "project3_v2",
        embedding_model: OllamaEmbeddingModel | None = None,
    ):
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
        )

        self.embedding_model = (
            embedding_model or OllamaEmbeddingModel()
        )

    def add_documents(
        self,
        evidence: list[Evidence],
    ) -> None:
        """Add evidence chunks to the vector store."""

        if not evidence:
            return

        documents = [item.content for item in evidence]

        embeddings = self.embedding_model.embed_many(documents)

        self.collection.upsert(
            ids=[item.evidence_id for item in evidence],
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": item.document_id,
                    "chunk_id": item.chunk_id,
                    "source": item.source or "",
                }
                for item in evidence
            ],
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[Evidence]:
        """Retrieve the most relevant evidence for a query."""

        query_embedding = self.embedding_model.embed(query)

        results: dict[str, Any] = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieved: list[Evidence] = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        distances = results.get("distances", [[]])[0]

        for index, evidence_id in enumerate(ids):
            metadata = metadatas[index]

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            # Chroma distance is not directly a probability.
            # Convert it into a bounded relevance indicator
            # for our Evidence contract.
            relevance_score = (
                max(0.0, min(1.0, 1.0 - distance))
                if distance is not None
                else None
            )

            retrieved.append(
                Evidence(
                    evidence_id=evidence_id,
                    document_id=metadata["document_id"],
                    chunk_id=metadata["chunk_id"],
                    content=documents[index],
                    source=metadata.get("source") or None,
                    relevance_score=relevance_score,
                    metadata=metadata,
                )
            )

        return retrieved