import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

from src.v2.models.schemas import Claim, Evidence, ReferenceAnswer, ReferenceValidation
from src.v2.memory.schemas import VerifiedAnswer


class ChromaAnswerMemory:
    """Persistent ChromaDB store for independently verified reference answers."""

    def __init__(
        self,
        persist_directory: str = "data/v2/vector_store",
        collection_name: str = "verified_answer_memory",
        embedding_model=None,
    ):
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"purpose": "verified_reference_answers", "hnsw:space": "cosine"},
        )
        self.embedding_model = embedding_model

    @staticmethod
    def make_memory_id(question: str) -> str:
        normalized = " ".join(question.lower().split())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"answer_{digest[:24]}"

    def _embed(self, text: str) -> list[float]:
        if self.embedding_model is None:
            raise ValueError("An embedding_model is required for answer memory.")
        return self.embedding_model.embed(text)

    def save(
        self,
        *,
        question: str,
        reference_answer: ReferenceAnswer,
        validation: ReferenceValidation,
        evidence: list[Evidence],
    ) -> VerifiedAnswer:
        """Persist a reference answer only after independent validation succeeds."""

        if validation.status != "valid":
            raise ValueError("Only valid reference answers may be stored in memory.")

        now = datetime.now(timezone.utc).isoformat()
        memory_id = self.make_memory_id(question)

        existing = self.collection.get(ids=[memory_id], include=["metadatas"])
        previous_version = 0
        if existing.get("metadatas"):
            metadata = existing["metadatas"][0]
            if metadata:
                previous_version = int(metadata.get("version", 0))

        memory = VerifiedAnswer(
            memory_id=memory_id,
            question=question,
            answer=reference_answer.answer,
            claims=reference_answer.claims,
            evidence_refs=reference_answer.evidence_refs,
            source_documents=sorted({item.document_id for item in evidence}),
            evidence=evidence,
            validation_score=validation.score,
            validation_confidence=validation.confidence,
            created_at=(
                existing.get("metadatas", [{}])[0].get("created_at", now)
                if existing.get("metadatas") and existing["metadatas"][0]
                else now
            ),
            updated_at=now,
            version=previous_version + 1,
        )

        self.collection.upsert(
            ids=[memory_id],
            documents=[question],
            embeddings=[self._embed(question)],
            metadatas=[self._to_metadata(memory)],
        )
        return memory

    def search(
        self,
        *,
        question: str,
        top_k: int = 3,
        min_similarity: float = 0.85,
    ) -> list[tuple[VerifiedAnswer, float]]:
        """Retrieve verified answers whose semantic similarity clears the threshold."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0 and 1")

        if self.collection.count() == 0:
            return []

        result: dict[str, Any] = self.collection.query(
            query_embeddings=[self._embed(question)],
            n_results=min(top_k, self.collection.count()),
            where={"validation_status": "valid"},
        )

        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches: list[tuple[VerifiedAnswer, float]] = []
        for index, memory_id in enumerate(ids):
            metadata = metadatas[index] or {}
            distance = distances[index] if index < len(distances) else None
            similarity = (
                max(0.0, min(1.0, 1.0 - float(distance)))
                if distance is not None
                else 0.0
            )

            if similarity < min_similarity:
                continue

            matches.append((self._from_metadata(memory_id, metadata), similarity))

        return matches

    @staticmethod
    def _to_metadata(memory: VerifiedAnswer) -> dict[str, str | int]:
        return {
            "question": memory.question,
            "answer": memory.answer,
            "claims": json.dumps([claim.model_dump() for claim in memory.claims]),
            "evidence_refs": json.dumps(memory.evidence_refs),
            "source_documents": json.dumps(memory.source_documents),
            "evidence": json.dumps([item.model_dump() for item in memory.evidence]),
            "validation_status": memory.validation_status,
            "validation_score": str(memory.validation_score),
            "validation_confidence": str(memory.validation_confidence),
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
            "version": memory.version,
        }

    @staticmethod
    def _from_metadata(memory_id: str, metadata: dict[str, Any]) -> VerifiedAnswer:
        return VerifiedAnswer(
            memory_id=memory_id,
            question=metadata["question"],
            answer=metadata["answer"],
            claims=[Claim.model_validate(item) for item in json.loads(metadata["claims"])],
            evidence_refs=json.loads(metadata["evidence_refs"]),
            source_documents=json.loads(metadata["source_documents"]),
            evidence=[Evidence.model_validate(item) for item in json.loads(metadata["evidence"])],
            validation_status="valid",
            validation_score=float(metadata["validation_score"]),
            validation_confidence=float(metadata["validation_confidence"]),
            created_at=metadata["created_at"],
            updated_at=metadata["updated_at"],
            version=int(metadata["version"]),
        )
