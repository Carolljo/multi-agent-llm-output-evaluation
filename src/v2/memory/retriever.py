from src.v2.memory.schemas import AnswerMemoryMatch
from src.v2.memory.store import ChromaAnswerMemory


class AnswerMemoryRetriever:
    """Application-facing semantic retriever for verified answer memory."""

    def __init__(
        self,
        store: ChromaAnswerMemory,
        min_similarity: float = 0.85,
        top_k: int = 3,
    ):
        self.store = store
        self.min_similarity = min_similarity
        self.top_k = top_k

    def search(self, question: str) -> list[AnswerMemoryMatch]:
        matches = self.store.search(
            question=question,
            top_k=self.top_k,
            min_similarity=self.min_similarity,
        )
        return [
            AnswerMemoryMatch(memory=memory, similarity=similarity)
            for memory, similarity in matches
        ]
