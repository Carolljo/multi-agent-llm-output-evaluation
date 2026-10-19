
import re
from collections.abc import Iterable

from rank_bm25 import BM25Okapi

from src.v2.models.schemas import Evidence


class BM25Retriever:
    """Retrieve evidence chunks using BM25 keyword ranking."""

    def __init__(self) -> None:
        self._evidence: dict[str, Evidence] = {}
        self._tokenized_documents: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._indexed_ids: list[str] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize text into lowercase word tokens."""
        return re.findall(r"\b\w+\b", text.lower())

    def add_documents(self, evidence: Iterable[Evidence]) -> None:
        """Add or update evidence and rebuild the BM25 index."""

        for item in evidence:
            if not item.content.strip():
                continue

            self._evidence[item.evidence_id] = item

        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the index from the current evidence collection."""

        self._indexed_ids = list(self._evidence)

        self._tokenized_documents = [
            self._tokenize(self._evidence[evidence_id].content)
            for evidence_id in self._indexed_ids
        ]

        if not self._tokenized_documents:
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(self._tokenized_documents)

    
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[Evidence]:
        """Return the highest-ranked evidence for a query."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = self._tokenize(query)

        if not query_tokens or self._bm25 is None:
            return []

        # Ignore query tokens absent from the indexed corpus.
        vocabulary = {
            token
            for document in self._tokenized_documents
            for token in document
        }
        matching_tokens = [
            token for token in query_tokens
            if token in vocabulary
        ]

        if not matching_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Require actual lexical overlap so zero-scoring,
        # unrelated documents are not returned.
        query_terms = set(matching_tokens)
        candidates = []

        for index, document_tokens in enumerate(
            self._tokenized_documents
        ):
            if not query_terms.intersection(document_tokens):
                continue

            candidates.append(
                (index, float(scores[index]))
            )

        # Python's sort is stable, so equal scores preserve
        # the original insertion order.
        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            self._evidence[self._indexed_ids[index]].model_copy(
                deep=True
            )
            for index, _ in candidates[:top_k]
        ]
