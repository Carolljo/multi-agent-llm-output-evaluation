
from sentence_transformers import CrossEncoder

from src.v2.models.schemas import Evidence


class CrossEncoderReranker:
    """Rerank evidence by jointly evaluating each query-chunk pair."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )

    def rerank(
        self,
        query: str,
        evidence: list[Evidence],
        top_k: int = 5,
    ) -> list[Evidence]:
        """Return evidence ordered by cross-encoder relevance."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not query.strip() or not evidence:
            return []

        pairs = [
            (query, item.content)
            for item in evidence
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(evidence, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results: list[Evidence] = []

        for item, score in ranked[:top_k]:
            reranked_item = item.model_copy(deep=True)

            # Keep the raw model score separate from the existing
            # relevance_score, which has a [0, 1] constraint.
            reranked_item.metadata = {
                **reranked_item.metadata,
                "reranker_score": float(score),
            }

            results.append(reranked_item)

        return results

