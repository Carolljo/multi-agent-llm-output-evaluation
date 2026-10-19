
from collections.abc import Sequence

from src.v2.models.schemas import Evidence


class ReciprocalRankFusion:
    """Combine ranked evidence lists using Reciprocal Rank Fusion."""

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        self.k = k

    def fuse(
        self,
        *ranked_lists: Sequence[Evidence],
        top_k: int = 5,
    ) -> list[Evidence]:
        """Fuse ranked evidence lists and return the top results."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        scores: dict[str, float] = {}
        evidence_by_id: dict[str, Evidence] = {}

        for ranked_list in ranked_lists:
            seen_in_list: set[str] = set()

            for rank, evidence in enumerate(ranked_list, start=1):
                evidence_id = evidence.evidence_id

                # A single retriever must not reward duplicate IDs
                # by listing the same evidence multiple times.
                if evidence_id in seen_in_list:
                    continue

                seen_in_list.add(evidence_id)

                scores[evidence_id] = scores.get(
                    evidence_id, 0.0
                ) + 1.0 / (self.k + rank)

                # Preserve the first encountered Evidence object.
                evidence_by_id.setdefault(evidence_id, evidence)

        ranked_ids = sorted(
            scores,
            key=lambda evidence_id: scores[evidence_id],
            reverse=True,
        )

        fused_results: list[Evidence] = []

        for evidence_id in ranked_ids[:top_k]:
            evidence = evidence_by_id[evidence_id].model_copy(
                deep=True
            )

            # This score is an RRF score, not a probability.
            # Do not overwrite relevance_score with an
            # uncalibrated value.
            evidence.metadata = {
                **evidence.metadata,
                "rrf_score": scores[evidence_id],
            }

            fused_results.append(evidence)

        return fused_results

