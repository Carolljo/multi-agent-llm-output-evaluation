from typing import Literal

from pydantic import BaseModel, Field

from src.v2.models.schemas import Evidence


class EvidenceQuality(BaseModel):
    """Quality assessment for a retrieved evidence item."""

    evidence_id: str

    relevance: float = Field(ge=0.0, le=1.0)
    source_quality: float = Field(ge=0.0, le=1.0)
    freshness: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)

    overall_score: float = Field(ge=0.0, le=1.0)

    quality_level: Literal[
        "high",
        "medium",
        "low",
    ]

    reasoning: str


class EvidenceQualityScorer:
    """
    Deterministically assess retrieved evidence quality.

    This first version deliberately avoids another LLM call.
    It uses retrieval relevance and source metadata so that
    evidence quality assessment remains fast and explainable.
    """

    def score(self, evidence: Evidence) -> EvidenceQuality:
        relevance = (
            evidence.relevance_score
            if evidence.relevance_score is not None
            else 0.0
        )

        source_quality = self._score_source(evidence)
        freshness = self._score_freshness(evidence)
        consistency = self._score_consistency(evidence)

        overall_score = (
            0.40 * relevance
            + 0.25 * source_quality
            + 0.15 * freshness
            + 0.20 * consistency
        )

        overall_score = max(
            0.0,
            min(1.0, overall_score),
        )

        if overall_score >= 0.80:
            quality_level = "high"
        elif overall_score >= 0.50:
            quality_level = "medium"
        else:
            quality_level = "low"

        reasoning = (
            f"Evidence relevance={relevance:.2f}, "
            f"source_quality={source_quality:.2f}, "
            f"freshness={freshness:.2f}, "
            f"consistency={consistency:.2f}. "
            f"Overall quality={overall_score:.2f}."
        )

        return EvidenceQuality(
            evidence_id=evidence.evidence_id,
            relevance=relevance,
            source_quality=source_quality,
            freshness=freshness,
            consistency=consistency,
            overall_score=overall_score,
            quality_level=quality_level,
            reasoning=reasoning,
        )

    def score_many(
        self,
        evidence: list[Evidence],
    ) -> list[EvidenceQuality]:
        """Score multiple evidence items."""

        return [
            self.score(item)
            for item in evidence
        ]

    @staticmethod
    def _score_source(evidence: Evidence) -> float:
        """
        Estimate source quality from available metadata.

        Local documents are treated as known sources.
        Explicit source-quality metadata can override this.
        """

        metadata = evidence.metadata or {}

        explicit_score = metadata.get("source_quality")

        if explicit_score is not None:
            try:
                return max(
                    0.0,
                    min(1.0, float(explicit_score)),
                )
            except (TypeError, ValueError):
                pass

        if evidence.source:
            return 0.80

        return 0.50

    @staticmethod
    def _score_freshness(evidence: Evidence) -> float:
        """
        Estimate freshness from optional metadata.

        If no freshness information exists, use a neutral score.
        """

        metadata = evidence.metadata or {}

        explicit_score = metadata.get("freshness_score")

        if explicit_score is not None:
            try:
                return max(
                    0.0,
                    min(1.0, float(explicit_score)),
                )
            except (TypeError, ValueError):
                pass

        # Current local documents do not yet expose dates.
        # Neutral score avoids pretending we know their age.
        return 0.50

    @staticmethod
    def _score_consistency(evidence: Evidence) -> float:
        """
        Estimate consistency from optional metadata.

        Cross-document consistency will be implemented in a later
        version when multiple evidence items can be compared.
        """

        metadata = evidence.metadata or {}

        explicit_score = metadata.get("consistency_score")

        if explicit_score is not None:
            try:
                return max(
                    0.0,
                    min(1.0, float(explicit_score)),
                )
            except (TypeError, ValueError):
                pass

        # A single isolated chunk cannot establish cross-source
        # consistency, so use a neutral baseline.
        return 0.50