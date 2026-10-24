
import pytest
from pydantic import ValidationError

from src.v2.evidence.quality import (
    EvidenceQuality,
    EvidenceQualityScorer,
)
from src.v2.models.schemas import Evidence


@pytest.fixture
def scorer():
    return EvidenceQualityScorer()


def make_evidence(**overrides):
    data = {
        "evidence_id": "E1",
        "document_id": "doc1",
        "chunk_id": "chunk1",
        "content": "Machine learning learns patterns from data.",
        "source": "ml.txt",
        "relevance_score": 0.90,
        "metadata": {},
    }
    data.update(overrides)
    return Evidence(**data)


def test_score_uses_expected_weighted_formula(scorer):
    evidence = make_evidence(
        relevance_score=1.0,
        source="ml.txt",
        metadata={
            "source_quality": 1.0,
            "freshness_score": 1.0,
            "consistency_score": 1.0,
        },
    )

    result = scorer.score(evidence)

    assert result.overall_score == pytest.approx(1.0)
    assert result.quality_level == "high"


def test_missing_relevance_defaults_to_zero(scorer):
    evidence = make_evidence(relevance_score=None)

    result = scorer.score(evidence)

    assert result.relevance == 0.0


def test_source_quality_defaults_to_point_eight_when_source_exists(scorer):
    result = scorer.score(make_evidence(source="known.txt"))

    assert result.source_quality == pytest.approx(0.80)


def test_source_quality_defaults_to_point_five_without_source(scorer):
    result = scorer.score(make_evidence(source=None))

    assert result.source_quality == pytest.approx(0.50)


@pytest.mark.parametrize(
    ("metadata_key", "expected_field"),
    [
        ("source_quality", "source_quality"),
        ("freshness_score", "freshness"),
        ("consistency_score", "consistency"),
    ],
)
def test_explicit_metadata_scores_are_clamped(
    scorer,
    metadata_key,
    expected_field,
):
    evidence = make_evidence(
        metadata={metadata_key: 1.7}
    )

    result = scorer.score(evidence)

    assert getattr(result, expected_field) == 1.0


def test_invalid_metadata_score_uses_default(scorer):
    evidence = make_evidence(
        metadata={
            "source_quality": "invalid",
            "freshness_score": "invalid",
            "consistency_score": "invalid",
        }
    )

    result = scorer.score(evidence)

    assert result.source_quality == pytest.approx(0.80)
    assert result.freshness == pytest.approx(0.50)
    assert result.consistency == pytest.approx(0.50)


def test_quality_level_boundaries(scorer):
    high = scorer.score(
        make_evidence(
            relevance_score=1.0,
            metadata={
                "source_quality": 1.0,
                "freshness_score": 1.0,
                "consistency_score": 1.0,
            },
        )
    )

    low = scorer.score(
        make_evidence(
            relevance_score=0.0,
            source=None,
            metadata={
                "source_quality": 0.0,
                "freshness_score": 0.0,
                "consistency_score": 0.0,
            },
        )
    )

    assert high.overall_score == pytest.approx(1.0)
    assert high.quality_level == "high"

    assert low.overall_score == pytest.approx(0.0)
    assert low.quality_level == "low"


def test_score_many_preserves_input_order(scorer):
    evidence = [
        make_evidence(evidence_id="E1"),
        make_evidence(evidence_id="E2"),
    ]

    results = scorer.score_many(evidence)

    assert [item.evidence_id for item in results] == ["E1", "E2"]


@pytest.mark.parametrize(
    "field",
    [
        "relevance",
        "source_quality",
        "freshness",
        "consistency",
        "overall_score",
    ],
)
def test_evidence_quality_rejects_out_of_range_scores(field):
    data = {
        "evidence_id": "E1",
        "relevance": 0.5,
        "source_quality": 0.5,
        "freshness": 0.5,
        "consistency": 0.5,
        "overall_score": 0.5,
        "quality_level": "medium",
        "reasoning": "Test",
    }

    data[field] = 1.1

    with pytest.raises(ValidationError):
        EvidenceQuality(**data)

